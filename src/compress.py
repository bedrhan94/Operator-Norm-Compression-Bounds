"""Low-rank truncation + factor-wise uniform quantization (FPGA-style).

Implements, per layer:
  * truncated SVD  ``W_{i,k} = U_k Sigma_k V_k^T``  (Eckart-Young: ``||W - W_k||_2 = sigma_{k+1}``),
  * the symmetric uniform quantizer ``Q_b``,
  * factor-wise quantization ``W_hat = Q_bU(U_k) Q_bSigma(Sigma_k) Q_bV(V_k)^T``,
  * the **measured** factor-quant error ``eta = ||U Sigma V^T - Q(U)Q(Sigma)Q(V)^T||_2``,
  * its **closed-form** upper bound (first-order; see :func:`closed_form_eta`).

Reshape convention for conv: weight ``(out, in, kh, kw)`` <-> matrix ``(out, in*kh*kw)``.
The reshaped-matrix SVD is what *builds* the low-rank tensor and the FPGA factors; the
rigorous operator-norm terms used in Theorem 2 are computed separately in bounds.py by
power-iterating the difference *operators*.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class QuantBits:
    """Per-factor bit-widths (U, Sigma, V)."""

    bU: int
    bS: int
    bV: int

    @classmethod
    def uniform(cls, b: int) -> "QuantBits":
        return cls(b, b, b)


def quantize_uniform(A: torch.Tensor, bits: int) -> Tuple[torch.Tensor, float]:
    """Symmetric uniform quantizer ``Q_b``.  (Uniform Quantization Error Model, sec.1)

    Range ``R = max|A|``, step ``s = 2R/(2^b - 1)``; round to the nearest multiple of
    ``s`` in ``[-R, R]``. Guarantees elementwise error ``<= s/2 = R/(2^b - 1)``.
    Returns ``(Q_b(A), R)``. ``bits<=0`` or all-zero ``A`` is a no-op (R=0).
    """
    R = float(A.abs().max())
    if bits <= 0 or R == 0.0:
        return A.clone(), R
    s = 2.0 * R / (2 ** bits - 1)
    q = torch.round(A / s) * s
    q = torch.clamp(q, -R, R)
    return q, R


def _to_matrix(weight: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "conv":
        return weight.reshape(weight.shape[0], -1)
    return weight  # linear already 2D


def _to_weight(matrix: torch.Tensor, weight_shape: torch.Size, kind: str) -> torch.Tensor:
    if kind == "conv":
        return matrix.reshape(weight_shape)
    return matrix


def closed_form_eta(sigma1: float, m: int, n: int, k: int, bits: QuantBits,
                    R_U: float, R_S: float, R_V: float) -> float:
    """Closed-form upper bound on ``eta^fac`` (sec.1, "Closed-form eta bound").

    ``eta <= sigma_1 * ( sqrt(m k) R_U/(2^bU-1) + sqrt(n k) R_V/(2^bV-1) )
            + sqrt(k) R_Sigma/(2^bSigma-1)``  with  N_U=mk, N_V=nk, N_Sigma=k.

    NOTE (approximation, logged per spec): this is the *first-order* bound -- it drops
    second/third-order cross terms (Q(U)-U)(Q(Sigma)-Sigma)... and uses ``||U||=||V||=1``,
    ``||Sigma||=sigma_1``. It upper-bounds the measured matrix ``eta`` whenever quant
    errors are small (validated at b>=4 in tests/test_compress.py).
    """
    term_u = math.sqrt(m * k) * R_U / (2 ** bits.bU - 1)
    term_v = math.sqrt(n * k) * R_V / (2 ** bits.bV - 1)
    term_s = math.sqrt(k) * R_S / (2 ** bits.bS - 1)
    return sigma1 * (term_u + term_v) + term_s


@dataclass
class CompressedLayer:
    """Result of compressing one layer at rank ``k`` and bit-widths ``bits``."""

    index: int
    kind: str
    m: int
    n: int
    k: int
    bits: QuantBits
    lowrank_weight: torch.Tensor   # W_{i,k}  (full-precision rank-k, conv/linear shape)
    quant_weight: torch.Tensor     # W_hat_i  (factor-quantized, conv/linear shape)
    sigma1: float                  # sigma_1(W_i)        (matrix)
    sigma_k1: float                # sigma_{k+1}(W_i)    (matrix, Eckart-Young truncation)
    eta_measured: float            # ||U Sigma V^T - Q(U)Q(Sigma)Q(V)^T||_2 (matrix)
    eta_closedform: float          # closed-form upper bound on the above
    R_U: float
    R_S: float
    R_V: float

    def memory_cost(self) -> float:
        """Memory proxy ``C_i = b * k * (m + n + 1)`` (uses bU for the single-b case)."""
        b = self.bits.bU
        return b * self.k * (self.m + self.n + 1)

    def bop_cost(self, b_act: int) -> float:
        """BOP proxy ``BOP_i ∝ k (m + n) b b_act``."""
        b = self.bits.bU
        return self.k * (self.m + self.n) * b * b_act


def compress_layer(weight: torch.Tensor, kind: str, k: int, bits: QuantBits,
                   index: int = 0) -> CompressedLayer:
    """Truncate to rank ``k`` and factor-quantize one layer's weight tensor."""
    weight_shape = weight.shape
    M = _to_matrix(weight, kind)
    m, n = M.shape
    rank_max = min(m, n)
    k = int(max(1, min(k, rank_max)))

    # Full SVD (matrices here are small: <= a few thousand on a side).
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)   # U:(m,r) S:(r) Vh:(r,n)
    V = Vh.t()                                             # (n, r)
    sigma1 = float(S[0]) if S.numel() > 0 else 0.0
    sigma_k1 = float(S[k]) if k < S.numel() else 0.0      # sigma_{k+1} (matrix)

    U_k = U[:, :k].contiguous()       # (m, k)
    s_k = S[:k].contiguous()          # (k,)
    V_k = V[:, :k].contiguous()       # (n, k)

    lowrank_matrix = (U_k * s_k.unsqueeze(0)) @ V_k.t()    # U_k diag(s_k) V_k^T  -> (m, n)

    # Factor-wise quantization: quantize each factor independently, then recombine.
    Qu, R_U = quantize_uniform(U_k, bits.bU)
    Qs, R_S = quantize_uniform(s_k, bits.bS)
    Qv, R_V = quantize_uniform(V_k, bits.bV)
    quant_matrix = (Qu * Qs.unsqueeze(0)) @ Qv.t()         # (m, n)

    eta_measured = float(torch.linalg.matrix_norm(lowrank_matrix - quant_matrix, ord=2))
    eta_cf = closed_form_eta(sigma1, m, n, k, bits, R_U, R_S, R_V)

    return CompressedLayer(
        index=index, kind=kind, m=m, n=n, k=k, bits=bits,
        lowrank_weight=_to_weight(lowrank_matrix, weight_shape, kind),
        quant_weight=_to_weight(quant_matrix, weight_shape, kind),
        sigma1=sigma1, sigma_k1=sigma_k1,
        eta_measured=eta_measured, eta_closedform=eta_cf,
        R_U=R_U, R_S=R_S, R_V=R_V,
    )


def singular_values(weight: torch.Tensor, kind: str) -> torch.Tensor:
    """All singular values of the reshaped weight matrix (for rank sweeps / diagnostics)."""
    return torch.linalg.svdvals(_to_matrix(weight, kind))
