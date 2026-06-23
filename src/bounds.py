"""The bound machinery: Gamma_i, H_{i-1}, S_i, Theorem 1 / Theorem 2, and rho.

This module composes spectral.py (operator norms) and compress.py (low-rank + quant)
into the network error bound. Each function cites the result it implements.

Bound "modes" for the per-layer ``||Delta W_i||`` term:
  * ``op_decomposed``  : ``||W_i-W_{i,k}||_op + ||W_{i,k}-W_hat_i||_op``  (Theorem 2,
    operator-norm form -- RIGOROUS for conv & linear; default for validity experiments).
  * ``op_total``       : ``||W_i - W_hat_i||_op``                         (Theorem 1, tightest rigorous).
  * ``matrix_measured``: ``sigma_{k+1} + eta_measured``                    (matrix-norm form).
  * ``matrix_closedform``: ``sigma_{k+1} + eta_closedform``                (cheap predictive proxy; Exp C).

Only the ``op_*`` modes are guaranteed ``>= ||Delta W_i||_op``; the matrix modes mix a
reshaped-matrix sigma with a conv operator error and may under-bound k x k convs (this is
exactly the trap that ``op_decomposed`` avoids -- see spectral.py).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .compress import CompressedLayer, QuantBits, compress_layer
from .models import AnalysisModel
from .spectral import PowerIterConfig, difference_operator_norm, operator_norm

VALID_MODES = ("op_decomposed", "op_total", "matrix_measured", "matrix_closedform")


# --------------------------------------------------------------------------- #
# Gamma_i  (Theorem 1)
# --------------------------------------------------------------------------- #
def gamma_factors(a: Sequence[float]) -> List[float]:
    """``Gamma_i = prod_{j=i+1..L} a_j`` (empty product = 1).  (Theorem 1)

    ``a`` is 0-indexed ``[a_0,...,a_{L-1}]`` with ``a_i = L_i ||W_i||_2``.
    Returns ``[Gamma_0,...,Gamma_{L-1}]``; the last is 1 (empty suffix).
    """
    L = len(a)
    gamma = [1.0] * L
    for i in range(L - 2, -1, -1):
        gamma[i] = gamma[i + 1] * a[i + 1]
    return gamma


# --------------------------------------------------------------------------- #
# H_{i-1}  (calibration of activation-norm suprema)
# --------------------------------------------------------------------------- #
@dataclass
class Calibration:
    """Per-layer activation-norm statistics. ``H_max[i]`` upper-bounds ``||h_{i-1}||``."""

    H_max: List[float]          # max_{x in C} ||h_hat_{i-1}(x)||  (the "exact" H on C)
    H_pct: List[float]          # high-percentile variant (e.g. 99.9%)
    percentile: float
    n_samples: int


@torch.no_grad()
def calibrate_H(model: AnalysisModel, weights: Optional[Dict[int, torch.Tensor]],
                data, percentile: float = 99.9, device: Optional[torch.device] = None,
                max_batches: Optional[int] = None) -> Calibration:
    """Estimate ``H_{i-1} = max_{x in C} ||h_hat_{i-1}(x)||`` on a calibration set ``C``.

    Norms are taken on the **compressed** network (``weights`` = quantized weights).
    ``data`` may be a DataLoader, an ``(x, y)`` tuple, or a single ``x`` tensor.
    Also returns the ``percentile``-th percentile variant (sec.1: report a high-pct H).
    """
    device = device or model.device
    per_layer: List[List[np.ndarray]] = [[] for _ in range(model.num_layers)]
    n = 0

    def consume(x: torch.Tensor):
        nonlocal n
        x = x.to(device)
        _, norms = model.run(x, weights=weights, collect_norms=True)
        for j, v in norms.items():
            per_layer[j].append(v.detach().cpu().numpy())
        n += x.shape[0]

    if isinstance(data, torch.Tensor):
        consume(data)
    elif isinstance(data, (tuple, list)) and len(data) >= 1 and isinstance(data[0], torch.Tensor):
        consume(data[0])
    else:  # DataLoader
        for b, batch in enumerate(data):
            if max_batches is not None and b >= max_batches:
                break
            x = batch[0] if isinstance(batch, (tuple, list)) else batch
            consume(x)

    H_max, H_pct = [], []
    for j in range(model.num_layers):
        vals = np.concatenate(per_layer[j]) if per_layer[j] else np.zeros(1)
        H_max.append(float(vals.max()))
        H_pct.append(float(np.percentile(vals, percentile)))
    return Calibration(H_max=H_max, H_pct=H_pct, percentile=percentile, n_samples=n)


# --------------------------------------------------------------------------- #
# Per-layer bound terms
# --------------------------------------------------------------------------- #
@dataclass
class LayerBound:
    index: int
    kind: str
    L: float            # L_i (Lipschitz of phi_i)
    W_norm: float       # ||W_i||_op
    a: float            # a_i = L_i ||W_i||_op
    gamma: float        # Gamma_i
    H_in: float         # H_{i-1} used (exact or percentile)
    S: float            # sensitivity S_i = Gamma_i L_i H_{i-1}
    sigma_op: float     # ||W_i - W_{i,k}||_op
    eta_op: float       # ||W_{i,k} - W_hat_i||_op
    deltaW_op: float    # ||W_i - W_hat_i||_op
    sigma_k1: float     # sigma_{k+1}(W_i)  (matrix)
    eta_measured: float
    eta_closedform: float

    def deltaW_term(self, mode: str) -> float:
        if mode == "op_decomposed":
            return self.sigma_op + self.eta_op
        if mode == "op_total":
            return self.deltaW_op
        if mode == "matrix_measured":
            return self.sigma_k1 + self.eta_measured
        if mode == "matrix_closedform":
            return self.sigma_k1 + self.eta_closedform
        raise ValueError(f"unknown mode {mode!r}; choose from {VALID_MODES}")

    def contribution(self, mode: str) -> float:
        """Layer contribution ``S_i * (delta-term)`` to the network bound (sec.1)."""
        return self.S * self.deltaW_term(mode)


@dataclass
class NetworkBound:
    layers: List[LayerBound]
    mode: str
    H_kind: str   # "exact" | "calibration" | "percentile"

    def total(self, mode: Optional[str] = None) -> float:
        """Theorem 2 bound = sum_i Gamma_i L_i H_{i-1} (sigma_{k+1} + eta)."""
        m = mode or self.mode
        return float(sum(lb.contribution(m) for lb in self.layers))

    def contributions(self, mode: Optional[str] = None) -> List[float]:
        m = mode or self.mode
        return [lb.contribution(m) for lb in self.layers]


def build_layer_bounds(model: AnalysisModel, compressed: Sequence[CompressedLayer],
                       calib: Calibration, use_percentile: bool = False,
                       pic: PowerIterConfig = PowerIterConfig()) -> List[LayerBound]:
    """Assemble per-layer bound terms (operator norms + sensitivities)."""
    specs = model.layer_specs()
    Ls = model.lipschitz()
    # a_i = L_i ||W_i||_op  (operator norm of the ORIGINAL folded weight at its input size)
    W_norms = [operator_norm(s.weight, s.kind, s.input_shape, s.stride, s.padding, pic) for s in specs]
    a = [Ls[i] * W_norms[i] for i in range(len(specs))]
    gammas = gamma_factors(a)
    H = calib.H_pct if use_percentile else calib.H_max

    out: List[LayerBound] = []
    for i, s in enumerate(specs):
        cl = compressed[i]
        sigma_op = difference_operator_norm(s.weight, cl.lowrank_weight, s.kind,
                                            s.input_shape, s.stride, s.padding, pic)
        eta_op = difference_operator_norm(cl.lowrank_weight, cl.quant_weight, s.kind,
                                          s.input_shape, s.stride, s.padding, pic)
        deltaW_op = difference_operator_norm(s.weight, cl.quant_weight, s.kind,
                                             s.input_shape, s.stride, s.padding, pic)
        S = gammas[i] * Ls[i] * H[i]
        out.append(LayerBound(
            index=i, kind=s.kind, L=Ls[i], W_norm=W_norms[i], a=a[i], gamma=gammas[i],
            H_in=H[i], S=S, sigma_op=sigma_op, eta_op=eta_op, deltaW_op=deltaW_op,
            sigma_k1=cl.sigma_k1, eta_measured=cl.eta_measured, eta_closedform=cl.eta_closedform,
        ))
    return out


def total_with_H(layer_bounds: Sequence[LayerBound], H: Sequence[float], mode: str) -> float:
    """Re-evaluate the network bound with a different ``H`` (reuses op-norm terms).

    Lets experiments compare exact-H vs calibration-H bounds without recomputing the
    (expensive) operator norms -- only the ``S_i = Gamma_i L_i H_{i-1}`` factor changes.
    """
    return float(sum(lb.gamma * lb.L * H[i] * lb.deltaW_term(mode)
                     for i, lb in enumerate(layer_bounds)))


@dataclass
class Sensitivities:
    S: List[float]            # S_i = Gamma_i L_i H_{i-1}
    gamma: List[float]
    a: List[float]            # a_i = L_i ||W_i||_op
    W_norm: List[float]
    L: List[float]
    H: List[float]


def compute_sensitivities(model: AnalysisModel, calib: Calibration,
                          use_percentile: bool = False,
                          pic: PowerIterConfig = PowerIterConfig()) -> Sensitivities:
    """Sensitivity weights ``S_i = Gamma_i L_i H_{i-1}`` for every layer (sec.1)."""
    specs = model.layer_specs()
    Ls = model.lipschitz()
    W_norms = [operator_norm(s.weight, s.kind, s.input_shape, s.stride, s.padding, pic) for s in specs]
    a = [Ls[i] * W_norms[i] for i in range(len(specs))]
    gammas = gamma_factors(a)
    H = calib.H_pct if use_percentile else calib.H_max
    S = [gammas[i] * Ls[i] * H[i] for i in range(len(specs))]
    return Sensitivities(S=S, gamma=gammas, a=a, W_norm=W_norms, L=Ls, H=list(H))


# --------------------------------------------------------------------------- #
# Measured logit error and rho  (Tightness, sec.1)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def measure_logit_errors(model: AnalysisModel, weights: Dict[int, torch.Tensor],
                         x: torch.Tensor, device: Optional[torch.device] = None) -> np.ndarray:
    """Per-sample ``||z(x) - z_hat(x)||_2`` on the **logits** (not softmax)."""
    device = device or model.device
    x = x.to(device)
    z = model.forward_full(x)
    z_hat = model.forward_compressed(x, weights)
    return (z - z_hat).norm(dim=1).detach().cpu().numpy()


@dataclass
class Lemma1Check:
    per_layer_violation_rate: List[float]  # fraction of samples violating the recurrence at layer i
    per_layer_tightness: List[float]       # median e_i / RHS_i (how tight the recurrence is)
    total_violations: int


def lemma1_recurrence_check(model: AnalysisModel, compressed_weights: Dict[int, torch.Tensor],
                           x: torch.Tensor, pic: PowerIterConfig = PowerIterConfig(),
                           device: Optional[torch.device] = None) -> Lemma1Check:
    """Numerically validate Lemma 1 per layer, per sample (the proof's mechanism, not just the
    final bound):  e_i <= L_i ||W_i||_2 e_{i-1} + L_i ||Delta W_i||_2 ||h_hat_{i-1}||.

    ``e_i = ||h_i - h_hat_i||`` from the full vs compressed activations. Under the lemma every
    sample satisfies it at every layer (violation rate 0). Not wrapped in no_grad: the conv
    operator-norm adjoint (autograd VJP) needs grad enabled; the forwards are no_grad locally."""
    device = device or model.device
    x = x.to(device)
    specs = model.layer_specs()
    Ls = model.lipschitz()
    with torch.no_grad():
        acts_full = model.forward_activations(x, None)
        acts_comp = model.forward_activations(x, compressed_weights)

    def flat_norm(a):
        return a.reshape(a.shape[0], -1).norm(dim=1)

    # e_j = ||h_j - h_hat_j|| for j = 0..L (e_0 = 0 since inputs match).
    e = [flat_norm(acts_full[j] - acts_comp[j]) for j in range(len(acts_full))]
    viol_rate, tightness, total = [], [], 0
    for j, s in enumerate(specs):
        Wn = operator_norm(s.weight, s.kind, s.input_shape, s.stride, s.padding, pic)
        dWn = difference_operator_norm(s.weight, compressed_weights[j], s.kind,
                                       s.input_shape, s.stride, s.padding, pic)
        h_hat_in = flat_norm(acts_comp[j])                  # ||h_hat_{j}|| (input to layer j)
        rhs = Ls[j] * Wn * e[j] + Ls[j] * dWn * h_hat_in    # bound on e_{j+1}
        lhs = e[j + 1]
        viol = (lhs > rhs + 1e-4).float()
        viol_rate.append(float(viol.mean()))
        total += int(viol.sum())
        ratio = (lhs / rhs.clamp_min(1e-30)).cpu().numpy()
        tightness.append(float(np.median(ratio)))
    return Lemma1Check(per_layer_violation_rate=viol_rate, per_layer_tightness=tightness,
                       total_violations=total)


@dataclass
class RhoStats:
    rho: np.ndarray            # per-sample rho(x) = ||z-z_hat|| / bound
    errors: np.ndarray         # per-sample ||z-z_hat||
    bound: float               # the scalar Theorem-2 bound
    violation_rate: float      # fraction with ||z-z_hat|| > bound
    rho_mean: float
    rho_median: float
    rho_max: float


def rho_statistics(errors: np.ndarray, bound: float) -> RhoStats:
    """Tightness ratio ``rho = error / bound`` (sec.1). Under an exact bound, ``0<=rho<=1``;
    rho~1 tight, rho~0 loose. ``violation_rate`` = fraction of inputs exceeding the bound."""
    bound = max(bound, 1e-30)
    rho = errors / bound
    return RhoStats(
        rho=rho, errors=errors, bound=bound,
        violation_rate=float((errors > bound).mean()),
        rho_mean=float(rho.mean()), rho_median=float(np.median(rho)), rho_max=float(rho.max()),
    )


# --------------------------------------------------------------------------- #
# High-level orchestration
# --------------------------------------------------------------------------- #
@dataclass
class ConfigResult:
    """Everything needed to log one (k_i, b_i) configuration to CSV."""

    ks: List[int]
    bits: List[int]
    bound_total: float
    bound_per_layer: List[float]
    rho: RhoStats
    layer_bounds: List[LayerBound]
    mode: str
    H_kind: str


def compress_all(model: AnalysisModel, ks: Sequence[int],
                 bits: Sequence[QuantBits]) -> List[CompressedLayer]:
    """Compress every layer at its ``(k_i, bits_i)``."""
    specs = model.layer_specs()
    return [compress_layer(s.weight, s.kind, ks[i], bits[i], index=i) for i, s in enumerate(specs)]


def quant_weights(compressed: Sequence[CompressedLayer]) -> Dict[int, torch.Tensor]:
    return {cl.index: cl.quant_weight for cl in compressed}


def analyze_config(model: AnalysisModel, ks: Sequence[int], bits: Sequence[QuantBits],
                   eval_x: torch.Tensor, calib_data=None, mode: str = "op_decomposed",
                   percentile: float = 99.9, use_percentile_H: bool = False,
                   pic: PowerIterConfig = PowerIterConfig(),
                   device: Optional[torch.device] = None) -> ConfigResult:
    """End-to-end analysis of one configuration.

    If ``calib_data`` is None, H is taken on ``eval_x`` itself -> **exact H** on the eval
    set, so the bound is guaranteed ``>=`` every measured error (used by the unit test).
    Otherwise H is calibrated on ``calib_data`` (may under-bound -> violations reported).
    """
    device = device or model.device
    compressed = compress_all(model, ks, bits)
    weights = quant_weights(compressed)

    if calib_data is None:
        calib = calibrate_H(model, weights, eval_x, percentile=percentile, device=device)
        H_kind = "percentile" if use_percentile_H else "exact"
    else:
        calib = calibrate_H(model, weights, calib_data, percentile=percentile, device=device)
        H_kind = "percentile" if use_percentile_H else "calibration"

    layer_bounds = build_layer_bounds(model, compressed, calib,
                                      use_percentile=use_percentile_H, pic=pic)
    nb = NetworkBound(layers=layer_bounds, mode=mode, H_kind=H_kind)
    bound_total = nb.total()
    errors = measure_logit_errors(model, weights, eval_x, device=device)
    rho = rho_statistics(errors, bound_total)

    return ConfigResult(
        ks=list(ks), bits=[b.bU for b in bits], bound_total=bound_total,
        bound_per_layer=nb.contributions(), rho=rho, layer_bounds=layer_bounds,
        mode=mode, H_kind=H_kind,
    )
