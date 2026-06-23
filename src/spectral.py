"""Spectral norms via power iteration, and per-layer Lipschitz constants.

Everything here is an **operator** (2-norm) computation -- the same norm in which
Lemma 1 / Theorem 1 are stated. Crucially, a k x k conv's operator norm is *not* the
2-norm of its reshaped weight matrix; it depends on the input spatial size and the
stride/padding. So conv norms are estimated by power-iterating the actual convolution
operator, with the adjoint obtained exactly via an autograd vector-Jacobian product
(``conv_transpose2d`` is avoided -- it is only the true adjoint for specific
output-padding choices and silently gives a wrong operator otherwise).

Keeping ``||W_i||``, ``||W_i - W_{i,k}||`` and ``||W_{i,k} - W_hat_i||`` all in this
operator norm is what makes Theorem 2 a *rigorous* upper bound for conv layers, not
just for linear ones (see bounds.py).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F


@dataclass
class PowerIterConfig:
    iters: int = 100
    tol: float = 1e-6
    seed: int = 0


def _unit(g: torch.Generator, shape, device, dtype) -> torch.Tensor:
    v = torch.randn(shape, generator=g, device=device, dtype=dtype)
    return v / (v.norm() + 1e-12)


def power_iteration_matrix(W: torch.Tensor, cfg: PowerIterConfig = PowerIterConfig()) -> float:
    """Largest singular value ``sigma_1`` of a 2D matrix ``W`` by power iteration.

    Validated against ``torch.linalg.matrix_norm(W, 2)`` in tests/test_spectral.py.
    """
    assert W.dim() == 2, "power_iteration_matrix expects a 2D tensor"
    device, dtype = W.device, W.dtype
    g = torch.Generator(device="cpu").manual_seed(cfg.seed)
    v = _unit(g, (W.shape[1],), "cpu", dtype).to(device)
    sigma = 0.0
    for _ in range(cfg.iters):
        u = W @ v
        un = u.norm()
        if un < 1e-20:
            return 0.0
        u = u / un
        v = W.t() @ u
        new_sigma = v.norm().item()
        v = v / (new_sigma + 1e-20)
        if abs(new_sigma - sigma) <= cfg.tol * max(new_sigma, 1e-12):
            sigma = new_sigma
            break
        sigma = new_sigma
    return float(sigma)


def _conv_adjoint(weight: torch.Tensor, u: torch.Tensor, input_shape: Tuple[int, int, int],
                  stride, padding) -> torch.Tensor:
    """Exact adjoint A^T u of the convolution operator via autograd VJP.

    For a linear map ``y = conv(x, W)``, grad of ``<y, u>`` w.r.t. ``x`` equals ``A^T u``.
    """
    x = torch.zeros((1,) + tuple(input_shape), device=weight.device,
                    dtype=weight.dtype, requires_grad=True)
    y = F.conv2d(x, weight, None, stride=stride, padding=padding)
    (grad,) = torch.autograd.grad(y, x, grad_outputs=u, retain_graph=False, create_graph=False)
    return grad.detach()


def conv_operator_norm(weight: torch.Tensor, input_shape: Tuple[int, int, int],
                       stride=(1, 1), padding=(1, 1),
                       cfg: PowerIterConfig = PowerIterConfig()) -> float:
    """Operator 2-norm of a conv layer at a given input resolution, by power iteration.

    ``input_shape`` is ``(C_in, H, W)``. The norm depends on ``H, W, stride, padding``;
    iterate at the layer's *actual* feature-map size.
    """
    device, dtype = weight.device, weight.dtype
    g = torch.Generator(device="cpu").manual_seed(cfg.seed)
    v = _unit(g, (1,) + tuple(input_shape), "cpu", dtype).to(device)
    sigma = 0.0
    with torch.no_grad():
        pass
    for _ in range(cfg.iters):
        u = F.conv2d(v, weight, None, stride=stride, padding=padding)  # A v
        un = u.norm()
        if un < 1e-20:
            return 0.0
        u = u / un
        v = _conv_adjoint(weight, u, input_shape, stride, padding)      # A^T u
        new_sigma = v.norm().item()
        v = v / (new_sigma + 1e-20)
        if abs(new_sigma - sigma) <= cfg.tol * max(new_sigma, 1e-12):
            sigma = new_sigma
            break
        sigma = new_sigma
    return float(sigma)


def operator_norm(weight: torch.Tensor, kind: str, input_shape, stride=(1, 1),
                  padding=(0, 0), cfg: PowerIterConfig = PowerIterConfig()) -> float:
    """Dispatch operator-2-norm for a ``"linear"`` or ``"conv"`` weight tensor.

    For ``"linear"`` the operator norm equals the matrix 2-norm (power iteration on the
    2D weight). For ``"conv"`` it is the convolution-operator norm at ``input_shape``.
    """
    if kind == "linear":
        W = weight if weight.dim() == 2 else weight.reshape(weight.shape[0], -1)
        return power_iteration_matrix(W, cfg)
    if kind == "conv":
        return conv_operator_norm(weight, input_shape, stride=stride, padding=padding, cfg=cfg)
    raise ValueError(f"unknown layer kind: {kind}")


def difference_operator_norm(weight_a: torch.Tensor, weight_b: torch.Tensor, kind: str,
                             input_shape, stride=(1, 1), padding=(0, 0),
                             cfg: PowerIterConfig = PowerIterConfig()) -> float:
    """``||A - B||_op`` for two weight tensors of identical shape (same conv geometry).

    Used for the rigorous truncation term ``||W_i - W_{i,k}||_op`` and quant term
    ``||W_{i,k} - W_hat_i||_op`` (and the total ``||Delta W_i||_op``).
    """
    assert weight_a.shape == weight_b.shape
    return operator_norm(weight_a - weight_b, kind, input_shape, stride=stride,
                         padding=padding, cfg=cfg)
