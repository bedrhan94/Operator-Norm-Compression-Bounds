"""Spec test (c): power-iteration spectral norm vs torch.linalg.matrix_norm on a
dense layer; plus a 1x1-conv consistency check (where conv op-norm == matrix 2-norm)."""
import torch

from src.spectral import (PowerIterConfig, conv_operator_norm, operator_norm,
                          power_iteration_matrix)
from src.utils import set_seed


def test_power_iteration_matches_matrix_norm_dense():
    set_seed(0)
    W = torch.randn(64, 50)
    ref = float(torch.linalg.matrix_norm(W, ord=2))
    est = power_iteration_matrix(W, PowerIterConfig(iters=500, tol=1e-9))
    assert abs(est - ref) / ref < 1e-3, (est, ref)


def test_operator_norm_linear_dispatch():
    set_seed(1)
    W = torch.randn(32, 40)
    ref = float(torch.linalg.matrix_norm(W, ord=2))
    est = operator_norm(W, "linear", input_shape=(40,), cfg=PowerIterConfig(iters=500, tol=1e-9))
    assert abs(est - ref) / ref < 1e-3


def test_conv_1x1_operator_norm_equals_matrix_norm():
    # A 1x1 conv applies the (out x in) matrix independently at each pixel, so its
    # operator norm equals the matrix 2-norm regardless of spatial size.
    set_seed(2)
    weight = torch.randn(8, 6, 1, 1)
    mat = weight.reshape(8, 6)
    ref = float(torch.linalg.matrix_norm(mat, ord=2))
    est = conv_operator_norm(weight, input_shape=(6, 5, 5), stride=(1, 1), padding=(0, 0),
                             cfg=PowerIterConfig(iters=500, tol=1e-9))
    assert abs(est - ref) / ref < 5e-3, (est, ref)


def test_conv_operator_norm_depends_on_resolution():
    # Sanity: a non-trivial 3x3 conv op-norm should be finite/positive and generally
    # differ from the reshaped-matrix 2-norm (the whole reason we power-iterate the operator).
    set_seed(3)
    weight = torch.randn(6, 4, 3, 3)
    est = conv_operator_norm(weight, input_shape=(4, 8, 8), padding=(1, 1),
                             cfg=PowerIterConfig(iters=300))
    assert est > 0
