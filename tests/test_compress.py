"""Spec test (b): closed-form eta^fac upper-bounds measured eta^fac (run at b>=4, since
the closed-form is a first-order bound). Plus the elementwise uniform-quant error bound."""
import torch

from src.compress import QuantBits, compress_layer, quantize_uniform
from src.utils import set_seed


def test_uniform_quant_elementwise_error_bound():
    # ||A - Q_b(A)||_inf <= s/2 = R/(2^b - 1).  (Uniform Quantization Error Model)
    set_seed(0)
    A = torch.randn(50, 30)
    for b in (2, 4, 8):
        Q, R = quantize_uniform(A, b)
        s_half = R / (2 ** b - 1)
        assert float((A - Q).abs().max()) <= s_half + 1e-6, b


def test_closed_form_eta_upper_bounds_measured_linear():
    set_seed(1)
    W = torch.randn(40, 60)  # linear weight (out, in)
    for b in (4, 6, 8):
        cl = compress_layer(W, "linear", k=20, bits=QuantBits.uniform(b))
        assert cl.eta_closedform >= cl.eta_measured - 1e-8, (b, cl.eta_closedform, cl.eta_measured)


def test_closed_form_eta_upper_bounds_measured_conv():
    set_seed(2)
    W = torch.randn(16, 8, 3, 3)  # conv weight (out, in, kh, kw)
    for b in (4, 6, 8):
        cl = compress_layer(W, "conv", k=10, bits=QuantBits.uniform(b))
        assert cl.eta_closedform >= cl.eta_measured - 1e-8, (b, cl.eta_closedform, cl.eta_measured)


def test_truncation_sigma_k1_matches_eckart_young():
    # ||W - W_k||_2 == sigma_{k+1}  (Eckart-Young), checked in matrix space.
    set_seed(3)
    W = torch.randn(30, 50)
    k = 12
    cl = compress_layer(W, "linear", k=k, bits=QuantBits.uniform(8))
    resid = float(torch.linalg.matrix_norm(W - cl.lowrank_weight, ord=2))
    assert abs(resid - cl.sigma_k1) / max(cl.sigma_k1, 1e-9) < 1e-4
