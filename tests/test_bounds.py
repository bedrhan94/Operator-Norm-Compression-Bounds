"""Spec test (a): on a tiny network the bound is NEVER below the measured logit error
under exact H. This is the load-bearing correctness guarantee (success criterion #1)."""
import torch

from src.bounds import (analyze_config, compress_all, gamma_factors,
                        lemma1_recurrence_check, quant_weights)
from src.compress import QuantBits
from src.models import AnalysisModel, VGGConfig, build_vgg
from src.spectral import PowerIterConfig
from src.utils import set_seed


def _tiny_analysis_model():
    set_seed(0)
    net = build_vgg(VGGConfig.tiny()).eval()
    return AnalysisModel.from_sequential(net, input_size=32, in_channels=3)


def test_gamma_factors_suffix_product():
    a = [2.0, 3.0, 5.0]
    # Gamma_0 = a1*a2 = 15, Gamma_1 = a2 = 5, Gamma_2 = 1 (empty)
    assert gamma_factors(a) == [15.0, 5.0, 1.0]


def test_bound_never_violated_under_exact_H_op_decomposed():
    model = _tiny_analysis_model()
    set_seed(7)
    x = torch.randn(24, 3, 32, 32)
    L = model.num_layers
    pic = PowerIterConfig(iters=200, tol=1e-7)

    # Several compression settings, including aggressive ones.
    configs = [
        ([1] * L, [3] * L),
        ([2] * L, [4] * L),
        ([4] * L, [2] * L),
    ]
    for ks, bs in configs:
        ks = [max(1, k) for k in ks]
        bits = [QuantBits.uniform(b) for b in bs]
        res = analyze_config(model, ks, bits, eval_x=x, calib_data=None,
                             mode="op_decomposed", pic=pic)
        # Exact H on the eval set => bound must dominate every measured error.
        assert res.rho.violation_rate == 0.0, (ks, bs, res.rho.rho_max)
        assert res.rho.rho_max <= 1.0 + 1e-6, (ks, bs, res.rho.rho_max)
        assert res.bound_total >= float(res.rho.errors.max()) - 1e-6


def test_lemma1_recurrence_holds_per_layer_per_sample():
    # Validate the proof mechanism: e_i <= L_i||W_i||e_{i-1} + L_i||dW_i|| ||h_hat_{i-1}||
    # must hold for every layer and every sample (Lemma 1).
    model = _tiny_analysis_model()
    set_seed(11)
    x = torch.randn(20, 3, 32, 32)
    L = model.num_layers
    pic = PowerIterConfig(iters=200, tol=1e-7)
    for ks, b in [([1] * L, 3), ([2] * L, 4), ([3] * L, 2)]:
        compressed = compress_all(model, [max(1, k) for k in ks], [QuantBits.uniform(b)] * L)
        chk = lemma1_recurrence_check(model, quant_weights(compressed), x, pic=pic)
        assert chk.total_violations == 0, (ks, b, chk.per_layer_violation_rate)


def test_op_total_also_valid_and_tighter_than_decomposed():
    model = _tiny_analysis_model()
    set_seed(8)
    x = torch.randn(16, 3, 32, 32)
    L = model.num_layers
    bits = [QuantBits.uniform(3) for _ in range(L)]
    ks = [2] * L
    pic = PowerIterConfig(iters=200, tol=1e-7)

    res_dec = analyze_config(model, ks, bits, eval_x=x, mode="op_decomposed", pic=pic)
    res_tot = analyze_config(model, ks, bits, eval_x=x, mode="op_total", pic=pic)
    # op_total uses ||Delta W||_op directly <= sigma_op + eta_op (triangle), so it is
    # rigorous and no looser than the decomposed bound.
    assert res_tot.rho.violation_rate == 0.0
    assert res_tot.bound_total <= res_dec.bound_total + 1e-5
