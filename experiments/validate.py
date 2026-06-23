"""Extra validations of the claims (spec sec.5), beyond A/B/C:

  1. Lemma 1 mechanism  -- the per-layer error recurrence holds at every layer / sample.
  2. Tightness regime    -- the operator-norm bound is *achievable* (rho->1) for a single
                            operator, and its looseness with depth is exactly singular-subspace
                            misalignment in the product (a controlled explanation of the loose rho).
  3. Adversarial / OOD   -- the bound holds for-all-x (not just test images) under exact H; and
                            clean-calibrated H can be violated by OOD inputs (H coverage matters).

    python experiments/validate.py --config configs/model_deep.yaml --deep
"""
from __future__ import annotations

import argparse
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from common import (apply_variant, device_from, get_analysis_model, get_datasets,
                    get_eval_calib, load_experiment_config, pic_from, variant_suffix, write_csv)
from src.bounds import (build_layer_bounds, calibrate_H, compress_all, lemma1_recurrence_check,
                        measure_logit_errors, quant_weights, total_with_H)
from src.compress import QuantBits, quantize_uniform
from src.utils import results_dir, set_seed


# --------------------------------------------------------------------------- #
# 1. Lemma 1 mechanism on the real network.
# --------------------------------------------------------------------------- #
def check_lemma1(model, eval_x, device, pic) -> Dict:
    L = model.num_layers
    rows = []
    for ks_frac, b in [(0.5, 4), (0.25, 3), (1.0, 2)]:
        ks = [max(1, round(ks_frac * min(s.matrix_shape))) for s in model.layer_specs()]
        comp = compress_all(model, ks, [QuantBits.uniform(b)] * L)
        chk = lemma1_recurrence_check(model, quant_weights(comp), eval_x, pic=pic, device=device)
        rows.append({"rank_frac": ks_frac, "bits": b, "total_violations": chk.total_violations,
                     "max_layer_violrate": max(chk.per_layer_violation_rate),
                     "median_tightness": float(np.median(chk.per_layer_tightness))})
        print(f"[V1] Lemma1 (k~{ks_frac}, b={b}): total_violations={chk.total_violations}  "
              f"max per-layer viol-rate={max(chk.per_layer_violation_rate):.3f}  "
              f"median recurrence tightness e_i/RHS={np.median(chk.per_layer_tightness):.3f}")
    return rows


# --------------------------------------------------------------------------- #
# 2. Tightness regime (synthetic, direct matrix algebra).
# --------------------------------------------------------------------------- #
def check_tightness(device) -> List[Dict]:
    g = torch.Generator(device="cpu").manual_seed(0)
    rows = []
    # (a) single operator: rho = ||dW x|| / (||dW|| ||x||); =1 at the top singular vector.
    W = torch.randn(96, 96, generator=g)
    Wq, _ = quantize_uniform(W, 4)
    dW = Wq - W
    U, S, Vh = torch.linalg.svd(dW)
    v1 = Vh[0]                                   # top right singular vector
    rho_aligned = float((dW @ v1).norm() / (S[0] * v1.norm()))
    xr = torch.randn(96, 256, generator=g)
    rho_rand = ((dW @ xr).norm(dim=0) / (S[0] * xr.norm(dim=0)))
    print(f"[V2] single operator: rho(top-singular input)={rho_aligned:.4f} (=1 => bound ACHIEVED); "
          f"random-input rho median={float(rho_rand.median()):.4f}, max={float(rho_rand.max()):.4f}")
    rows.append({"case": "single_op", "rho_aligned": rho_aligned,
                 "rho_rand_median": float(rho_rand.median()), "rho_rand_max": float(rho_rand.max())})

    # (b) depth: y = W_L...W_1 x, perturb W_1, feed the aligned input. rho decays with depth because
    # the product bound assumes every layer's top singular subspace aligns -- they do not.
    for L in (1, 2, 4, 8):
        Ws = [torch.randn(64, 64, generator=g) for _ in range(L)]
        W1q, _ = quantize_uniform(Ws[0], 4)
        dW1 = W1q - Ws[0]
        Uu, Ss, Vh1 = torch.linalg.svd(dW1)
        x = Vh1[0]                               # aligns the layer-1 perturbation
        delta = dW1 @ x
        for j in range(1, L):
            delta = Ws[j] @ delta
        measured = float(delta.norm())
        bound = float(Ss[0]) * float(np.prod([float(torch.linalg.matrix_norm(Ws[j], 2)) for j in range(1, L)]))
        rho = measured / max(bound, 1e-30)
        print(f"[V2] linear depth L={L}: rho={rho:.4f}  (tight at L=1, decays => compositional looseness)")
        rows.append({"case": f"linear_depth_{L}", "rho_aligned": rho,
                     "rho_rand_median": None, "rho_rand_max": None})
    return rows


# --------------------------------------------------------------------------- #
# 3. Adversarial / OOD inputs -- the bound is 'for all x'.
# --------------------------------------------------------------------------- #
def _fgsm(model, x, y, eps=0.03):
    x = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model.forward_full(x), y)
    (grad,) = torch.autograd.grad(loss, x)
    return (x + eps * grad.sign()).detach()


def check_ood(model, clean_x, clean_y, device, pic) -> List[Dict]:
    L = model.num_layers
    ks = [max(1, round(0.5 * min(s.matrix_shape))) for s in model.layer_specs()]
    comp = compress_all(model, ks, [QuantBits.uniform(4)] * L)
    weights = quant_weights(comp)
    clean_calib = calibrate_H(model, weights, clean_x, device=device)
    lb = build_layer_bounds(model, comp, clean_calib, pic=pic)  # op-norm terms (clean H here)

    ood_sets = {
        "clean": clean_x,
        "gaussian_noise": torch.randn_like(clean_x),
        "scaled_x2": (clean_x * 2.0),
        "fgsm_adv": _fgsm(model, clean_x, clean_y.to(device)),
    }
    rows = []
    for name, ox in ood_sets.items():
        ox = ox.to(device)
        errs = measure_logit_errors(model, weights, ox, device=device)
        exact = calibrate_H(model, weights, ox, device=device)            # exact H over THIS set
        b_exact = total_with_H(lb, exact.H_max, "op_decomposed")
        b_clean = total_with_H(lb, clean_calib.H_max, "op_decomposed")
        v_exact = float((errs > b_exact).mean())
        v_clean = float((errs > b_clean).mean())
        print(f"[V3] OOD={name:14s}  viol-rate exact-H={v_exact:.3f}  clean-H={v_clean:.3f}  "
              f"(maxH this/clean = {max(exact.H_max)/max(clean_calib.H_max):.2f}x)")
        rows.append({"ood": name, "viol_exactH": v_exact, "viol_cleanH": v_clean,
                     "Hmax_ratio_to_clean": max(exact.H_max) / max(clean_calib.H_max)})
    return rows


def run(cfg: Dict) -> None:
    set_seed(cfg["seed"])
    device = device_from(cfg)
    model, loaded = get_analysis_model(cfg, device)
    print(f"[V] model on {device}; checkpoint loaded={loaded}; layers={model.num_layers}")
    _, test_set = get_datasets(cfg)
    eval_x, eval_y, _ = get_eval_calib(cfg, test_set)
    eval_x = eval_x.to(device)
    pic = pic_from(cfg)
    sfx = variant_suffix(cfg)

    print("\n-- 1. Lemma 1 mechanism --")
    r1 = check_lemma1(model, eval_x, device, pic)
    print("\n-- 2. Tightness regime (synthetic) --")
    r2 = check_tightness(device)
    print("\n-- 3. Adversarial / OOD --")
    r3 = check_ood(model, eval_x, eval_y, device, pic)

    write_csv(results_dir() / f"validate_lemma1{sfx}.csv", r1)
    write_csv(results_dir() / f"validate_tightness{sfx}.csv", r2)
    write_csv(results_dir() / f"validate_ood{sfx}.csv", r3)
    print(f"\n[V] wrote validate_*{sfx}.csv to results/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--sn", action="store_true")
    ap.add_argument("--deep", action="store_true")
    args = ap.parse_args()
    cfg = load_experiment_config(args.config)
    if args.sn or args.deep:
        apply_variant(cfg, deep=args.deep, sn=args.sn)
    run(cfg)


if __name__ == "__main__":
    main()
