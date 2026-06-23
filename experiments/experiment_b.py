"""Experiment B -- ranking correlation (spec sec.3).

Compress one layer at a time (fixed moderate k, b), measure the resulting logit-error
increase, and correlate that empirical per-layer sensitivity against the predicted
weight S_i = Gamma_i L_i H_{i-1}. Report Spearman and Kendall coefficients with p-values.

Success criterion: positive, significant rank-correlation.

    python experiments/experiment_b.py --config configs/experiment_b.yaml
    python experiments/experiment_b.py --smoke
"""
from __future__ import annotations

import argparse
from typing import Dict, List

import numpy as np
from scipy import stats

from common import (apply_variant, device_from, get_analysis_model, get_datasets,
                    get_eval_calib, load_experiment_config, pic_from, variant_suffix, write_csv)
from src.bounds import (calibrate_H, compute_sensitivities, measure_logit_errors)
from src.compress import QuantBits, compress_layer
from src.data import tensor_batch
from src.utils import figures_dir, results_dir, set_seed
from src.viz import scatter_sensitivity


def conv_trunk_robustness(model, S, conv_mask, test_set, eval_size, probes, seeds, device):
    """Re-measure the conv-trunk Spearman across (eval-subset seed x probe (k,b)) settings.

    `S` is fixed (deterministic from the trained model); only the empirical sensitivity
    measurement varies, so this is the sampling distribution of the rank-correlation. Returns
    a list of per-setting dicts."""
    specs = model.layer_specs()
    out = []
    for seed in seeds:
        ex, _ = tensor_batch(test_set, eval_size, seed=seed)
        ex = ex.to(device)
        for rf, bb in probes:
            emp = []
            for i, s in enumerate(specs):
                k_i = max(1, round(rf * min(s.matrix_shape)))
                cl = compress_layer(s.weight, s.kind, k_i, QuantBits.uniform(bb), index=i)
                errs = measure_logit_errors(model, {i: cl.quant_weight}, ex, device=device)
                emp.append(float(errs.mean()))
            emp = np.array(emp)
            rc, pc = stats.spearmanr(S[conv_mask], emp[conv_mask])
            ra, pa = stats.spearmanr(S, emp)
            out.append({"seed": seed, "rank_frac": rf, "bits": bb,
                        "rho_conv": float(rc), "p_conv": float(pc),
                        "rho_all": float(ra), "p_all": float(pa)})
    return out


def run(cfg: Dict, smoke: bool, robust: bool = False) -> None:
    set_seed(cfg["seed"])
    device = device_from(cfg)
    model, loaded = get_analysis_model(cfg, device)
    print(f"[B] model on {device}; checkpoint loaded={loaded}; layers={model.num_layers}")
    _, test_set = get_datasets(cfg)
    eval_x, _, calib_x = get_eval_calib(cfg, test_set)
    eval_x = eval_x.to(device)
    pic = pic_from(cfg)
    pct = cfg["percentile"]

    eb = cfg.get("experiment_b", {})
    rank_frac = float(eb.get("rank_frac", 0.5))
    b = int(eb.get("bits", 4))
    agg = eb.get("aggregate", "mean")  # how to summarise per-sample error increase

    specs = model.layer_specs()

    # Predicted sensitivity S_i (H calibrated on the *full* net -> consistent baseline).
    calib_full = calibrate_H(model, weights=None, data=calib_x, percentile=pct, device=device)
    sens = compute_sensitivities(model, calib_full, pic=pic)

    empirical: List[float] = []
    predicted_full: List[float] = []   # S_i * (sigma_{k+1} + eta_i): the full per-layer term
    rows: List[Dict] = []
    for i, s in enumerate(specs):
        k_i = max(1, round(rank_frac * min(s.matrix_shape)))
        cl = compress_layer(s.weight, s.kind, k_i, QuantBits.uniform(b), index=i)
        errors = measure_logit_errors(model, {i: cl.quant_weight}, eval_x, device=device)
        emp = float(errors.mean()) if agg == "mean" else float(errors.max())
        empirical.append(emp)
        pred_full = float(sens.S[i] * (cl.sigma_k1 + cl.eta_measured))
        predicted_full.append(pred_full)
        rows.append({
            "layer": i, "kind": s.kind, "k": k_i, "bits": b,
            "S_i": sens.S[i], "gamma_i": sens.gamma[i], "a_i": sens.a[i],
            "W_norm_i": sens.W_norm[i], "H_im1": sens.H[i],
            "sigma_k1": cl.sigma_k1, "eta_measured": cl.eta_measured,
            "predicted_full_term": pred_full,   # diagnostic: S_i*(sigma_{k+1}+eta)
            "empirical_sensitivity": emp,
        })

    S = np.array(sens.S)
    emp = np.array(empirical)
    pf = np.array(predicted_full)
    kinds = [r["kind"] for r in rows]
    rho_s, p_s = stats.spearmanr(S, emp)
    tau, p_k = stats.kendalltau(S, emp)
    # Diagnostic only: how well the *full* per-layer term predicts the error. Disambiguates
    # "theory wrong" from "S_i is only part of the predictor" if the S_i ranking looks weak.
    rho_full, p_full = stats.spearmanr(pf, emp)
    # Conv-trunk-only: the final classifier (Gamma=1 but direct logit perturbation) inverts the
    # relationship; excluding it tells us whether ranking holds within the conv stack or the
    # looseness is pervasive.
    conv_mask = np.array([k == "conv" for k in kinds])
    if conv_mask.sum() >= 3:
        rho_conv, p_conv = stats.spearmanr(S[conv_mask], emp[conv_mask])
    else:
        rho_conv, p_conv = float("nan"), float("nan")

    sfx = variant_suffix(cfg)
    csv_path = write_csv(results_dir() / f"experiment_b{sfx}.csv", rows)
    labels = [f"L{r['layer']}({r['kind'][0]})" for r in rows]
    fig_path = scatter_sensitivity(S, emp, figures_dir() / f"experiment_b_ranking{sfx}.pdf",
                                   labels=labels, spearman=float(rho_s), kendall=float(tau))

    print(f"[B] PRIMARY  corr(S_i, error):  Spearman rho={rho_s:.4f} (p={p_s:.3g})   "
          f"Kendall tau={tau:.4f} (p={p_k:.3g})")
    print(f"[B] diagnostic corr(S_i*(sigma_k1+eta), error): Spearman rho={rho_full:.4f} "
          f"(p={p_full:.3g})")
    print(f"[B] diagnostic conv-trunk only corr(S_i, error): Spearman rho={rho_conv:.4f} "
          f"(p={p_conv:.3g})  [{int(conv_mask.sum())} conv layers]")
    print(f"[B] wrote {csv_path}\n[B] wrote {fig_path}")

    if robust and int(conv_mask.sum()) >= 3:
        seeds = list(range(31, 31 + (3 if smoke else 8)))
        probes = [(0.25, 4), (0.5, 4), (0.75, 4), (0.5, 3), (0.5, 6)]
        rob = conv_trunk_robustness(model, S, conv_mask, test_set,
                                    cfg["eval_size"], probes, seeds, device)
        rho_c = np.array([r["rho_conv"] for r in rob])
        p_c = np.array([r["p_conv"] for r in rob])
        rob_csv = write_csv(results_dir() / f"experiment_b_robust{sfx}.csv", rob)
        print(f"[B] ROBUSTNESS conv-trunk Spearman over {len(rob)} settings "
              f"({len(seeds)} seeds x {len(probes)} probes):")
        print(f"[B]   rho = {rho_c.mean():.3f} +/- {rho_c.std():.3f}  "
              f"[min {rho_c.min():.3f}, max {rho_c.max():.3f}]")
        print(f"[B]   fraction p<0.05: {float((p_c < 0.05).mean()):.2f}   "
              f"median p: {np.median(p_c):.3g}   fraction rho>0: {float((rho_c > 0).mean()):.2f}")
        print(f"[B] wrote {rob_csv}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--sn", action="store_true", help="spectral-normalized model variant")
    ap.add_argument("--deep", action="store_true", help="deep VGG-16 model variant")
    ap.add_argument("--robust", action="store_true",
                    help="multi-seed x multi-probe robustness sweep of the conv-trunk Spearman")
    args = ap.parse_args()
    cfg = load_experiment_config(args.config)
    if args.sn or args.deep:
        apply_variant(cfg, deep=args.deep, sn=args.sn)
    if args.smoke:
        cfg["model"] = {"arch": "tiny"}
        cfg["data"] = {"synthetic": True, "download": False}
        cfg["checkpoint"] = "checkpoints/vgg_smoke.pt"
        cfg["calibration_size"] = 64
        cfg["eval_size"] = 48
        cfg["power_iter"] = {"iters": 80, "tol": 1e-6}
    run(cfg, smoke=args.smoke, robust=args.robust)


if __name__ == "__main__":
    main()
