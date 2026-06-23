"""Experiment A -- bound validity (spec sec.3).

Sweep many (k_i, b_i) configurations (uniform + random per-layer). For each, compute the
Theorem-2 predicted bound and the measured logit error ||z - z_hat||_2, then report the
rho distribution and violation rate under BOTH exact H and calibration H.

Success criterion: under exact H the bound is never violated (all scatter points on/below
the y = x line). Violations appear only under calibration / percentile H, and are reported.

    python experiments/experiment_a.py --config configs/experiment_a.yaml
    python experiments/experiment_a.py --smoke
"""
from __future__ import annotations

import argparse
import random
from typing import Dict, List, Sequence, Tuple

import numpy as np

from common import (apply_variant, device_from, get_analysis_model, get_datasets,
                    get_eval_calib, load_experiment_config, pic_from, variant_suffix, write_csv)
from src.bounds import (NetworkBound, build_layer_bounds, calibrate_H, compress_all,
                        measure_logit_errors, quant_weights, rho_statistics, total_with_H)
from src.compress import QuantBits
from src.utils import figures_dir, results_dir, set_seed
from src.viz import scatter_measured_vs_predicted


def _rank_max(model) -> List[int]:
    return [min(s.matrix_shape) for s in model.layer_specs()]


def generate_configs(model, n_configs: int, b_choices: Sequence[int],
                     rank_fracs: Sequence[float], uniform_fraction: float,
                     seed: int) -> List[Tuple[List[int], List[int]]]:
    """Mixed sweep: ``uniform_fraction`` of configs share one (k-frac, b); the rest are
    random per-layer."""
    rng = random.Random(seed)
    rmax = _rank_max(model)
    L = len(rmax)
    configs: List[Tuple[List[int], List[int]]] = []
    n_uniform = int(round(n_configs * uniform_fraction))
    for _ in range(n_uniform):
        rf = rng.choice(rank_fracs)
        b = rng.choice(b_choices)
        ks = [max(1, round(rf * rmax[i])) for i in range(L)]
        configs.append((ks, [b] * L))
    for _ in range(n_configs - n_uniform):
        ks = [max(1, round(rng.choice(rank_fracs) * rmax[i])) for i in range(L)]
        bs = [rng.choice(b_choices) for _ in range(L)]
        configs.append((ks, bs))
    return configs


def run(cfg: Dict, smoke: bool) -> None:
    set_seed(cfg["seed"])
    device = device_from(cfg)
    model, loaded = get_analysis_model(cfg, device)
    print(f"[A] model on {device}; checkpoint loaded={loaded}; layers={model.num_layers}")
    _, test_set = get_datasets(cfg)
    eval_x, _, calib_x = get_eval_calib(cfg, test_set)
    eval_x = eval_x.to(device)
    pic = pic_from(cfg)
    mode = cfg.get("experiment_a", {}).get("mode", "op_decomposed")
    pct = cfg["percentile"]

    ea = cfg.get("experiment_a", {})
    n_configs = 6 if smoke else int(ea.get("n_configs", 40))
    b_choices = ea.get("b_choices", [2, 3, 4, 5, 6, 8])
    rank_fracs = ea.get("rank_fracs", [0.1, 0.25, 0.5, 0.75, 1.0])
    uniform_fraction = float(ea.get("uniform_fraction", 0.5))

    configs = generate_configs(model, n_configs, b_choices, rank_fracs, uniform_fraction,
                               seed=cfg["seed"])

    rows: List[Dict] = []
    bounds_exact, max_errs = [], []
    viol_exact = viol_calib = viol_pct = 0
    for cid, (ks, bs) in enumerate(configs):
        bits = [QuantBits.uniform(b) for b in bs]
        compressed = compress_all(model, ks, bits)
        weights = quant_weights(compressed)
        errors = measure_logit_errors(model, weights, eval_x, device=device)

        # Op-norms computed once via build_layer_bounds (exact-H calibration on eval set).
        calib_exact = calibrate_H(model, weights, eval_x, percentile=pct, device=device)
        layer_bounds = build_layer_bounds(model, compressed, calib_exact, pic=pic)
        nb = NetworkBound(layer_bounds, mode=mode, H_kind="exact")
        b_exact = nb.total()

        # Calibration H on a disjoint subset -> may under-bound.
        calib_real = calibrate_H(model, weights, calib_x, percentile=pct, device=device)
        b_calib = total_with_H(layer_bounds, calib_real.H_max, mode)
        b_pct = total_with_H(layer_bounds, calib_real.H_pct, mode)

        r_exact = rho_statistics(errors, b_exact)
        r_calib = rho_statistics(errors, b_calib)
        r_pct = rho_statistics(errors, b_pct)
        viol_exact += int(r_exact.violation_rate > 0)
        viol_calib += int(r_calib.violation_rate > 0)
        viol_pct += int(r_pct.violation_rate > 0)

        bounds_exact.append(b_exact)
        max_errs.append(float(errors.max()))
        rows.append({
            "config_id": cid, "ks": "|".join(map(str, ks)), "bits": "|".join(map(str, bs)),
            "bound_exactH": b_exact, "bound_calibH": b_calib, "bound_pctH": b_pct,
            "err_max": float(errors.max()), "err_mean": float(errors.mean()),
            "rho_mean_exactH": r_exact.rho_mean, "rho_max_exactH": r_exact.rho_max,
            "violrate_exactH": r_exact.violation_rate,
            "violrate_calibH": r_calib.violation_rate, "violrate_pctH": r_pct.violation_rate,
            "mode": mode,
        })

    sfx = variant_suffix(cfg)
    csv_path = write_csv(results_dir() / f"experiment_a{sfx}.csv", rows)
    overall_viol_exact = float(np.mean([r["violrate_exactH"] for r in rows]))
    fig_path = scatter_measured_vs_predicted(
        max_errs, bounds_exact, figures_dir() / f"experiment_a_validity{sfx}.pdf",
        violation_rate=overall_viol_exact)

    print(f"[A] configs={len(configs)}  exact-H configs with any violation={viol_exact}  "
          f"calib-H={viol_calib}  pct-H={viol_pct}")
    print(f"[A] overall exact-H sample violation rate={overall_viol_exact:.4f} "
          f"(must be 0.0 for success criterion #1)")
    print(f"[A] wrote {csv_path}\n[A] wrote {fig_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--sn", action="store_true", help="spectral-normalized model variant")
    ap.add_argument("--deep", action="store_true", help="deep VGG-16 model variant")
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
    run(cfg, smoke=args.smoke)


if __name__ == "__main__":
    main()
