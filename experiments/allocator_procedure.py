"""Ranking-vs-procedure test for the depth allocation collapse (referee #6).

Raw S_i-guided allocation collapses at depth because S_i spans ~9 orders of magnitude, so the
budget piles onto layer 0. Is the failure in S_i's *ranking* or in the *procedure* (minimizing a
magnitude-weighted sum)? We compare allocators that use only S_i's ORDERING - rank(S_i) and
log(S_i), both with bounded spread - against uniform, raw S_i, and the bound-free empirical weight.
If rank/log-S_i still fail to beat uniform at depth, the ranking is the problem; if they recover,
it was the procedure.

    python experiments/allocator_procedure.py --deep
Writes results/allocator_procedure{suffix}.csv.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from scipy.stats import rankdata
from torch.utils.data import DataLoader, Subset

from common import (apply_variant, device_from, get_analysis_model, get_datasets,
                    load_experiment_config, pic_from, variant_suffix, write_csv)
from experiment_c import (best_under_budget, empirical_layer_weights, evaluate_allocation,
                          guided_frontier, layer_candidate_grid, uniform_candidates)
from src.bounds import calibrate_H, compute_sensitivities
from src.data import tensor_batch
from src.utils import results_dir, set_seed


def run(cfg, smoke):
    set_seed(cfg["seed"])
    device = device_from(cfg)
    model, loaded = get_analysis_model(cfg, device)
    print(f"[AP] model on {device}; loaded={loaded}; layers={model.num_layers}")
    _, test_set = get_datasets(cfg)
    pic = pic_from(cfg); pct = cfg["percentile"]
    ec = cfg.get("experiment_c", {})
    rank_fracs = ec.get("rank_fracs", [0.1, 0.2, 0.35, 0.5, 0.7, 1.0])
    b_choices = ec.get("b_choices", [2, 3, 4, 6, 8])
    n_budgets = 4 if smoke else int(ec.get("n_budgets", 8))
    acc_size = 64 if smoke else int(ec.get("accuracy_size", 2000))

    n = len(test_set); half = max(1, n // 2)
    calib_pool, score_pool = Subset(test_set, range(half)), Subset(test_set, range(half, n))
    calib_x, _ = tensor_batch(calib_pool, cfg["calibration_size"], seed=23)
    eval_x, _ = tensor_batch(score_pool, cfg["eval_size"], seed=11)
    eval_x = eval_x.to(device)
    acc_loader = DataLoader(Subset(score_pool, range(min(acc_size, len(score_pool)))), batch_size=128)

    calib = calibrate_H(model, weights=None, data=calib_x, percentile=pct, device=device)
    S = np.array(compute_sensitivities(model, calib, pic=pic).S)
    emp = np.array(empirical_layer_weights(model, calib_x, 0.5, 4, device))

    # Weight schemes that use only S_i's ORDERING (bounded spread):
    rank_S = rankdata(S)                      # 1..L, preserves order
    log_S = np.log(np.maximum(S, 1e-30))      # compresses 9-order spread, preserves order
    flat = np.ones_like(S)                    # CONTROL: knapsack with no S_i (min sum(sigma+eta))
    schemes = {"uniform": None, "flat": flat, "raw_S": S, "rank_S": rank_S, "log_S": log_S,
               "empirical": emp}

    fronts = {}
    for name, w in schemes.items():
        if name == "uniform":
            fronts[name] = uniform_candidates(model, S, rank_fracs, b_choices)
        else:
            fronts[name] = guided_frontier(layer_candidate_grid(model, list(w), rank_fracs, b_choices))

    cost_min = min(p[0] for p in fronts["uniform"]); cost_max = max(p[0] for p in fronts["uniform"])
    budgets = np.linspace(cost_min, cost_max, n_budgets)

    rows, accs = [], {k: [] for k in schemes}
    for B in budgets:
        picks = {k: best_under_budget(fronts[k], B) for k in schemes}
        if any(v is None for v in picks.values()):
            continue
        row = {"budget": float(B)}
        for k, pk in picks.items():
            ev = evaluate_allocation(model, pk[2], eval_x, calib_x, acc_loader, device, pic, pct,
                                     "op_decomposed")
            accs[k].append(ev["acc"]); row[f"{k}_acc"] = ev["acc"]
        rows.append(row)

    sfx = variant_suffix(cfg)
    path = write_csv(results_dir() / f"allocator_procedure{sfx}.csv", rows)
    order = list(schemes.keys())
    print("[AP] accuracy by budget: " + " / ".join(order))
    for r in rows:
        print("[AP]  B={:>10.0f}  ".format(r["budget"])
              + " / ".join("{:.3f}".format(r[f"{k}_acc"]) for k in order))
    u = np.array(accs["uniform"])
    for k in order:
        if k == "uniform":
            continue
        a = np.array(accs[k])
        print(f"[AP] {k:>9s} >= uniform in {int((a >= u - 1e-9).sum())}/{len(a)} budgets; "
              f"low-budget acc {a[0]:.3f} (uniform {u[0]:.3f})")
    print("[AP] wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--sn", action="store_true")
    ap.add_argument("--deep", action="store_true")
    args = ap.parse_args()
    cfg = load_experiment_config(args.config)
    if args.sn or args.deep:
        apply_variant(cfg, deep=args.deep, sn=args.sn)
    if args.smoke:
        cfg["model"] = {"arch": "tiny"}; cfg["data"] = {"synthetic": True, "download": False}
        cfg["checkpoint"] = "checkpoints/vgg_smoke.pt"; cfg["calibration_size"] = 64
        cfg["eval_size"] = 48; cfg["power_iter"] = {"iters": 60, "tol": 1e-6}
    run(cfg, smoke=args.smoke)


if __name__ == "__main__":
    main()
