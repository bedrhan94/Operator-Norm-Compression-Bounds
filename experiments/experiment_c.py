"""Experiment C -- budget-constrained allocation (spec sec.3).

Solve   min_{k_i,b_i}  sum_i S_i (sigma_{k_i+1} + eta^fac_i)   s.t.  sum_i C_i <= B,
with memory proxy C_i = b_i k_i (m_i + n_i + 1), via **Lagrangian relaxation** over a
per-layer (k, b) candidate grid (each layer chooses independently for a multiplier lambda;
sweeping lambda traces the cost/bound frontier). Baseline: the best *uniform* (k, b) at the
same budget. We sweep B and compare actual test accuracy and logit error.

Why Lagrangian relaxation: the problem is a separable multiple-choice knapsack; for a fixed
lambda each layer's choice decouples, so a 1-D sweep over lambda yields the whole frontier in
O(#layers * #candidates * #lambda) -- no ILP solver dependency needed.

Success criterion: the S_i-guided curve Pareto-dominates uniform across the budget sweep.

    python experiments/experiment_c.py --config configs/experiment_c.yaml
    python experiments/experiment_c.py --smoke
"""
from __future__ import annotations

import argparse
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from common import (apply_variant, device_from, get_analysis_model, get_datasets,
                    get_eval_calib, load_experiment_config, pic_from, variant_suffix, write_csv)
from src.bounds import (NetworkBound, build_layer_bounds, calibrate_H, compress_all,
                        compute_sensitivities, measure_logit_errors, quant_weights)
from src.compress import QuantBits, compress_layer
from src.models import evaluate_accuracy
from src.utils import figures_dir, results_dir, set_seed
from src.viz import pareto_budget, pareto_budget_multi


def empirical_layer_weights(model, eval_x, rank_frac: float, bits: int, device) -> List[float]:
    """Direct per-layer sensitivity: mean ||z-z_hat|| when each layer alone is compressed at a
    reference (k, b). This is a black-box measurement used as an allocation weight; it uses **no
    bound quantity** (no Gamma_i, H_{i-1}, or S_i) and so bypasses the surrogate entirely. It is
    included as a reference allocation, not as a validation of the theoretical S_i."""
    specs = model.layer_specs()
    weights: List[float] = []
    for i, s in enumerate(specs):
        k = max(1, round(rank_frac * min(s.matrix_shape)))
        cl = compress_layer(s.weight, s.kind, k, QuantBits.uniform(bits), index=i)
        err = measure_logit_errors(model, {i: cl.quant_weight}, eval_x, device=device)
        weights.append(float(err.mean()))
    return weights


def layer_candidate_grid(model, S: Sequence[float], rank_fracs: Sequence[float],
                         b_choices: Sequence[int]) -> List[List[Dict]]:
    """Per-layer (k, b) candidates with cost and predicted-bound value.

    Budgeting objective uses S_i * (sigma_{k+1} + eta^fac) with the **measured** factor-quant
    error eta^fac (spec sec.1); the closed-form is only its upper bound and its sqrt(k) growth
    would add a spurious "more rank is worse" pressure, so we use eta_measured here.
    """
    specs = model.layer_specs()
    grids: List[List[Dict]] = []
    for i, s in enumerate(specs):
        m, n = s.matrix_shape
        rmax = min(m, n)
        cands: List[Dict] = []
        seen = set()
        for rf in rank_fracs:
            k = max(1, min(round(rf * rmax), rmax))
            if k in seen:
                continue
            seen.add(k)
            for b in b_choices:
                cl = compress_layer(s.weight, s.kind, k, QuantBits.uniform(b), index=i)
                cost = float(b * k * (m + n + 1))
                value = float(S[i] * (cl.sigma_k1 + cl.eta_measured))
                cands.append({"layer": i, "k": k, "b": b, "cost": cost, "value": value})
        grids.append(cands)
    return grids


def solve_lagrangian(grids: List[List[Dict]], lam: float) -> Tuple[List[Dict], float, float]:
    alloc, tot_cost, tot_val = [], 0.0, 0.0
    for cands in grids:
        best = min(cands, key=lambda c: c["value"] + lam * c["cost"])
        alloc.append(best); tot_cost += best["cost"]; tot_val += best["value"]
    return alloc, tot_cost, tot_val


def guided_frontier(grids: List[List[Dict]], n_lambda: int = 60) -> List[Tuple[float, float, List[Dict]]]:
    """Trace (cost, value, allocation) by sweeping the Lagrange multiplier."""
    lams = np.concatenate([[0.0], np.logspace(-12, 6, n_lambda)])
    pts = []
    for lam in lams:
        alloc, cost, val = solve_lagrangian(grids, float(lam))
        pts.append((cost, val, alloc))
    return pts


def best_under_budget(points: Sequence[Tuple[float, float, List[Dict]]], B: float):
    feas = [p for p in points if p[0] <= B + 1e-9]
    return min(feas, key=lambda p: p[1]) if feas else None


def uniform_candidates(model, S: Sequence[float], rank_fracs: Sequence[float],
                       b_choices: Sequence[int]) -> List[Tuple[float, float, List[Dict]]]:
    """All uniform (same k-frac, same b) allocations as (cost, value, alloc)."""
    specs = model.layer_specs()
    grids = layer_candidate_grid(model, S, rank_fracs, b_choices)
    pts = []
    rmax = [min(s.matrix_shape) for s in specs]
    for rf in rank_fracs:
        for b in b_choices:
            alloc = []
            cost = val = 0.0
            ok = True
            for i, cands in enumerate(grids):
                k = max(1, min(round(rf * rmax[i]), rmax[i]))
                match = [c for c in cands if c["k"] == k and c["b"] == b]
                if not match:
                    ok = False; break
                alloc.append(match[0]); cost += match[0]["cost"]; val += match[0]["value"]
            if ok:
                pts.append((cost, val, alloc))
    return pts


def alloc_to_weights(model, alloc: Sequence[Dict]) -> Tuple[Dict[int, torch.Tensor], List[int], List[int]]:
    specs = model.layer_specs()
    ks = [a["k"] for a in alloc]
    bs = [a["b"] for a in alloc]
    bits = [QuantBits.uniform(b) for b in bs]
    compressed = compress_all(model, ks, bits)
    return quant_weights(compressed), ks, bs


def evaluate_allocation(model, alloc, eval_x, calib_x, acc_loader, device, pic, pct, mode):
    weights, ks, bs = alloc_to_weights(model, alloc)
    errors = measure_logit_errors(model, weights, eval_x, device=device)
    acc = evaluate_accuracy(lambda x: model.forward_compressed(x, weights), acc_loader, device)
    calib = calibrate_H(model, weights, calib_x, percentile=pct, device=device)
    lb = build_layer_bounds(model, compress_all(model, ks, bits=[QuantBits.uniform(b) for b in bs]),
                            calib, pic=pic)
    bound = NetworkBound(lb, mode=mode, H_kind="calibration").total()
    return {"acc": acc, "err_mean": float(errors.mean()), "bound": bound,
            "ks": "|".join(map(str, ks)), "bits": "|".join(map(str, bs))}


def run(cfg: Dict, smoke: bool) -> None:
    set_seed(cfg["seed"])
    device = device_from(cfg)
    model, loaded = get_analysis_model(cfg, device)
    print(f"[C] model on {device}; checkpoint loaded={loaded}; layers={model.num_layers}")
    if not loaded:
        print("[C] WARNING: no trained checkpoint -> accuracy numbers are not meaningful "
              "(logit-error / bound comparison is still valid).")
    train_set, test_set = get_datasets(cfg)
    pic = pic_from(cfg)
    pct = cfg["percentile"]
    mode = cfg.get("experiment_c", {}).get("mode", "op_decomposed")

    ec = cfg.get("experiment_c", {})
    rank_fracs = ec.get("rank_fracs", [0.1, 0.2, 0.35, 0.5, 0.7, 1.0])
    b_choices = ec.get("b_choices", [2, 3, 4, 6, 8])
    n_budgets = 4 if smoke else int(ec.get("n_budgets", 8))
    acc_size = 64 if smoke else int(ec.get("accuracy_size", 2000))

    # Disjoint calibration / scoring split so the empirical-sensitivity weights are NOT measured
    # on the same data the accuracy/logit-error are reported on (no leakage in the headline).
    from torch.utils.data import DataLoader, Subset
    from src.data import tensor_batch
    n = len(test_set)
    half = max(1, n // 2)
    calib_pool = Subset(test_set, list(range(0, half)))          # weights + H calibrated here
    score_pool = Subset(test_set, list(range(half, n)))          # err + accuracy scored here
    calib_x, _ = tensor_batch(calib_pool, cfg["calibration_size"], seed=23)
    eval_x, _ = tensor_batch(score_pool, cfg["eval_size"], seed=11)
    eval_x = eval_x.to(device)
    n_acc = min(acc_size, len(score_pool))
    acc_loader = DataLoader(Subset(score_pool, list(range(n_acc))), batch_size=128)

    # Predicted sensitivities (H on full net, calibration split).
    calib_full = calibrate_H(model, weights=None, data=calib_x, percentile=pct, device=device)
    sens = compute_sensitivities(model, calib_full, pic=pic)

    # Theoretical S_i-guided allocation (the paper's surrogate).
    guided_pts = guided_frontier(layer_candidate_grid(model, sens.S, rank_fracs, b_choices))
    uniform_pts = uniform_candidates(model, sens.S, rank_fracs, b_choices)
    # Empirical-sensitivity-guided allocation: weight = measured per-layer logit-error increase,
    # calibrated on the calibration split. NOTE: this uses NO bound quantity (no Gamma/H/S_i) -- it
    # is a direct measurement, included as a reference that bypasses the surrogate entirely.
    calib_rf = float(ec.get("calib_rank_frac", 0.5))
    calib_b = int(ec.get("calib_bits", 4))
    emp_w = empirical_layer_weights(model, calib_x, calib_rf, calib_b, device)
    emp_pts = guided_frontier(layer_candidate_grid(model, emp_w, rank_fracs, b_choices))

    cost_min = min(p[0] for p in uniform_pts)
    cost_max = max(p[0] for p in uniform_pts)
    budgets = np.linspace(cost_min, cost_max, n_budgets)

    rows: List[Dict] = []
    acc_g, acc_u, acc_e, err_g, err_u, err_e = [], [], [], [], [], []
    used_budgets = []
    for B in budgets:
        g = best_under_budget(guided_pts, B)
        u = best_under_budget(uniform_pts, B)
        e = best_under_budget(emp_pts, B)
        if g is None or u is None or e is None:
            continue
        eg = evaluate_allocation(model, g[2], eval_x, calib_x, acc_loader, device, pic, pct, mode)
        eu = evaluate_allocation(model, u[2], eval_x, calib_x, acc_loader, device, pic, pct, mode)
        ee = evaluate_allocation(model, e[2], eval_x, calib_x, acc_loader, device, pic, pct, mode)
        used_budgets.append(float(B))
        acc_g.append(eg["acc"]); acc_u.append(eu["acc"]); acc_e.append(ee["acc"])
        err_g.append(eg["err_mean"]); err_u.append(eu["err_mean"]); err_e.append(ee["err_mean"])
        rows.append({
            "budget": float(B),
            "Si_guided_acc": eg["acc"], "uniform_acc": eu["acc"], "empirical_acc": ee["acc"],
            "Si_guided_err_mean": eg["err_mean"], "uniform_err_mean": eu["err_mean"],
            "empirical_err_mean": ee["err_mean"],
            "Si_guided_ks": eg["ks"], "uniform_ks": eu["ks"], "empirical_ks": ee["ks"],
            "Si_guided_bits": eg["bits"], "uniform_bits": eu["bits"], "empirical_bits": ee["bits"],
            "Si_guided_cost": g[0], "uniform_cost": u[0], "empirical_cost": e[0],
        })

    sfx = variant_suffix(cfg)
    csv_path = write_csv(results_dir() / f"experiment_c{sfx}.csv", rows)
    acc_series = [("empirical-sensitivity (no bound)", acc_e, "o-", "tab:blue"),
                  ("$S_i$-guided (bound)", acc_g, "^-", "tab:green"),
                  ("uniform", acc_u, "s--", "tab:gray")]
    err_series = [("empirical-sensitivity (no bound)", err_e, "o-", "tab:blue"),
                  ("$S_i$-guided (bound)", err_g, "^-", "tab:green"),
                  ("uniform", err_u, "s--", "tab:gray")]
    fig_acc = pareto_budget_multi(used_budgets, acc_series,
                                  figures_dir() / f"experiment_c_accuracy{sfx}.png", ylabel="Test accuracy")
    fig_err = pareto_budget_multi(used_budgets, err_series,
                                  figures_dir() / f"experiment_c_logiterror{sfx}.png",
                                  ylabel="Mean logit error $\\|z-\\hat z\\|_2$", lower_is_better=True)

    if rows:
        cu = float(np.mean([e >= u - 1e-9 for e, u in zip(acc_e, acc_u)]))
        cg = float(np.mean([e >= g - 1e-9 for e, g in zip(acc_e, acc_g)]))
        gu = float(np.mean([g >= u - 1e-9 for g, u in zip(acc_g, acc_u)]))
        print(f"[C] budgets={len(rows)}  acc-dominance (frac >=):  empirical>=uniform={cu:.2f}  "
              f"empirical>=S_i={cg:.2f}  S_i_guided>=uniform={gu:.2f}")
    print(f"[C] wrote {csv_path}\n[C] wrote {fig_acc}\n[C] wrote {fig_err}")


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
        cfg["power_iter"] = {"iters": 60, "tol": 1e-6}
    run(cfg, smoke=args.smoke)


if __name__ == "__main__":
    main()
