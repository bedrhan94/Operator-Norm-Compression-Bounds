"""Retrain robustness for criterion #2: does the conv-trunk ranking sign survive RETRAINING?

The main experiments use one trained model per dataset; the 8 "seeds" there are eval-subset
reshuffles, not retrainings. This script retrains the deep VGG-16 from scratch with several
seeds per dataset and recomputes the conv-trunk Spearman(S_i, empirical sensitivity), so we can
state whether the CIFAR(+) / SVHN(-) sign is a property of the dataset or of one particular run.

    python experiments/seed_robustness.py            # 3 seeds x {cifar10, svhn}
Writes results/seed_robustness.csv.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.bounds import calibrate_H, compute_sensitivities, measure_logit_errors
from src.compress import QuantBits, compress_layer
from src.data import get_datasets, make_loaders, tensor_batch
from src.models import AnalysisModel, VGGConfig, build_vgg, evaluate_accuracy
from src.spectral import PowerIterConfig
from src.utils import get_device, results_dir, set_seed

EPOCHS = 40
SEEDS = [0, 1, 2]
DATASETS = ["cifar10", "svhn"]
LR = {"cifar10": 0.05, "svhn": 0.05}


def train_one(dataset: str, seed: int, device) -> nn.Sequential:
    set_seed(seed)
    train_set, test_set = get_datasets(root="./data", dataset=dataset, download=True)
    train_loader, test_loader = make_loaders(train_set, test_set, batch_size=128, num_workers=2)
    net = build_vgg(VGGConfig.deep()).to(device)
    opt = torch.optim.SGD(net.parameters(), lr=LR[dataset], momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    loss_fn = nn.CrossEntropyLoss()
    for ep in range(EPOCHS):
        net.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); loss_fn(net(x), y).backward(); opt.step()
        sched.step()
    net.eval()
    acc = evaluate_accuracy(net, test_loader, device)
    return net, acc, test_set


def conv_trunk_spearman(net, test_set, device) -> tuple:
    model = AnalysisModel.from_sequential(net, input_size=32, in_channels=3).to(device)
    pic = PowerIterConfig(iters=150)
    calib_x, _ = tensor_batch(test_set, 512, seed=23)
    eval_x, _ = tensor_batch(test_set, 256, seed=11)
    eval_x = eval_x.to(device)
    calib = calibrate_H(model, weights=None, data=calib_x, percentile=99.9, device=device)
    sens = compute_sensitivities(model, calib, pic=pic)
    specs = model.layer_specs()
    S, emp, is_conv = [], [], []
    for i, s in enumerate(specs):
        k = max(1, round(0.5 * min(s.matrix_shape)))
        cl = compress_layer(s.weight, s.kind, k, QuantBits.uniform(4), index=i)
        err = measure_logit_errors(model, {i: cl.quant_weight}, eval_x, device=device)
        S.append(sens.S[i]); emp.append(float(err.mean())); is_conv.append(s.kind == "conv")
    S, emp, is_conv = np.array(S), np.array(emp), np.array(is_conv)
    rho_c, p_c = stats.spearmanr(S[is_conv], emp[is_conv])
    rho_a, p_a = stats.spearmanr(S, emp)
    return float(rho_c), float(p_c), float(rho_a), float(p_a)


def main():
    device = get_device()
    rows = []
    for ds in DATASETS:
        for seed in SEEDS:
            t0 = time.time()
            net, acc, test_set = train_one(ds, seed, device)
            rho_c, p_c, rho_a, p_a = conv_trunk_spearman(net, test_set, device)
            dt = time.time() - t0
            print(f"[seed-robust] {ds} seed={seed} acc={acc:.4f} conv-trunk rho={rho_c:+.3f} "
                  f"(p={p_c:.3f}) all-layer rho={rho_a:+.3f}  ({dt:.0f}s)", flush=True)
            rows.append({"dataset": ds, "seed": seed, "epochs": EPOCHS, "test_acc": acc,
                         "conv_trunk_rho": rho_c, "conv_trunk_p": p_c, "all_layer_rho": rho_a})
    # summary per dataset
    for ds in DATASETS:
        rc = [r["conv_trunk_rho"] for r in rows if r["dataset"] == ds]
        print(f"[seed-robust] {ds}: conv-trunk rho over {len(rc)} retrains = "
              f"[{min(rc):+.3f}, {max(rc):+.3f}], mean {np.mean(rc):+.3f}, "
              f"{sum(r>0 for r in rc)}/{len(rc)} positive", flush=True)
    import csv
    p = results_dir() / "seed_robustness.csv"
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("[seed-robust] wrote", p, flush=True)


if __name__ == "__main__":
    main()
