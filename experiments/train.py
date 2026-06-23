"""Train the VGG-style CIFAR-10 baseline and save a checkpoint.

    python experiments/train.py --config configs/model.yaml
    python experiments/train.py --smoke          # tiny net + synthetic data, CPU seconds

Trains with a cosine LR schedule and **early stopping (patience)** on test accuracy, and
saves the *best* checkpoint seen (not the last). The checkpoint feeds all three experiments;
A/B only need error propagation (work even untrained), C's accuracy needs a trained model.
"""
from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import torch
import torch.nn as nn

from common import (build_trainable, checkpoint_path, device_from, get_datasets,
                    load_experiment_config, vgg_config_from)
from src.data import make_loaders
from src.models import evaluate_accuracy
from src.utils import set_seed


def train(cfg: dict, epochs: int, smoke: bool, patience: int) -> Path:
    set_seed(cfg["seed"])
    device = device_from(cfg)
    train_cfg = cfg.get("train", {})
    lr = float(train_cfg.get("lr", 0.05))
    wd = float(train_cfg.get("weight_decay", 5e-4))
    batch = int(train_cfg.get("batch_size", 128))
    momentum = float(train_cfg.get("momentum", 0.9))
    min_delta = float(train_cfg.get("min_delta", 1e-4))

    train_set, test_set = get_datasets(cfg)
    nworkers = 0 if smoke else int(train_cfg.get("num_workers", 2))
    train_loader, test_loader = make_loaders(train_set, test_set, batch_size=batch,
                                             num_workers=nworkers)

    net = build_trainable(cfg).to(device)
    opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    loss_fn = nn.CrossEntropyLoss()

    out = checkpoint_path(cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    max_batches = 2 if smoke else None

    best_acc = -1.0
    best_state = None
    best_epoch = -1
    no_improve = 0
    print(f"[train] device={device}  epochs={epochs}  patience={patience}  "
          f"train={len(train_set)}  test={len(test_set)}")
    for ep in range(epochs):
        net.train()
        t0 = time.time()
        last_loss = float("nan")
        for b, (x, y) in enumerate(train_loader):
            if max_batches is not None and b >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(net(x), y)
            loss.backward()
            opt.step()
            last_loss = loss.item()
        sched.step()
        net.eval()
        acc = evaluate_accuracy(net, test_loader, device, max_batches=(2 if smoke else None))
        lr_now = sched.get_last_lr()[0]
        flag = ""
        if acc > best_acc + min_delta:
            best_acc, best_epoch = acc, ep + 1
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in net.state_dict().items()})
            no_improve = 0
            torch.save({"model": best_state, "cfg": cfg, "acc": best_acc, "epoch": best_epoch}, out)
            flag = "  *best (saved)"
        else:
            no_improve += 1
        print(f"epoch {ep + 1}/{epochs}  loss={last_loss:.3f}  test_acc={acc:.4f}  "
              f"lr={lr_now:.4f}  ({time.time() - t0:.1f}s){flag}")
        if patience and no_improve >= patience and not smoke:
            print(f"[train] early stop: no improvement for {patience} epochs "
                  f"(best={best_acc:.4f} @ epoch {best_epoch})")
            break

    if best_state is None:  # smoke / degenerate: save whatever we have
        best_state = {k: v.detach().cpu() for k, v in net.state_dict().items()}
        best_acc, best_epoch = acc, epochs
        torch.save({"model": best_state, "cfg": cfg, "acc": best_acc, "epoch": best_epoch}, out)
    print(f"[train] saved best checkpoint -> {out}  (test_acc={best_acc:.4f} @ epoch {best_epoch})")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny net + synthetic data + 1 epoch (pipeline check)")
    args = ap.parse_args()

    cfg = load_experiment_config(args.config)
    if args.smoke:
        cfg["model"] = {"arch": "tiny"}
        cfg["data"] = {"synthetic": True, "download": False}
        cfg["checkpoint"] = "checkpoints/vgg_smoke.pt"
    tcfg = cfg.get("train", {})
    epochs = args.epochs if args.epochs is not None else (1 if args.smoke else int(tcfg.get("epochs", 100)))
    patience = args.patience if args.patience is not None else int(tcfg.get("patience", 15))
    train(cfg, epochs=epochs, smoke=args.smoke, patience=patience)


if __name__ == "__main__":
    main()
