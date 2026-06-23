"""Shared experiment plumbing: config loading, model/data construction, CSV IO."""
from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

# Make ``src`` importable when scripts are run directly (python experiments/foo.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import data as datamod  # noqa: E402
from src.models import AnalysisModel, VGGConfig, build_vgg, evaluate_accuracy  # noqa: E402
from src.spectral import PowerIterConfig  # noqa: E402
from src.utils import ROOT, get_device, load_config, set_seed  # noqa: E402

DEFAULTS: Dict[str, Any] = {
    "seed": 0,
    "device": None,                     # auto
    "model": {"arch": "full"},
    "data": {"root": "./data", "synthetic": False, "download": True},
    "power_iter": {"iters": 100, "tol": 1.0e-6},
    "calibration_size": 512,
    "eval_size": 256,
    "checkpoint": "checkpoints/vgg_cifar10.pt",
    "percentile": 99.9,
}


def merge_defaults(cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in DEFAULTS.items()}
    if cfg:
        for k, v in cfg.items():
            if isinstance(v, dict) and isinstance(out.get(k), dict):
                out[k].update(v)
            else:
                out[k] = v
    return out


def load_experiment_config(path: Optional[str]) -> Dict[str, Any]:
    raw = load_config(path) if path else {}
    return merge_defaults(raw)


def vgg_config_from(cfg: Dict[str, Any]) -> VGGConfig:
    m = cfg.get("model", {})
    arch = m.get("arch", "full")
    if arch == "tiny":
        return VGGConfig.tiny()
    base = VGGConfig.deep() if arch == "deep" else VGGConfig()
    if "conv_stages" in m:
        base.conv_stages = [tuple(s) for s in m["conv_stages"]]
    if "fc_dims" in m:
        base.fc_dims = list(m["fc_dims"])
    if "batchnorm" in m:
        base.batchnorm = bool(m["batchnorm"])
    if "spectral_norm" in m:
        base.spectral_norm = bool(m["spectral_norm"])
    if "sn_scale" in m:
        base.sn_scale = float(m["sn_scale"])
    return base


def apply_variant(cfg: Dict[str, Any], deep: bool = False, sn: bool = False) -> Dict[str, Any]:
    """Overlay a model variant: ``deep`` (VGG-16) x ``sn`` (spectral-norm, no BN).

    Sets the matching checkpoint and an ``output_suffix`` so each variant's CSV/figures are
    saved alongside the others (e.g. ``experiment_b_deep_sn.csv``) instead of clobbering them.
    """
    model: Dict[str, Any] = {"arch": "deep" if deep else "full"}
    if sn:
        model["spectral_norm"] = True
        model["batchnorm"] = False
        # Deep no-BN SN nets need a fixed gain to train; must match the training config and
        # the checkpoint's parametrization structure (shallow SN was trained at scale 1.0).
        if deep:
            model["sn_scale"] = 1.4
    cfg["model"] = model
    ckpt = {(False, False): "checkpoints/vgg_cifar10.pt",
            (False, True): "checkpoints/vgg_sn.pt",
            (True, False): "checkpoints/vgg_deep.pt",
            (True, True): "checkpoints/vgg_deep_sn.pt"}[(deep, sn)]
    cfg["checkpoint"] = ckpt
    cfg["output_suffix"] = ("_deep" if deep else "") + ("_sn" if sn else "")
    return cfg


def apply_sn_overrides(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Back-compat: spectral-norm (shallow) variant."""
    return apply_variant(cfg, deep=False, sn=True)


def variant_suffix(cfg: Dict[str, Any]) -> str:
    """Output filename suffix for the active model variant (e.g. ``_deep_sn``)."""
    return cfg.get("output_suffix", "")


def build_trainable(cfg: Dict[str, Any]) -> torch.nn.Sequential:
    return build_vgg(vgg_config_from(cfg))


def checkpoint_path(cfg: Dict[str, Any]) -> Path:
    p = Path(cfg["checkpoint"])
    return p if p.is_absolute() else (ROOT / p)


def get_analysis_model(cfg: Dict[str, Any], device: torch.device,
                       require_checkpoint: bool = False) -> Tuple[AnalysisModel, bool]:
    """Build the VGG, load a checkpoint if present, fold BN -> AnalysisModel.

    Returns ``(model, loaded)``. ``loaded`` is False when no checkpoint was found (the
    net is then randomly initialised -- fine for A/B which test error propagation, but C's
    accuracy numbers are only meaningful with a trained checkpoint)."""
    net = build_trainable(cfg)
    ckpt = checkpoint_path(cfg)
    loaded = False
    if ckpt.exists():
        state = torch.load(ckpt, map_location="cpu")
        net.load_state_dict(state["model"] if "model" in state else state)
        loaded = True
    elif require_checkpoint:
        raise FileNotFoundError(f"checkpoint not found: {ckpt}")
    net = net.to(device).eval()
    cfgv = vgg_config_from(cfg)
    model = AnalysisModel.from_sequential(net, input_size=cfgv.input_size,
                                          in_channels=cfgv.in_channels).to(device)
    return model, loaded


def get_datasets(cfg: Dict[str, Any]):
    d = cfg.get("data", {})
    return datamod.get_datasets(root=d.get("root", "./data"),
                                synthetic=d.get("synthetic", False),
                                download=d.get("download", True),
                                dataset=d.get("dataset", "cifar10"))


def get_eval_calib(cfg: Dict[str, Any], test_set) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(eval_x, eval_y, calib_x)`` -- disjoint subsets of the test set."""
    eval_x, eval_y = datamod.tensor_batch(test_set, cfg["eval_size"], seed=11)
    calib_x, _ = datamod.tensor_batch(test_set, cfg["calibration_size"], seed=23)
    return eval_x, eval_y, calib_x


def pic_from(cfg: Dict[str, Any]) -> PowerIterConfig:
    p = cfg.get("power_iter", {})
    return PowerIterConfig(iters=int(p.get("iters", 100)), tol=float(p.get("tol", 1e-6)),
                           seed=int(p.get("seed", 0)))


def device_from(cfg: Dict[str, Any]) -> torch.device:
    return get_device(cfg.get("device"))


def write_csv(path, rows: Sequence[Dict[str, Any]], fieldnames: Optional[List[str]] = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    fieldnames = fieldnames or list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return path
