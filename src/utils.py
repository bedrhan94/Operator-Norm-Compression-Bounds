"""Shared utilities: deterministic seeding, device selection, config + IO helpers."""
from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

# Repository root = parent of the ``src`` directory.
ROOT = Path(__file__).resolve().parent.parent


def set_seed(seed: int = 0) -> None:
    """Make a run reproducible across python / numpy / torch (CPU + CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic cuDNN; the small models here do not need the fast nondeterministic kernels.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(prefer: str | None = None) -> torch.device:
    """Return the compute device. ``prefer`` overrides auto-detection (e.g. ``"cpu"``)."""
    if prefer is not None:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_config(path: str | os.PathLike) -> dict[str, Any]:
    """Load a YAML config file into a plain dict."""
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def ensure_dir(path: str | os.PathLike) -> Path:
    """Create ``path`` (and parents) if missing; return it as a Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def results_dir() -> Path:
    return ensure_dir(ROOT / "results")


def figures_dir() -> Path:
    return ensure_dir(ROOT / "figures")


def checkpoints_dir() -> Path:
    return ensure_dir(ROOT / "checkpoints")


@dataclass
class RunPaths:
    """Convenience bundle of the standard output directories."""

    results: Path
    figures: Path
    checkpoints: Path

    @classmethod
    def make(cls) -> "RunPaths":
        return cls(results_dir(), figures_dir(), checkpoints_dir())
