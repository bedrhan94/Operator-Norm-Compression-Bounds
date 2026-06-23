"""CIFAR-10 data loaders plus a synthetic fallback for CPU smoke tests / CI.

The synthetic dataset lets the whole pipeline (train loop, compression, bounds)
run without downloading the 170 MB CIFAR-10 archive. It is *not* a substitute for
real evaluation -- experiments load real CIFAR-10 by default.
"""
from __future__ import annotations

from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset

# Per-channel normalisation constants (standard values).
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
SVHN_MEAN = (0.4377, 0.4438, 0.4728)
SVHN_STD = (0.1980, 0.2010, 0.1970)
NUM_CLASSES = 10


class SyntheticCIFAR(Dataset):
    """Deterministic random 3x32x32 images with random labels.

    Used only for smoke tests; a model cannot reach meaningful accuracy on it,
    but every tensor shape matches real CIFAR-10 so the code paths are exercised.
    """

    def __init__(self, n: int = 256, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.x = torch.randn(n, 3, 32, 32, generator=g)
        self.y = torch.randint(0, NUM_CLASSES, (n,), generator=g)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.x[idx], int(self.y[idx])


def _transforms(train: bool, mean, std, augment: bool = True):
    import torchvision.transforms as T  # local import: torchvision optional for smoke tests

    if train and augment:
        return T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
                          T.ToTensor(), T.Normalize(mean, std)])
    return T.Compose([T.ToTensor(), T.Normalize(mean, std)])


def get_datasets(root: str = "./data", synthetic: bool = False, download: bool = True,
                 dataset: str = "cifar10"):
    """Return ``(train_set, test_set)`` for ``cifar10`` or ``svhn`` (both 32x32x3, 10 classes).
    Falls back to synthetic if torchvision/data is unavailable or ``synthetic=True``."""
    if synthetic:
        return SyntheticCIFAR(n=512, seed=1), SyntheticCIFAR(n=256, seed=2)
    try:
        import torchvision

        ds = dataset.lower()
        if ds == "cifar10":
            train = torchvision.datasets.CIFAR10(root=root, train=True, download=download,
                                                 transform=_transforms(True, CIFAR10_MEAN, CIFAR10_STD))
            test = torchvision.datasets.CIFAR10(root=root, train=False, download=download,
                                                transform=_transforms(False, CIFAR10_MEAN, CIFAR10_STD))
        elif ds == "svhn":
            # SVHN: no horizontal flip (digits); light crop aug only.
            train = torchvision.datasets.SVHN(root=root, split="train", download=download,
                                              transform=_transforms(True, SVHN_MEAN, SVHN_STD, augment=False))
            test = torchvision.datasets.SVHN(root=root, split="test", download=download,
                                             transform=_transforms(False, SVHN_MEAN, SVHN_STD))
        else:
            raise ValueError(f"unknown dataset {dataset!r} (use 'cifar10' or 'svhn')")
        return train, test
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"[data] {dataset} unavailable ({exc}); falling back to synthetic data.")
        return SyntheticCIFAR(n=512, seed=1), SyntheticCIFAR(n=256, seed=2)


def make_loaders(
    train_set: Dataset,
    test_set: Dataset,
    batch_size: int = 128,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, test_loader


def calibration_subset(dataset: Dataset, size: int, seed: int = 0) -> Subset:
    """Deterministic subset C used to estimate H_{i-1} = max_{x in C} ||h_{i-1}(x)||."""
    g = torch.Generator().manual_seed(seed)
    n = len(dataset)
    size = min(size, n)
    idx = torch.randperm(n, generator=g)[:size].tolist()
    return Subset(dataset, idx)


def tensor_batch(dataset: Dataset, size: int, seed: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collate a fixed-size ``(x, y)`` tensor batch from a dataset (for bound evaluation)."""
    sub = calibration_subset(dataset, size, seed=seed)
    xs, ys = [], []
    for x, y in sub:
        xs.append(x)
        ys.append(int(y))
    return torch.stack(xs), torch.tensor(ys, dtype=torch.long)
