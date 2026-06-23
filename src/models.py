"""CIFAR-10 model + the BN-folded ``AnalysisModel`` used by the bound machinery.

Design note (theory <-> code correspondence):

The abstract network of the theory is a *strict* feed-forward chain
``h_0 = x``, ``h_i = phi_i(W_i h_{i-1})`` with each ``phi_i`` being ``L_i``-Lipschitz.
To make Lemma 1 / Theorem 1 hold literally we:

* use a **VGG-style net with no skip connections** (a residual block is not a plain
  composition, so it would break the recurrence);
* **fold BatchNorm into the preceding conv** at inference, so every ``phi_i`` is a
  composition of ReLU / max-pool / flatten -- each 1-Lipschitz in the Euclidean norm.
  Hence ``L_i = 1`` for every layer (documented in :meth:`AnalysisModel.lipschitz`).

The folded *bias* is identical in the full-precision and compressed nets, so it
cancels in the error recurrence ``||phi(Wh+b) - phi(W h_hat + b)|| <= L ||Wh - W h_hat||``
and is therefore **never quantized** -- only weights are.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parametrizations

from .data import NUM_CLASSES


# --------------------------------------------------------------------------- #
# Trainable VGG-style model (Conv-BN-ReLU stacks + FC head).
# --------------------------------------------------------------------------- #
@dataclass
class VGGConfig:
    """Architecture knobs. ``conv_stages`` = list of (out_channels, pool?)."""

    conv_stages: Sequence[Tuple[int, bool]] = field(
        default_factory=lambda: [
            (64, False), (64, True),
            (128, False), (128, True),
            (256, False), (256, True),
        ]
    )
    fc_dims: Sequence[int] = field(default_factory=lambda: [256])
    in_channels: int = 3
    num_classes: int = NUM_CLASSES
    input_size: int = 32
    batchnorm: bool = True
    spectral_norm: bool = False  # constrain matrix ||W||_2 -> ~sn_scale (tightens the bound regime)
    sn_scale: float = 1.0        # fixed per-layer Lipschitz gain on the spectral-normed weight;
                                 # ~sqrt(2) compensates ReLU variance-halving so deep no-BN nets train

    @staticmethod
    def tiny() -> "VGGConfig":
        """Small net for CPU smoke tests / unit tests (trains in seconds, low accuracy)."""
        return VGGConfig(conv_stages=[(8, True), (16, True)], fc_dims=[32])

    @staticmethod
    def deep() -> "VGGConfig":
        """VGG-16-style (13 conv + 2 FC = 15 weight layers, no skips). Depth gives the
        layer-ranking correlation (#2) real statistical power (n=13 conv vs 6)."""
        stages = [
            (64, False), (64, True),
            (128, False), (128, True),
            (256, False), (256, False), (256, True),
            (512, False), (512, False), (512, True),
            (512, False), (512, False), (512, True),
        ]
        return VGGConfig(conv_stages=stages, fc_dims=[512])


class _Scale(nn.Module):
    """Parametrization that multiplies the (already spectral-normalized) weight by a constant."""

    def __init__(self, c: float):
        super().__init__()
        self.c = float(c)

    def forward(self, W: torch.Tensor) -> torch.Tensor:
        return self.c * W

    def right_inverse(self, W: torch.Tensor) -> torch.Tensor:
        return W / self.c


def _maybe_sn(module: nn.Module, apply: bool, scale: float = 1.0) -> nn.Module:
    """Wrap a conv/linear in spectral-norm (optionally with a fixed gain ``scale``).

    With the parametrization, ``module.weight`` returns the effective weight on access
    (matrix 2-norm ~``scale``), so BN folding and the analysis chain see it automatically.
    A k x k conv's operator norm can still exceed ``scale``, but ``Gamma_i`` is bounded by
    ~``scale^L`` instead of blowing up. ``scale ~ sqrt(2)`` lets deep no-BN nets train by
    offsetting ReLU's variance-halving (a hard ||W||<=1 net is too contractive to learn)."""
    if not apply:
        return module
    module = parametrizations.spectral_norm(module)
    if scale != 1.0:
        from torch.nn.utils import parametrize
        parametrize.register_parametrization(module, "weight", _Scale(scale))
    return module


def build_vgg(cfg: VGGConfig) -> nn.Sequential:
    """Build the trainable VGG-style net as a flat ``nn.Sequential`` (eases BN folding).

    If ``spectral_norm`` is set, BatchNorm is normally disabled (``batchnorm=False``) because
    folding BN re-introduces per-channel scaling that would undo the Lipschitz constraint.
    """
    sn = cfg.spectral_norm
    use_bn = cfg.batchnorm
    scale = cfg.sn_scale
    layers: List[nn.Module] = []
    c_in = cfg.in_channels
    n_pools = 0
    for c_out, pool in cfg.conv_stages:
        # bias only when there is no following BN to provide the shift.
        conv = nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=not use_bn)
        layers.append(_maybe_sn(conv, sn, scale))
        if use_bn:
            layers.append(nn.BatchNorm2d(c_out))
        layers.append(nn.ReLU(inplace=False))
        if pool:
            layers.append(nn.MaxPool2d(2, 2))
            n_pools += 1
        c_in = c_out
    layers.append(nn.Flatten())
    spatial = cfg.input_size // (2 ** n_pools)
    feat = c_in * spatial * spatial
    for d in cfg.fc_dims:
        layers.append(_maybe_sn(nn.Linear(feat, d), sn, scale))
        layers.append(nn.ReLU(inplace=False))
        feat = d
    layers.append(_maybe_sn(nn.Linear(feat, cfg.num_classes), sn, scale))  # final: phi_L = identity
    return nn.Sequential(*layers)


# --------------------------------------------------------------------------- #
# BN folding.
# --------------------------------------------------------------------------- #
def fold_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(W_folded, b_folded)`` equivalent to ``BN(conv(x))`` at inference.

    BN(y)_c = gamma_c (y_c - mean_c)/sqrt(var_c+eps) + beta_c, so with
    ``s_c = gamma_c / sqrt(var_c + eps)``:  W'_c = s_c W_c,  b'_c = (b_c - mean_c) s_c + beta_c.
    """
    W = conv.weight.detach().clone()
    if conv.bias is not None:
        b = conv.bias.detach().clone()
    else:
        b = torch.zeros(conv.out_channels, device=W.device, dtype=W.dtype)
    s = bn.weight.detach() / torch.sqrt(bn.running_var.detach() + bn.eps)
    W_folded = W * s.reshape(-1, 1, 1, 1)
    b_folded = (b - bn.running_mean.detach()) * s + bn.bias.detach()
    return W_folded, b_folded


# --------------------------------------------------------------------------- #
# Op descriptors for the BN-folded analysis chain.
# --------------------------------------------------------------------------- #
@dataclass
class ConvOp:
    weight: torch.Tensor  # (out, in, kh, kw)  -- full-precision W_i (folded)
    bias: torch.Tensor    # (out,)             -- never quantized
    stride: Tuple[int, int]
    padding: Tuple[int, int]
    kind: str = "conv"


@dataclass
class LinearOp:
    weight: torch.Tensor  # (out, in)
    bias: torch.Tensor    # (out,)
    kind: str = "linear"


@dataclass
class ReLUOp:
    kind: str = "relu"
    lipschitz: float = 1.0


@dataclass
class MaxPoolOp:
    kernel: int = 2
    stride: int = 2
    kind: str = "maxpool"
    lipschitz: float = 1.0  # max-pooling is 1-Lipschitz in L2


@dataclass
class FlattenOp:
    kind: str = "flatten"
    lipschitz: float = 1.0


WeightOp = (ConvOp, LinearOp)
NonlinOp = (ReLUOp, MaxPoolOp, FlattenOp)


@dataclass
class LayerSpec:
    """Per-weight-layer metadata consumed by spectral.py / compress.py / bounds.py."""

    index: int                       # 0-based position in the weight-layer chain
    kind: str                        # "conv" | "linear"
    weight: torch.Tensor             # full-precision folded W_i
    bias: torch.Tensor
    stride: Tuple[int, int]
    padding: Tuple[int, int]
    input_shape: Tuple[int, ...]     # (C, H, W) for conv input, (features,) for linear
    matrix_shape: Tuple[int, int]    # (m_i, n_i): conv -> (out, in*kh*kw); linear -> (out, in)
    lipschitz: float                 # L_i for phi_i (the nonlinearity *after* this layer)


class AnalysisModel:
    """A BN-folded, strictly feed-forward view of a trained net.

    Holds an ordered list of ops and exposes the chain that the bounds operate on.
    Forward passes can override individual weight tensors (to evaluate the compressed
    network f_hat) without mutating any module state.
    """

    def __init__(self, ops: List[object], input_size: int = 32, in_channels: int = 3):
        self.ops = ops
        self.input_size = input_size
        self.in_channels = in_channels
        self.weight_op_positions = [i for i, op in enumerate(ops) if isinstance(op, WeightOp)]
        self._specs: Optional[List[LayerSpec]] = None

    # -- construction --------------------------------------------------------- #
    @classmethod
    def from_sequential(cls, model: nn.Sequential, input_size: int = 32,
                        in_channels: int = 3) -> "AnalysisModel":
        model = model.eval()
        children = list(model.children())
        ops: List[object] = []
        i = 0
        while i < len(children):
            m = children[i]
            if isinstance(m, nn.Conv2d):
                if i + 1 < len(children) and isinstance(children[i + 1], nn.BatchNorm2d):
                    W, b = fold_conv_bn(m, children[i + 1])
                    i += 2
                else:
                    W = m.weight.detach().clone()
                    b = (m.bias.detach().clone() if m.bias is not None
                         else torch.zeros(m.out_channels, device=W.device, dtype=W.dtype))
                    i += 1
                ops.append(ConvOp(weight=W, bias=b,
                                  stride=tuple(m.stride), padding=tuple(m.padding)))
            elif isinstance(m, nn.Linear):
                Wl = m.weight.detach().clone()
                b = (m.bias.detach().clone() if m.bias is not None
                     else torch.zeros(m.out_features, device=Wl.device, dtype=Wl.dtype))
                ops.append(LinearOp(weight=Wl, bias=b))
                i += 1
            elif isinstance(m, nn.ReLU):
                ops.append(ReLUOp()); i += 1
            elif isinstance(m, nn.MaxPool2d):
                k = m.kernel_size if isinstance(m.kernel_size, int) else m.kernel_size[0]
                s = m.stride if isinstance(m.stride, int) else m.stride[0]
                ops.append(MaxPoolOp(kernel=k, stride=s)); i += 1
            elif isinstance(m, (nn.Flatten,)):
                ops.append(FlattenOp()); i += 1
            else:
                raise TypeError(f"Unsupported module in sequential: {type(m).__name__}")
        return cls(ops, input_size=input_size, in_channels=in_channels)

    # -- device --------------------------------------------------------------- #
    def to(self, device: torch.device) -> "AnalysisModel":
        for op in self.ops:
            if isinstance(op, WeightOp):
                op.weight = op.weight.to(device)
                op.bias = op.bias.to(device)
        return self

    @property
    def device(self) -> torch.device:
        return self.ops[self.weight_op_positions[0]].weight.device

    @property
    def num_layers(self) -> int:
        return len(self.weight_op_positions)

    # -- forward -------------------------------------------------------------- #
    def _apply_weight(self, op, x, weight):
        if isinstance(op, ConvOp):
            return F.conv2d(x, weight, op.bias, stride=op.stride, padding=op.padding)
        return F.linear(x, weight, op.bias)

    def run(self, x: torch.Tensor,
            weights: Optional[Dict[int, torch.Tensor]] = None,
            collect_norms: bool = False):
        """Run the chain. ``weights[j]`` overrides the j-th weight layer (compressed net).

        Returns ``(logits, input_norms)`` where ``input_norms[j]`` is the per-sample
        L2 norm of the (flattened) activation feeding weight layer j -- i.e. ||h_{j-1}||.
        """
        norms: Dict[int, torch.Tensor] = {}
        wl = 0
        for op in self.ops:
            if isinstance(op, WeightOp):
                if collect_norms:
                    norms[wl] = x.reshape(x.shape[0], -1).norm(dim=1).detach()
                W = op.weight if (weights is None or wl not in weights) else weights[wl]
                x = self._apply_weight(op, x, W)
                wl += 1
            elif isinstance(op, ReLUOp):
                x = F.relu(x)
            elif isinstance(op, MaxPoolOp):
                x = F.max_pool2d(x, op.kernel, op.stride)
            elif isinstance(op, FlattenOp):
                x = x.reshape(x.shape[0], -1)
        return x, norms

    def forward_activations(self, x: torch.Tensor,
                            weights: Optional[Dict[int, torch.Tensor]] = None) -> List[torch.Tensor]:
        """Return ``[h_0, h_1, ..., h_L]`` (h_0 = input, h_L = logits). ``h_j`` is the
        activation feeding weight layer j (post-phi_{j-1}); used to validate Lemma 1's
        per-layer error recurrence."""
        acts: List[torch.Tensor] = []
        wl = 0
        for op in self.ops:
            if isinstance(op, WeightOp):
                acts.append(x)  # h_{wl}: input to weight layer wl
                W = op.weight if (weights is None or wl not in weights) else weights[wl]
                x = self._apply_weight(op, x, W)
                wl += 1
            elif isinstance(op, ReLUOp):
                x = F.relu(x)
            elif isinstance(op, MaxPoolOp):
                x = F.max_pool2d(x, op.kernel, op.stride)
            elif isinstance(op, FlattenOp):
                x = x.reshape(x.shape[0], -1)
        acts.append(x)  # h_L: final logits
        return acts

    def forward_full(self, x: torch.Tensor) -> torch.Tensor:
        return self.run(x)[0]

    def forward_compressed(self, x: torch.Tensor,
                           weights: Dict[int, torch.Tensor]) -> torch.Tensor:
        return self.run(x, weights=weights)[0]

    # -- metadata ------------------------------------------------------------- #
    def lipschitz(self) -> List[float]:
        """L_i for each weight layer = product of Lipschitz constants of the
        nonlinearity ops *after* it (up to the next weight layer). All are 1 for the
        ReLU/pool/flatten op set, so every ``L_i = 1``."""
        L: List[float] = []
        positions = self.weight_op_positions + [len(self.ops)]
        for a, b in zip(positions[:-1], positions[1:]):
            prod = 1.0
            for op in self.ops[a + 1:b]:
                prod *= getattr(op, "lipschitz", 1.0)
            L.append(prod)
        return L

    def layer_specs(self) -> List[LayerSpec]:
        """Compute per-layer specs, probing input shapes with one dummy forward."""
        if self._specs is not None:
            return self._specs
        dev = self.device
        x = torch.zeros(1, self.in_channels, self.input_size, self.input_size, device=dev)
        lips = self.lipschitz()
        specs: List[LayerSpec] = []
        wl = 0
        for op in self.ops:
            if isinstance(op, WeightOp):
                in_shape = tuple(x.shape[1:])
                if isinstance(op, ConvOp):
                    out_ch, in_ch, kh, kw = op.weight.shape
                    matrix_shape = (out_ch, in_ch * kh * kw)
                    specs.append(LayerSpec(
                        index=wl, kind="conv", weight=op.weight, bias=op.bias,
                        stride=op.stride, padding=op.padding,
                        input_shape=in_shape, matrix_shape=matrix_shape, lipschitz=lips[wl]))
                else:
                    out_f, in_f = op.weight.shape
                    specs.append(LayerSpec(
                        index=wl, kind="linear", weight=op.weight, bias=op.bias,
                        stride=(1, 1), padding=(0, 0),
                        input_shape=(in_f,), matrix_shape=(out_f, in_f), lipschitz=lips[wl]))
                x = self._apply_weight(op, x, op.weight)
                wl += 1
            elif isinstance(op, ReLUOp):
                x = F.relu(x)
            elif isinstance(op, MaxPoolOp):
                x = F.max_pool2d(x, op.kernel, op.stride)
            elif isinstance(op, FlattenOp):
                x = x.reshape(x.shape[0], -1)
        self._specs = specs
        return specs


@torch.no_grad()
def evaluate_accuracy(forward_fn, loader, device: torch.device, max_batches: Optional[int] = None) -> float:
    """Top-1 accuracy of a ``forward_fn(x) -> logits`` over a loader."""
    correct = total = 0
    for b, (x, y) in enumerate(loader):
        if max_batches is not None and b >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        pred = forward_fn(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)
