"""Publication-ready matplotlib figures (saved as PNG to figures/).

One function per experiment figure described in the spec (sec.3). All use the Agg
backend so they render headless.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams.update({
    "figure.dpi": 120,
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "savefig.bbox": "tight",
})


def _save(fig, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def scatter_measured_vs_predicted(measured: Sequence[float], predicted: Sequence[float],
                                  path, title: str = "Experiment A: bound validity",
                                  violation_rate: Optional[float] = None) -> Path:
    """Measured logit error vs Theorem-2 predicted bound. Under exact H every point lies
    on/below the y=x line (predicted >= measured)."""
    measured = np.asarray(measured, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.scatter(predicted, measured, s=18, alpha=0.6, edgecolor="none", label="configs")
    lo = float(min(measured.min(), predicted.min(), 1e-12))
    hi = float(max(measured.max(), predicted.max(), 1e-9))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="y = x (tight)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Predicted bound  $\\sum_i \\Gamma_i L_i H_{i-1}(\\sigma_{k+1}+\\eta^{fac})$")
    ax.set_ylabel("Measured  $\\|z-\\hat z\\|_2$")
    sub = title if violation_rate is None else f"{title}  (violations: {violation_rate:.1%})"
    ax.set_title(sub)
    ax.legend(loc="upper left", framealpha=0.9)
    return _save(fig, path)


def scatter_sensitivity(S: Sequence[float], empirical: Sequence[float], path,
                        labels: Optional[Sequence[str]] = None,
                        spearman: Optional[float] = None, kendall: Optional[float] = None,
                        title: str = "Experiment B: sensitivity ranking") -> Path:
    """Predicted sensitivity ``S_i`` vs empirical per-layer logit-error increase."""
    S = np.asarray(S, dtype=float)
    empirical = np.asarray(empirical, dtype=float)
    fig, ax = plt.subplots(figsize=(5.4, 5))
    ax.scatter(S, empirical, s=40, color="tab:red", zorder=3)
    if labels is not None:
        for xi, yi, lab in zip(S, empirical, labels):
            ax.annotate(lab, (xi, yi), fontsize=8, xytext=(4, 2), textcoords="offset points")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Predicted sensitivity  $S_i = \\Gamma_i L_i H_{i-1}$")
    ax.set_ylabel("Empirical $\\Delta\\|z-\\hat z\\|_2$ (compress layer $i$)")
    bits = []
    if spearman is not None:
        bits.append(f"Spearman $\\rho$={spearman:.3f}")
    if kendall is not None:
        bits.append(f"Kendall $\\tau$={kendall:.3f}")
    ax.set_title(title + (("\n" + ", ".join(bits)) if bits else ""))
    return _save(fig, path)


def pareto_budget(budgets: Sequence[float], guided: Sequence[float], uniform: Sequence[float],
                  path, ylabel: str = "Test accuracy", lower_is_better: bool = False,
                  title: str = "Experiment C: allocation") -> Path:
    """Metric vs budget for S_i-guided vs uniform allocation."""
    budgets = np.asarray(budgets, dtype=float)
    fig, ax = plt.subplots(figsize=(5.8, 4.4))
    ax.plot(budgets, guided, "o-", color="tab:green", label="$S_i$-guided")
    ax.plot(budgets, uniform, "s--", color="tab:gray", label="uniform")
    ax.set_xlabel("Cost budget  $B$  (memory proxy $\\sum_i b_i k_i (m_i+n_i+1)$)")
    ax.set_ylabel(ylabel)
    if lower_is_better:
        ax.set_yscale("log")
    ax.set_title(title)
    ax.legend()
    return _save(fig, path)


def pareto_budget_multi(budgets: Sequence[float], series: Sequence[tuple], path,
                        ylabel: str = "Test accuracy", lower_is_better: bool = False,
                        title: str = "Experiment C: allocation") -> Path:
    """Metric vs budget for several allocation strategies.

    ``series`` is a list of ``(label, values, fmt, color)`` tuples, e.g.
    ``("calibrated-guided", accs, "o-", "tab:blue")``.
    """
    budgets = np.asarray(budgets, dtype=float)
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    for label, vals, fmt, color in series:
        ax.plot(budgets, vals, fmt, color=color, label=label, markersize=5)
    ax.set_xlabel("Cost budget  $B$  (memory proxy $\\sum_i b_i k_i (m_i+n_i+1)$)")
    ax.set_ylabel(ylabel)
    if lower_is_better:
        ax.set_yscale("log")
    ax.set_title(title)
    ax.legend()
    return _save(fig, path)
