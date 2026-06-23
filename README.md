# Operator-Norm Error Propagation for Factor-Quantized Low-Rank Neural Networks

**Valid but Compositionally-Loose Certificates and Task-Dependent Sensitivity Surrogates**

![Python](https://img.shields.io/badge/python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-CPU%2FCUDA-ee4c2c)
![License](https://img.shields.io/badge/license-MIT-green)
![tests](https://img.shields.io/badge/tests-12%20passing-brightgreen)

Empirical study (PyTorch / CIFAR-10 & SVHN) of a deterministic **operator-norm upper bound** on the
output error of neural networks compressed by per-layer **truncated-SVD + factor-wise uniform
quantization**, and of the practical claims built on it. The math is fixed and proven; this code
numerically tests it. Every computed quantity maps to a stated result and the implementing function
cites it (`# Theorem 1`, `# Lemma 2`, `# Eckart-Young`).

📄 **Paper:** [`paper/Bound_Validation.docx`](paper/Bound_Validation.docx) · 📊 **Results:** [`results/RESULTS.md`](results/RESULTS.md)

## TL;DR — three honest findings

1. **The bound is valid but vacuously loose.** Never violated (0/40 configs, every model, incl.
   adversarial/OOD) under the exact activation radius — but loose by a median **~6×10⁵** (shallow) to
   **~1.4×10⁹** (deep). Under the *exact* radius, non-violation follows from the theorem by construction
   (a consistency check); the substantive result is **tightness**, not validity. The looseness is
   provably **compositional**: a single operator attains the bound (ρ=1), but ρ decays geometrically with
   depth (1 → 0.52 → 0.11 → 0.008 for L = 1,2,4,8).
2. **The induced layer ranking is task-dependent.** The sensitivity surrogate `S_i = Γ_i L_i H_{i-1}`
   correlates with measured per-layer sensitivity **+0.56 on CIFAR-10 but −0.39 on SVHN**
   (conv-trunk; sign reproduced across 3 retrainings/dataset). So `S_i` is **not** a task-agnostic
   importance signal.
3. **`S_i` adds little to budget allocation.** Raw magnitude-weighted `S_i` allocation *collapses* at
   depth — but this is a **procedure artefact** of its 9-order magnitude spread. A control shows a
   **flat-weighted knapsack with no `S_i`** (minimize Σ(σ_{k+1}+η)) already recovers strong allocations
   (**87% CIFAR / 96% SVHN** at 16% budget). The allocation value lives in the bound's **per-layer error
   terms**, not the `S_i` coefficient.

**Bottom line:** the theorem is a sound worst-case certificate; its decomposed coefficient `S_i` is not
a useful, task-agnostic layer-importance signal.

## Install

```bash
python -m pip install -r requirements.txt
```

**Windows note.** PyTorch's CPU wheel needs the Microsoft VC++ runtime (`vcruntime140_1.dll`). If
`import torch` fails with `WinError 1114`, install the
[VC++ 2015–2022 Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe), or for a no-admin fix
`python -m pip install msvc-runtime`. For GPU on RTX 50-series (Blackwell, sm_120) use the CUDA 12.8
wheel: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128`.

## Quick check (CPU, seconds — no download, no training)

```bash
python -m pytest tests/ -q                  # 12 tests, incl. the bound-validity & Lemma-1 guarantees
python experiments/train.py        --smoke  # tiny net + synthetic data
python experiments/experiment_a.py --smoke  # bound validity
python experiments/experiment_b.py --smoke  # ranking
python experiments/experiment_c.py --smoke  # allocation
```

`--smoke` uses a tiny net + synthetic data to exercise every code path; the numbers are not meaningful.

## Full run (GPU recommended)

```bash
# Train baselines (downloads CIFAR-10 / SVHN; ~20-40 min each on a modern GPU)
python experiments/train.py --config configs/model.yaml          # shallow BN VGG (CIFAR-10, ~93.5%)
python experiments/train.py --config configs/model_sn.yaml       # spectral-norm twin (~88.9%)
python experiments/train.py --config configs/model_deep.yaml     # deep VGG-16 (CIFAR-10, ~93.3%)
python experiments/train.py --config configs/svhn.yaml           # deep VGG-16 (SVHN, ~95.9%)

# Experiments (each writes results/*.csv and figures/*.pdf; variants use --sn / --deep / --config)
python experiments/experiment_a.py --config configs/experiment_a.yaml          # A: bound validity
python experiments/experiment_b.py --config configs/experiment_b.yaml --deep --robust  # B: ranking
python experiments/experiment_c.py --config configs/experiment_c.yaml --deep          # C: allocation

# Extra validations and controls
python experiments/validate.py            --deep   # Lemma 1, tightness regime, OOD/FGSM
python experiments/seed_robustness.py              # ranking sign across 3 retrainings/dataset
python experiments/allocator_procedure.py --deep   # flat / rank / log-S_i allocator control

# Regenerate the paper
python paper/make_figures.py && python paper/make_paper.py   # needs python-docx
```

A/B test error propagation and run without a checkpoint; **C's accuracy requires a trained checkpoint**.

## Experiments

| Script | Studies |
|---|---|
| `experiment_a.py` | **A — bound validity**: sweep `(k_i, b_i)`, compare Theorem-2 bound vs measured `‖z−ẑ‖₂`; report ρ and violation rate under exact vs calibration H. |
| `experiment_b.py` | **B — ranking**: compress one layer at a time, correlate logit-error increase with `S_i` (Spearman/Kendall; `--robust` sweeps seeds×probes). |
| `experiment_c.py` | **C — allocation**: Lagrangian knapsack over `(k,b)` under memory budget; uniform / raw-`S_i` / empirical-oracle curves. |
| `validate.py` | Lemma-1 recurrence, tightness regime (ρ→1, compositional decay), OOD/FGSM robustness. |
| `seed_robustness.py` | Retrains 3 seeds/dataset to test whether the ranking sign is a dataset-level effect. |
| `allocator_procedure.py` | Control: `flat (w_i=1)` / `rank-S_i` / `log-S_i` allocators — isolates whether `S_i` adds value over the bound's error terms. |

## Key implementation choices (theory ↔ code)

* **VGG-style, no skip connections** — keeps the network a strict feed-forward chain so Lemma 1 holds.
* **BatchNorm folded into the preceding conv**, so every `φ_i` (ReLU / pool / flatten) is 1-Lipschitz
  ⇒ `L_i = 1`; the folded bias is shared between both nets and never quantized.
* **All rigorous bound terms are operator norms.** A k×k conv's operator norm ≠ its reshaped-matrix
  2-norm, so the truncation `‖W_i−W_{i,k}‖_op` and quant `‖W_{i,k}−Ŵ_i‖_op` terms are computed by
  power-iterating the *difference operators*, the adjoint obtained exactly via an autograd
  vector–Jacobian product. This makes Theorem 2 rigorous for convolutions, not just linear layers.
* **Exact radius** = empirical max of `‖ĥ_{i-1}(x)‖₂` over the evaluated set (not an analytic supremum);
  a calibration radius on a disjoint set is also reported, and violations under it are logged, never hidden.

## Repository layout

```
src/          models, spectral, compress, bounds, data, utils, viz   (the library)
experiments/  train.py, experiment_a/b/c.py, validate.py, seed_robustness.py, allocator_procedure.py
configs/      model*.yaml, experiment_*.yaml, svhn.yaml
tests/        test_spectral.py, test_compress.py, test_bounds.py   (12 passing)
results/      *.csv + RESULTS.md          figures/  *.pdf          paper/  make_*.py, .docx
```
Model checkpoints (`checkpoints/`) and dataset downloads (`data/`) are git-ignored; regenerate them with
`experiments/train.py`.

## Citing

See [`CITATION.cff`](CITATION.cff). Released under the [MIT License](LICENSE).
