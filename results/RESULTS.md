# Empirical results — operator-norm error bounds for factor-quantized low-rank nets

Real training runs (not smoke tests) on **4 trained models**: shallow BN VGG (CIFAR-10, 93.5%),
spectral-norm twin (CIFAR-10, 88.9%), deep VGG-16 (CIFAR-10, 93.3%), and deep VGG-16 (SVHN, 95.9%).
GPU: RTX 5060 Ti. CSVs/PDFs in `results/` and `figures/`; variants use suffixed filenames.

> **Bottom line (honest, one paragraph).** The **operator-norm bound (#1) is robustly validated** —
> across depth (8→15 layers), two datasets (CIFAR-10, SVHN), and OOD/adversarial inputs; it is
> *achievable* (ρ=1 for a single operator) and mechanistically confirmed (Lemma 1 holds per-layer),
> valid-but-loose with *provably compositional* looseness. The **practical surrogate claims built on
> `S_i` do not hold generally**: per-layer ranking (#2) flips sign by dataset (positive CIFAR-10,
> negative SVHN), and budget allocation (#3) needs a *bound-free* empirically-measured weight at depth.
> So the **theorem is solid; `S_i` as a task-agnostic importance signal is not.** The theory papers
> themselves frame the surrogate value as an open empirical question (the "Bound Tightness and
> Empirical Calibration" note allows "valid but loose", ρ≈0) — so none of this contradicts a proven
> statement. Sections below give each setting; the summary table is the shallow BN net unless noted.

## Summary vs the success criteria (spec §5)

| Criterion | Result | One-line verdict |
|---|---|---|
| **#1 bound validity** | ✅ **holds** | 0 violations under exact H across 40 configs at every depth (incl. 15-layer net); valid but very loose. |
| **#2 ranking correlation** | ✗ **dataset-dependent** | Conv-trunk **+0.56 on CIFAR-10** (robust, all 8 seeds) but **−0.39 on SVHN** (robust, 90% of settings) — the sign flips by dataset, so `S_i` ranking is **not** a general property. |
| **#3 allocation** | ◐ **`S_i` adds little** | Raw (magnitude-weighted) `S_i` collapses at depth (procedure artefact). Control: a **flat knapsack with no `S_i`** (w_i=1, min Σ(σ+η)) already recovers **87% (CIFAR) / 96% (SVHN)** at 16% budget — so allocation value is in the bound's **per-layer error terms**, not the `S_i` coefficient (log-`S_i` adds ≤few pts; nothing on SVHN). `S_i`'s ordering is task-dependent (rank-`S_i`: CIFAR yes, SVHN no). `experiments/allocator_procedure.py`. |

(Headline rows are the shallow BN net; the spectral-norm twin and the deep VGG-16 are in the two
sections below.)

## A — bound validity (`experiment_a.csv`, `figures/experiment_a_validity.pdf`)

- **Exact-H violation rate = 0.0000** over 40 (uniform + random per-layer) configs. The bound is
  never violated — success criterion #1 holds, including on six stacked 3×3 convs at real
  resolutions (the operator-norm-consistency case).
- The bound is **valid but loose**: median tightness `ρ_max ≈ 1.7e-6`, i.e. the predicted bound
  exceeds the measured logit error by a **median factor ≈ 5.8e5×** (range 1.5e5–2.0e6×).
- Because the bound is so loose, **no violations appear even under calibration/percentile H** here
  (the under-estimate of H is dwarfed by the looseness). The exact-vs-empirical-H violation split
  is reported (all 0) per spec.

## B — ranking correlation (`experiment_b.csv`, `figures/experiment_b_ranking.pdf`)

Compress one layer at a time (k = 0.5·rank, b = 4), correlate the logit-error increase with `S_i`.

- **Primary `corr(S_i, error)`: Spearman ρ = 0.00 (p = 1.0), Kendall τ = 0.00 (p = 1.0)** — no rank
  correlation. (The exact 0.0 is a genuine coincidence of this 8-point sample, Σd²=504, not a bug.)
- Diagnostic `corr(S_i·(σ_{k+1}+η), error)`: Spearman = −0.02 — rules out the ‖ΔW_i‖ confound.
- Diagnostic conv-trunk only (6 conv layers, excl. classifier): Spearman = **0.20 (p = 0.70)** —
  weakly positive but **insignificant**, so the looseness is *pervasive*, not just a classifier artefact.

**Mechanism.** `S_i = Γ_i L_i H_{i-1}` with `Γ_i = Π_{j>i} L_j‖W_j‖`. Spectral norms here run up to
~35, so `Γ` spans **2.1e5 → 1** monotonically with depth. `S_i` therefore over-weights early layers
by up to 6 orders of magnitude, while the *measured* per-layer sensitivity is flat (≈2.5–13.7) and
is in fact **largest for the final classifier** (smallest `S_i`, since it perturbs logits directly with
no downstream gain). This is the textbook looseness of spectral-norm-product Lipschitz bounds.

## C — budget allocation (`experiment_c.csv`, `figures/experiment_c_*.pdf`)

Lagrangian-relaxation allocation of `(k_i,b_i)` minimizing `Σ_i S_i(σ_{k+1,i}+η^fac_i)` under the
memory budget `C_i=b_i k_i(m_i+n_i+1)`, vs the best uniform `(k,b)` at the same budget. **The
objective uses the measured `η^fac`** (spec §1; the closed-form is only its upper bound and its
√k growth distorts the allocation).

| Budget | guided acc | uniform acc | guided err | uniform err |
|---:|---:|---:|---:|---:|
| 3.20M | 0.146 | 0.103 | 25.17 | 25.33 |
| 5.91M | **0.885** | 0.470 | 20.21 | 19.02 |
| 8.62M | **0.929** | 0.627 | 6.97 | 15.86 |
| 11.3M | **0.935** | 0.744 | 1.14 | 8.75 |
| 14.0M | **0.936** | 0.746 | 1.00 | 8.53 |
| 16.7M | **0.936** | 0.924 | 1.00 | 3.38 |
| 19.4M | 0.936 | 0.936 | 1.00 | 1.00 |

- **`S_i`-guided Pareto-dominates uniform**: it reaches ~93.5% accuracy at ≈5.9M budget, where
  uniform is at ~47%; uniform needs ≈19M (full) to match. Guided err ≤ uniform err in 88% of budgets.
- **Why does `S_i`-guided win *here* (shallow BN), despite weak ranking?** (This is local to the
  shallow BN net — it does **not** hold at depth; see the Depth study.) Allocation does not need accurate
  per-layer *ranking*. The huge `S_i` on the cheap early conv layers (small fan-in ⇒ low cost `C_i`)
  drives the optimizer to fully provision them and keep the tiny classifier at full rank, while
  compressing the expensive late layers — which empirically beats uniform compression. The win comes
  from coarse structure + cost asymmetry, not fine ranking — and it is fragile: at depth the same
  mechanism over-skews and collapses.
- `experiment_c.py` reports **three** curves — uniform, raw-`S_i`-guided (the surrogate), and an
  **empirical-sensitivity** reference that uses *no* bound quantity. On this shallow net raw `S_i`-guided
  beats uniform in 88% of budgets and the empirical reference in 100%; at depth (see the Depth study) raw
  `S_i` collapses and only the bound-free empirical allocation holds — i.e. the surrogate's allocation
  value does not survive depth.

## Spectral-normalized variant — the tighter-bound regime (`*_sn.csv`, `figures/*_sn.pdf`)

To test the open question directly, we trained a **Lipschitz-constrained** twin: the same VGG
with `spectral_norm` on every conv/linear and **no BatchNorm** (BN folding would re-introduce
scaling and undo the constraint). Test acc **88.89%** (vs 93.49%; the constraint costs ~4.6 pts).
Reproduce with `train.py --config configs/model_sn.yaml`, then `experiment_*.py --sn`.

This collapses the spectral-norm spread: conv-operator norms ≈ **1.5–2.4** (matrix ‖W‖₂ ≈ 1.0)
vs up to **35** in the BN net, so `Γ_0 ≈ 25` instead of **214,369**.

| Quantity | BN net (93.5%) | Spectral-norm net (88.9%) |
|---|---|---|
| **#1** exact-H violations | 0 / 40 ✅ | 0 / 40 ✅ |
| Bound looseness (median bound/err) | ~5.8e5× | **~1.9e3×**  (≈300× tighter) |
| Tightness ρ_max (median) | 1.7e-6 | **5.3e-4** |
| **#2** Spearman, all 8 layers | 0.00 | −0.05 |
| **#2** Spearman, conv trunk (6) | 0.20 (p=0.70) | **0.60 (p=0.21)** |
| **#3** guided vs uniform | guided **dominates** (93.5% @ 5.9M vs 47%) | **mixed** (uniform wins low/mid budget) |

**Reading.**
- **#1 is architecture-independent** — valid on both.
- **Tightening works as the theory predicts.** Constraining the spectral norms tightens the bound
  ~300× (Γ no longer compounds). The bound is still loose (~1900×) because conv operator norms
  exceed 1 and ReLU gating is unmodelled, but the regime shift is unambiguous.
- **Ranking on this net (CIFAR-10).** The conv-trunk correlation is 0.20 (BN) vs 0.60 (spectral-norm),
  both **not significant** (n=6) — suggestive but underpowered. **Caveat:** the deep + SVHN results
  below show the ranking is *dataset-dependent* (sign flips), so do **not** read this as "tightness
  fixes the ranking" — it is one CIFAR data point. The all-8 correlation stays ~0 because the final
  classifier (large empirical sensitivity, `Γ=1` ⇒ small `S_i`) is a structural outlier in both nets.
- **Allocation on this net.** `S_i`-guided dominates uniform on the *BN* net but only ties on the
  *spectral-norm* net (flat `S_i` ⇒ uniform near-optimal). This shallow win does **not** survive depth
  (Depth study: raw `S_i` collapses).

**Net (this section, CIFAR-10):** validity holds on both; the bound tightens ~300× under Lipschitz
control. The shallow ranking/allocation numbers are *not* general — see the Depth and Generality
sections for where the surrogate claims break.

## Depth study — VGG-16, 13 conv layers (`*_deep.csv`, `figures/*_deep.pdf`)

The shallow nets only give 6–8 layers, too few for the rank test (#2) to reach significance.
We therefore trained a **deep BN VGG-16** (13 conv + 2 FC, no skips; test acc **93.25%**) to give
the ranking correlation real statistical power (n=13 conv). Run with `train.py --config
configs/model_deep.yaml` then `experiment_*.py --deep`.

- **#1 still holds** at 15 weight layers (0/40 exact-H violations). The bound is *even looser* at
  depth (median bound/err ≈ **1.4e9×**) — `Γ` compounds over 13 layers — so tightness and depth pull
  in opposite directions.
- **#2 becomes significant *on CIFAR-10*.** Within the conv trunk: **Spearman ρ = 0.58, p = 0.039
  (n=13)** — a positive, significant rank-correlation between `S_i` and empirical per-layer sensitivity.
  **But this is CIFAR-specific: it reverses to −0.39 on SVHN** (see Generality). The
  all-layer correlation stays weak (ρ=0.11) because the FC classifier (`Γ≈1` ⇒ small `S_i`, but the
  single most sensitive layer empirically) is a structural outlier.
  - **Robustness** (`experiment_b.py --deep --robust`, 8 eval-subset seeds × 5 probe `(k,b)` settings
    = 40 measurements). The clean statement (8 ≈independent eval-subset seeds, each averaged over the
    5 probes): the conv-trunk Spearman is **0.50–0.61 and positive in all 8 seeds** (8/8 sign-test
    p ≈ 0.008) — strikingly consistent. Pooling all 40 settings: **ρ = 0.56 ± 0.30**, positive in 95%,
    median p = 0.026, 55% individually p<0.05. So the positive correlation is robust, not a single-run
    artefact; the wide per-*setting* spread reflects single-batch + probe noise (n=13), which the
    per-seed averaging removes. (We avoid a 40-sample sign-test: those settings are correlated, so it
    would overstate significance; the 8-seed test and the conservative single-run p=0.039 are the
    honest quotes.)

  | Net | conv layers | conv-trunk Spearman | significant? |
  |---|---:|---:|---|
  | shallow BN | 6 | 0.20 (p=0.70) | no |
  | shallow spectral-norm | 6 | 0.60 (p=0.21) | no (underpowered) |
  | **deep BN** | **13** | **0.58 (p=0.039)** | **yes** |

  Reading: on **CIFAR-10** the conv-trunk correlation is consistently moderate-positive (~0.55–0.60)
  and reaches significance once there are enough layers (so the shallow "≈0" was an all-layer + small-n
  artefact). **But this positive result is CIFAR-10-specific** — on SVHN the same conv-trunk correlation
  is negative (Generality section). So #2 is *supported on CIFAR-10 at depth* but **not general**; the
  classifier is a structural exception in every configuration on top of that.
- **#3 at depth: the paper's `S_i`-guided allocation does NOT dominate uniform — it collapses.**
  Raw `S_i` is over-skewed at depth (`Γ` spans ~9 orders), so the optimizer pours budget into layer 0
  and starves the rest → **10% (chance)** at low budget; it beats uniform in only **57%** of budgets
  (≈coin-flip). For reference we also allocate by a **directly measured** per-layer sensitivity weight
  (mean logit-error increase when each layer is compressed alone). This reference **uses no bound
  quantity at all** — no `Γ`, `H`, or `S_i` — so it *bypasses* the surrogate:

  | Budget | empirical (no bound) | raw `S_i` (bound) | uniform |
  |---:|---:|---:|---:|
  | 22.4M (≈16% of max) | **0.932** | 0.101 | 0.165 |
  | 41.4M | **0.934** | 0.591 | 0.777 |
  | 60.4M | 0.932 | 0.932 | 0.898 |
  | 136M (full) | 0.932 | 0.932 | 0.932 |

  The empirical reference reaches ~93% at ≈16% of the budget and beats uniform in 86% of budgets.
  **Interpretation (honest):** the allocation *framework* (multiple-choice knapsack over `(k,b)`) is
  sound, but the theoretical `S_i` is **not** the operative weight at depth — the relative per-layer
  ranking it implies is wrong (most sharply for the classifier: tiny `S_i`, largest true sensitivity),
  not merely mis-scaled (a pure rescale would leave the allocation unchanged, yet `S_i` gets 10% and the
  measured weight gets 93% at the same budget). So this is a **negative result for `S_i` as an allocation
  weight at depth**, plus a positive result for *measuring* sensitivity directly — which does not need the
  bound. (`experiment_c.py` reports all three curves; the empirical weight is calibrated on a split
  disjoint from the scored data.)

### A note on the deep spectral-norm variant (a tension, not a result)
We also attempted a **deep spectral-norm (no-BN)** twin to get "deep + tight" simultaneously. It does
**not train**: a 15-layer net with `‖W‖≤1` per layer is strongly contractive (signal vanishes
geometrically with depth → dead gradients, stuck at 10%). Adding a fixed gain `c≈√2` is not enough,
and the gain needed to train (`c≳1.9`) re-loosens `Γ` to ~BN levels. This is a **fundamental tension**:
a *tight* product-of-operator-norms bound requires a contractive net, which is hard to train deep
without BN/skips (both of which loosen or void the bound). Worth stating in the paper as a limitation
of the surrogate's tight regime. (Infra is in place — `configs/model_deep_sn.yaml`, `sn_scale` — if a
better-conditioned constrained-Lipschitz architecture is pursued.)

## Additional validations (`experiments/validate.py`, `results/validate_*_deep.csv`)

Three checks beyond A/B/C, on the deep VGG-16 (`validate.py --deep`):

**V1 — the proof mechanism (Lemma 1), not just the final bound.** For each compression config we
verify the per-layer recurrence `e_i ≤ L_i‖W_i‖e_{i-1} + L_i‖ΔW_i‖‖ĥ_{i-1}‖` sample-by-sample,
layer-by-layer. **0 violations** across all layers/samples (3 configs); the recurrence holds with
median `e_i/RHS ≈ 0.13`, i.e. ~8× slack *per layer* — which compounds over depth into the final
looseness. (Also unit-tested on a tiny net, `tests/test_bounds.py`.)

**V2 — the bound is achievable (not vacuous), and its looseness is compositional.** For a single
operator, the input aligned to the top singular vector of `ΔW` gives **ρ = 1.0000** — the
operator-norm bound is *exactly attained*. For a linear chain `y = W_L⋯W_1 x` (perturb `W_1`, feed
the aligned input), ρ decays geometrically with depth:

  | depth L | 1 | 2 | 4 | 8 |
  |---|---:|---:|---:|---:|
  | ρ (aligned) | **1.000** | 0.52 | 0.11 | 0.008 |

  This is a controlled demonstration of *why* the deep-net bound is loose: the product-of-operator-norms
  is tight for one operator but loses a factor each layer because consecutive layers' top singular
  subspaces do not align. It quantitatively explains the empirically observed looseness (median ~1.4e9×
  at depth) — the bound is valid and tight per-operator, loose only through composition.

**V3 — the bound holds for *all* inputs, not just test images.** Under exact H the bound is **never
violated (0%)** for Gaussian-noise, ×2-scaled, and **FGSM-adversarial** inputs — confirming the
"for all x" guarantee (#1) beyond the test distribution. The large looseness also means clean-calibrated
H is not violated even by ×2-activation inputs (a safety margin against H mis-calibration).

## Generality — second dataset (SVHN, `*_deep_svhn.csv`)

We re-ran the deep VGG-16 pipeline on **SVHN** (test acc **95.94%**; `configs/svhn.yaml`) to test whether
the findings are CIFAR-specific.

- **#1 replicates cleanly**: 0/40 exact-H violations — the proven bound holds on a second dataset, as
  expected (it is dataset-agnostic). Config-level Spearman(bound, error) = 0.39 (p=0.014), comparable to CIFAR.
- **#2 does NOT generalize — it reverses sign.** On SVHN the conv-trunk correlation is **negative in all
  8 eval-subset seeds** (per-seed mean −0.45 to −0.32; pooled −0.39 ± 0.28 over 40 settings) — exactly
  symmetric to CIFAR's 8/8 positive. Single-run all-layer −0.49 (p=0.06). **Not merely a magnitude artefact:** the
  matched conv-trunk diagnostic `S_i·(σ_{k+1}+η)` is **−0.24** (p=0.44; the all-layer diagnostic is −0.45) — insignificant on its own, so the dataset-dependence rests on the 8/8 per-seed reversal above; it is the
  *same code* that gives +0.56 on CIFAR — only the weights/data differ, so it cannot be a sign error.
  Reason: `S_i` always decreases with depth (via `Γ`), predicting early layers matter most; on CIFAR
  the early conv layers *are* the most sensitive empirically, but on **SVHN the late layers and
  classifier are most sensitive** (empirical sensitivity rises L8→L14), so `S_i`'s depth-bias anti-correlates.

  | conv-trunk Spearman | CIFAR-10 (deep) | SVHN (deep) |
  |---|---:|---:|
  | single run | +0.58 (p=0.039) | −0.24 (p=0.44) |
  | per-seed (8 eval subsets) | **8/8 positive** (0.50…0.61) | **8/8 negative** (−0.45…−0.32) |
  | `S_i·(σ+η)` diagnostic (conv-trunk) | (positive) | −0.24 (p=0.44); −0.45 all-layer |

  **Conclusion (honest): #2 is dataset-dependent, not a general property of the surrogate.** The `S_i`
  ranking matches empirical layer importance on CIFAR-10 but is anti-correlated on SVHN. So the earlier
  "conv-trunk ranking holds at depth" must be read as **CIFAR-10-specific**; across datasets the ranking
  claim is not supported. This is the central limitation the generality check surfaced — `S_i`'s
  structural early-layer bias (from the `Γ` product) is not a reliable task-agnostic importance signal.

- **Retraining robustness** (`experiments/seed_robustness.py`, `results/seed_robustness.csv`): to rule
  out a single-model artefact, 3 fresh retrains per dataset (40 epochs) give conv-trunk Spearman
  **+0.55 / +0.75 / +0.76 on CIFAR-10 (3/3 positive, mean +0.69)** vs **−0.54 / −0.58 / −0.63 on SVHN
  (3/3 negative, mean −0.58)**. The sign reversal is reproducible across retrainings, so #2's
  dataset-dependence is a **dataset-level effect, not a single-run artefact**.

## Caveats / scope

- **Coverage tested:** 3 architectures (shallow BN, spectral-norm, deep VGG-16) × 2 datasets
  (CIFAR-10, SVHN), plus OOD/adversarial inputs and a synthetic linear tightness study. #1 is validated
  across all; the surrogate (#2/#3) is where the limits show.
- **#2 is shown to be dataset-dependent on n=2 datasets** (CIFAR-10 +, SVHN −). Two datasets with
  opposite signs justify "not general," not "always negative" — a third dataset / a structured
  layer-importance model could refine *when* it holds. The classifier is a structural outlier in every
  setting.
- **#3:** raw `S_i` allocation is fragile (wins shallow, collapses deep); the robust allocator is the
  bound-free empirical-sensitivity weight, which is not part of the theory.
- **Looseness:** the bound is valid but loose (≈1e6× shallow, ≈1e9× deep); V2 shows this is purely
  compositional (tight per operator). `H` is calibrated on a held-out subset; the looseness means clean-H
  is not violated even by OOD inputs here, but on a tighter model it could be — both H modes are logged.
