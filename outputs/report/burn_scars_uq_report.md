# Reliability-Aware Uncertainty Quantification for Burn Scar Segmentation

An exploratory reproduction of the SHRUG-FM framework (Gonzalez-Calabuig et al.,
*SHRUG-FM: Reliability-Aware Foundation Models for Earth Observation*,
arXiv:2511.10370) on a burn-scar segmentation task.

## 1. Objective

Geospatial foundation models (GFMs) often fail silently in conditions
underrepresented during pretraining. SHRUG-FM's approach is to combine
several independent uncertainty signals — how unfamiliar the embedding
looks to the model, and how uncertain the model's own ensemble is — into a
single, interpretable risk score that predicts when a model's output is
likely to fail, so that low-confidence predictions can be flagged or
deferred rather than trusted blindly.

This experiment reproduces that approach end-to-end on one task (burn
scar segmentation) as a first validation step, using a frozen Earth
observation foundation model, before extending to additional tasks and
the full signal set.

## 2. Data

### 2.1 Dataset

**HLS Burn Scars** (`ibm-nasa-geospatial/hls_burn_scars`): 804 chips of
Harmonized Landsat-Sentinel (HLS) imagery over the contiguous United
States (2018–2021), 512x512 pixels, 6 spectral bands (Blue, Green, Red,
NIR-narrow, SWIR1, SWIR2), with a binary ground-truth mask (burned /
unburned / no-data) per chip.

### 2.2 Train / Validation / Test Split

The publicly released dataset ships only a 2-way split (train/validation);
it does not preserve the original paper's train/validation/test division.
We therefore drew our own fixed-seed random split of all 804 chips:

| Split | Chips | Role |
|---|---|---|
| Train | 485 | Ensemble training + fusion-model fitting |
| Validation | 54 | Fusion-model fitting |
| Test | 263 | Held out — all reported results below |
| (unused) | 2 | Excluded to match the target split sizes exactly |

## 3. Model and Ensemble

- **Backbone**: SSL4EO-S12 ViT-S/16, self-supervised (MoCo) pretrained on
  Sentinel-1/2 imagery, used **frozen** (no fine-tuning).
- **Decoder**: a lightweight convolutional upsampling head trained from
  scratch on top of the frozen backbone's patch embeddings, producing a
  per-pixel burned/unburned probability map.
- **Ensemble**: 5 decoder members, each trained independently on a
  bootstrap resample of the 485 training chips (a *bootstrap ensemble*,
  following SHRUG-FM's choice for the burn-scar task specifically), 20
  epochs each. Training loss decreased consistently across all 5 members
  (e.g. member 3: 0.32 → 0.077), indicating stable convergence.
- **Per-chip quality**: mean segmentation F1 across the ensemble was 0.625
  (train), 0.611 (validation), 0.650 (test).

## 4. Uncertainty Signals

Two of the three signal families from the original framework were
computed:

### 4.1 Representation-level familiarity (embedding OOD)

For each chip, the frozen backbone's embedding is compared against a
reference set of embeddings via k-means clustering (24 clusters):

- **Normalized centroid distance** — how far the chip's embedding sits
  from its nearest cluster centre, relative to the typical spread of that
  cluster.
- **NCDD (nearest-centroid distance deficit)** — how much closer the
  embedding is to its own assigned cluster than to all the others; a low
  or negative value flags an embedding that doesn't clearly belong to any
  familiar cluster.

Reference set used: the 485 training chips' own embeddings (see §9 for
why this matters).

### 4.2 Predictive uncertainty (ensemble disagreement)

For each chip, averaged over the region the ensemble predicts as burned:

- **Predictive entropy** — how uncertain the ensemble's *averaged*
  prediction is.
- **Mutual information** — how much the 5 ensemble members *disagree*
  with each other (the "epistemic" component of uncertainty).

### 4.3 Not yet included: input-level covariate OOD

The original framework's third signal family — percentile rank of
geophysical covariates (elevation, land cover, hydrological attributes)
against the pretraining distribution — requires auxiliary geospatial
datasets (Copernicus DEM, HydroATLAS) that have not yet been sourced.
Results below reflect the two signal families above only.

## 5. Signal Fusion

The four signals above are combined via a shallow, interpretable decision
tree (max depth 3), after a correlation filter and recursive feature
elimination narrow the signal set down. The tree is trained to predict
**failure** — defined as a chip's segmentation F1 falling below 0.6 — using
only the training and validation chips; the test chips are held out
entirely for evaluation.

## 6. Evaluation

Standard selective-prediction metrics, computed on the 263 held-out test
chips:

- **AURC** (Area Under the Risk-Coverage curve) — overall risk across all
  coverage levels; lower is better.
- **Risk@0.5 / Risk@0.75** — expected failure rate if only the 50% / 75%
  most-confident chips are accepted; lower is better.
- **AUDP** (Area Under the Discard-Performance curve) — segmentation
  quality (F1) as increasingly uncertain chips are discarded; higher is
  better.

Each metric is reported for the **fused** risk score, for each
**individual signal used alone**, and for a **random** (no-information)
baseline, to test whether combining signals adds value over any single
one.

## 7. Results

| Risk signal | AURC ↓ | Risk@0.5 ↓ | Risk@0.75 ↓ | AUDP ↑ |
|---|---|---|---|---|
| **Fused (all 4 signals)** | **0.096** | **0.057** | **0.110** | **0.765** |
| Predictive entropy (alone) | 0.214 | 0.099 | 0.137 | 0.673 |
| Mutual information (alone) | 0.286 | 0.224 | 0.233 | 0.628 |
| Normalized centroid distance (alone) | 0.264 | 0.270 | 0.289 | 0.664 |
| NCDD (alone) | 0.387 | 0.384 | 0.314 | 0.584 |
| Random discard (no information) | 0.302 | 0.281 | 0.290 | 0.636 |

↓ lower is better, ↑ higher is better. Bold = best-performing row per column.

## 8. Interpretation

- **Fusing signals clearly outperforms any single signal.** The fused
  score roughly halves AURC relative to the best individual signal
  (predictive entropy) and achieves the best score on every metric. This
  reproduces the central claim of the framework: a simple, interpretable
  combination of independent uncertainty signals identifies likely
  failures better than any one signal alone.
- **Predictive uncertainty (entropy) is the strongest individual signal**,
  clearly better than random and better than either embedding-OOD signal.
  This suggests that, for this task, *how confident the ensemble is*
  carries more failure-relevant information than *how unfamiliar the
  input embedding looks*.
- **The two embedding-OOD signals (NCDD, normalized centroid distance),
  used alone, perform at or below the random baseline.** This is the one
  result that should be read with caution rather than taken as a finding
  about the method itself — see §9.

## 9. Deviations from the Original Study

This reproduction differs from the original SHRUG-FM study in several
ways, grouped below by what they affect. None of these were accidental —
each is a deliberate scoping choice made to get a working first
reproduction with accessible data and compute — but together they mean
**absolute numbers here should not be compared directly to the paper's
published results.**

### 9.1 Completeness of the uncertainty signal

- **Only two of the three signal families are included.** The
  input-level covariate signal (geophysical percentile ranks against the
  pretraining distribution) is entirely absent (§4.3). The fused result
  here likely understates what the full three-signal method would
  achieve.
- **No threshold sweep.** The original method evaluates the
  predictive-uncertainty signal at several region-of-interest thresholds
  (from 0.5 down to 0.05) and can use several as separate features. Only
  a single fixed threshold (0.5) was used here, reducing the amount of
  information available to the fusion step.
- **Fewer candidate signals overall.** With only 4 total candidate
  features (vs. potentially dozens once the covariate signal's multiple
  sub-components are included), the correlation-filtering and
  feature-selection step preceding fusion had much less material to work
  with than in the original study.

### 9.2 Reference data for the embedding-familiarity signal

- **Reference set scale.** The original method judges how "familiar" an
  embedding is by comparing it to the model's full pretraining
  distribution (on the order of a million images). Here, for practical
  reasons, the reference set is just this task's own 485 training chips —
  three to four orders of magnitude smaller and far less diverse. This is
  the most likely explanation for the embedding-familiarity signals
  underperforming random guessing in §8: with so few reference points,
  "distance to the reference set" mostly reflects within-task variation
  rather than genuine unfamiliarity relative to pretraining.
- **Reduced cluster granularity**, as a direct consequence: 24 clusters
  were used instead of the original study's 64, since 64 clusters would
  leave an average of only ~8 reference points per cluster (too few for
  the statistic to be meaningful).

### 9.3 Model and training protocol

- **Simplified decoder.** A smaller, custom segmentation head was used in
  place of the original study's purpose-built decoder architecture.
- **Ensemble size not matched to the paper** (the paper does not disclose
  an exact member count); a conventional default of 5 members was used.
- **Training regime not matched to the paper** — epochs, learning rate,
  and any data augmentation were chosen independently, since the original
  study's full training configuration isn't published.
- **Single training run.** The original study repeats the entire
  training-and-evaluation pipeline across 30 independent random seeds and
  reports results as a mean with a confidence band (e.g. its headline
  Risk@0.5 = 0.090 ± 0.038). This reproduction reports a **single run**
  with no uncertainty estimate of its own — the numbers in §7 could shift
  somewhat under a different random seed, and that variability is
  unmeasured here.

### 9.4 Dataset and evaluation scope

- **Independently drawn data split.** The public release of this dataset
  does not preserve the original paper's exact train/validation/test
  assignment, so a new one was drawn (§2.2). Even signals computed
  identically to the paper would not be evaluated on the same held-out
  chips.
- **Single task.** The original study evaluates across three tasks (burn
  scar segmentation, flood mapping, landslide detection) to support its
  claim that the method generalizes across tasks. Only burn scars are
  covered here.
- **Single backbone.** Only one frozen foundation model (SSL4EO-S12
  ViT-S/16) was tested. The original study's broader claim — that the
  method works with *any* frozen model — is not yet exercised here.

### 9.5 Input preprocessing

- **Partial spectral coverage.** This dataset provides 6 of the 13
  spectral bands the pretrained backbone expects; the missing 7 were
  filled with zeros rather than any informed estimate.
- **Reduced spatial resolution.** Chips were downsampled from their
  native 512x512 to the backbone's required 224x224 input size.

### 9.6 Implementation provenance

The uncertainty-signal, fusion, and evaluation-metric computations in
this reproduction were implemented directly from the equations given in
the paper, because the authors' own code for these specific components
(as opposed to data loading and model training scaffolding) is not
publicly released. The implementation has therefore not been
cross-checked against an official reference implementation.

### Net effect

Taken together, these deviations mean the **internal comparison in this
experiment — fused score vs. each individual signal vs. random (§8) — is
methodologically sound and directly meaningful**, since every row in §7's
table was computed under the exact same conditions. **Comparing the
absolute numbers to the original paper's published results is not
meaningful** — the two experiments differ in signal completeness,
reference-set scale, training protocol, data split, and evaluation scope.

## 10. Suggested Next Steps

- Source the input-level covariate data (DEM elevation/slope,
  hydrological attributes) to add the third signal family and obtain a
  more complete reproduction.
- Broaden the embedding-OOD reference set — e.g. by pooling embeddings
  across the additional tasks (flood mapping, landslide detection) planned
  for later stages, or sourcing a genuine large-scale pretraining-style
  reference sample.
- Repeat the ensemble training and evaluation across multiple random
  seeds to obtain a confidence interval on these results, comparable to
  the original study's seed-averaged protocol, rather than relying on a
  single run.
- Extend to the other two tasks (flood mapping, landslide detection) to
  test whether the fusion approach's benefit generalizes beyond burn
  scars.
- Test at least one additional frozen foundation model backbone to
  exercise the model-agnosticism claim, rather than relying on a single
  backbone.
- Add formal statistical guarantees (conformal prediction) on top of the
  fused risk score, and evaluate whether those guarantees hold uniformly
  across different geographic regions (a spatial-fairness check), which
  is the next planned stage of this work.
