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

## 9. Methodological Limitations and Scope of Comparison

This reproduction differs from the original SHRUG-FM study in several
ways that affect how the numbers above should be interpreted:

1. **No input-level covariate signal (§4.3)** — only two of the three
   original signal families are included. The fused result likely
   understates what the full method would achieve.
2. **Narrow embedding-OOD reference set** — the original method compares
   each chip against the model's full *pretraining* distribution (on the
   order of a million images). Here, for practical reasons, the reference
   set is just this task's own 485 training chips — a far smaller and
   less diverse stand-in. This is the most likely explanation for the
   embedding-OOD signals underperforming random in §8: with so few
   reference points, "distance to the reference set" mostly reflects
   within-task variation rather than genuine unfamiliarity relative to
   pretraining.
3. **Independent train/val/test split** — the public release of this
   dataset does not preserve the original paper's exact split, so a new
   one was drawn. Absolute numbers are therefore not directly comparable
   to the published results even where the same signals are used.
4. **Simplified decoder** — a smaller, custom segmentation head was used
   in place of the original study's purpose-built decoder architecture.
5. **Input adaptation** — this dataset's 6 spectral bands had to be mapped
   into the 13-band input the pretrained backbone expects (missing bands
   filled with zeros), and chips were downsampled from 512x512 to the
   backbone's native 224x224 input size. Both are reasonable engineering
   choices but introduce information loss not present in the original
   study.

Given these differences, **the fused-vs-single-signal comparison within
this experiment (§8) is methodologically sound and directly meaningful.
Comparing absolute numbers to the original paper's published results is
not** — the two experiments differ in signal completeness, reference-set
scale, data split, and model architecture.

## 10. Suggested Next Steps

- Source the input-level covariate data (DEM elevation/slope,
  hydrological attributes) to add the third signal family and obtain a
  more complete reproduction.
- Broaden the embedding-OOD reference set — e.g. by pooling embeddings
  across the additional tasks (flood mapping, landslide detection) planned
  for later stages, or sourcing a genuine large-scale pretraining-style
  reference sample.
- Extend to the other two tasks (flood mapping, landslide detection) to
  test whether the fusion approach's benefit generalizes beyond burn
  scars.
- Add formal statistical guarantees (conformal prediction) on top of the
  fused risk score, and evaluate whether those guarantees hold uniformly
  across different geographic regions (a spatial-fairness check), which
  is the next planned stage of this work.
