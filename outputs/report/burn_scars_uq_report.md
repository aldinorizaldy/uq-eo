# Reliability-Aware Uncertainty Quantification for Earth Observation Segmentation

An exploratory reproduction of the SHRUG-FM framework (Gonzalez-Calabuig et al.,
*SHRUG-FM: Reliability-Aware Foundation Models for Earth Observation*,
arXiv:2511.10370), first on burn-scar segmentation (§2–9), then on flood
mapping with two very different models as a model-agnosticism check (§10).

## 1. Objective

Geospatial foundation models (GFMs) often fail silently in conditions
underrepresented during pretraining. SHRUG-FM's approach is to combine
several independent uncertainty signals — how unfamiliar the embedding
looks to the model, and how uncertain the model's own ensemble is — into a
single, interpretable risk score that predicts when a model's output is
likely to fail, so that low-confidence predictions can be flagged or
deferred rather than trusted blindly.

This experiment reproduces that approach end-to-end on burn scar
segmentation as a first validation step, using a frozen Earth observation
foundation model, then tests whether the same unchanged pipeline works on
a second task (flood mapping) with two models of very different kinds — a
large, already fine-tuned foundation model and a small network trained
from scratch.

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

Reference set used: the 485 training chips' own embeddings (see §11 for
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
  about the method itself — see §11.

## 9. Formal Guarantees: Conformal Prediction

As a further validation step beyond selective risk/AURC, we checked
whether formal statistical coverage guarantees (conformal prediction)
hold on top of the same model, and whether they hold *uniformly* across
the geography covered by the test set — a spatial-fairness check, since
average performance can look good while performance in some areas is
quietly much worse.

### 9.1 Method

A fresh 50/50 split of the previously held-out test set (263 chips) was
used: one half for conformal calibration, the other for final evaluation,
keeping this check fully independent of both the ensemble training and
the fusion-model fitting above. Three variants were compared:

- **Split conformal**: a single global threshold calibrated to target 90%
  coverage.
- **Mondrian conformal**: a separate threshold per geographic quadrant
  (chips grouped by whether they fall north/south and east/west of the
  median chip location), testing whether region-specific calibration
  improves spatial fairness.
- **Conformal fusion**: the same global approach as split conformal, but
  with the nonconformity score rescaled by each chip's predictive-entropy
  difficulty, testing whether accounting for a chip's inherent prediction
  difficulty improves coverage.

A fourth method, **conformal risk control**, separately calibrated a
probability threshold for classifying pixels as "burned" to guarantee the
pixel-level false-negative rate (missed burned pixels) stays below a
20% target.

### 9.2 Results

Target coverage: 90%.

| Method | Overall coverage | Best region | Worst region | Coverage disparity ↓ |
|---|---|---|---|---|
| Split conformal | 88.7% | 94.4% (SE) | 76.8% (SW) | 0.177 |
| Mondrian conformal | 88.8% | 91.1% (SE) | 79.8% (SW) | **0.113** |
| Conformal fusion | 88.7% | 94.7% (SE) | 77.2% (SW) | 0.175 |

Conformal risk control: target false-negative rate ≤ 20%; achieved 15.5%
on the evaluation half.

### 9.3 Interpretation

- **A single global coverage threshold is not spatially fair.** Split
  conformal hits its overall 90% target on average, but coverage ranges
  from 76.8% in the worst-covered region to 94.4% in the best — one
  region is substantially under-served relative to the rest, something
  the overall-coverage number alone completely hides.
- **Region-specific calibration (Mondrian conformal) measurably improves
  this.** Coverage disparity drops by about a third (0.177 → 0.113) when
  the threshold is calibrated separately per region instead of globally —
  direct evidence that *where* a chip is matters for how well-calibrated
  its guarantee is, and that accounting for this explicitly helps,
  echoing the same spatial-fairness concern that motivates the broader
  framework.
- **Difficulty-normalization alone does not fix geographic unfairness.**
  The conformal-fusion variant — rescaling by how uncertain the model's
  own prediction is — performs almost identically to the plain global
  approach. This is an informative negative result: a signal capturing
  *how hard a prediction is* is not a substitute for a signal capturing
  *where it comes from*; the under-coverage in the worst region looks like
  a systematic geographic effect, not just a byproduct of generically
  harder chips happening to be there.
- **The risk-control guarantee held with room to spare**, a useful
  complementary result showing the same cached outputs can support a
  different kind of formal guarantee (bounding one specific error type)
  rather than only general coverage.

## 10. Model-Agnostic Check: Flood Mapping with Two Different Models

A central claim of the framework is that it works with *any* model: the
uncertainty pipeline consumes only a model's output probabilities (and,
where available, its internal representations), never its internals. To
test this, the identical pipeline from §4–6 — same signals, same fusion,
same metrics — was run on a second task with two models that differ in
almost every respect. Only the thin model-specific "adapter" layer changed.

### 10.1 Data

**Sen1Floods11** (Cloud to Street): 446 hand-labeled Sentinel-2 chips
(512x512, 10 m resolution) from 11 flood events across 5 continents, with
water / not-water / no-data masks. The same 6 spectral bands as the burn
scar data were used — here a true subset of the full 13-band product, not
a zero-filled approximation.

To test robustness to geographic distribution shift, an entire continent
was held out: **all African chips (Ghana, Nigeria, Somalia) form the test
set** (71 chips); the rest are split 338 train / 37 validation.

### 10.2 The two models

| | Prithvi-EO-2.0 | U-Net |
|---|---|---|
| Type | Large foundation model (~300M parameters) | Small CNN trained from scratch |
| Training | Already fine-tuned for flood mapping by its authors; used frozen | Trained here on the 338 training chips |
| Source of "ensemble" uncertainty | Test-time augmentation: the same model run on 5 rotated/flipped views of each chip | A true ensemble: 5 independently trained networks |
| Representation (embedding) signal | Available | Not applicable — no pretraining to be familiar or unfamiliar with |

Both achieved similar segmentation quality (mean per-chip F1 on the
African test set: Prithvi 0.59, U-Net 0.61).

### 10.3 Results

Both models were scored on the same 46 test chips (chips where neither a
flood was present nor predicted have no defined F1 and are excluded).

| Risk signal | Prithvi AURC ↓ | U-Net AURC ↓ |
|---|---|---|
| **Fused** | **0.246** | **0.120** |
| Predictive entropy (alone) | 0.401 | 0.155 |
| Mutual information (alone) | 0.255 | 0.216 |
| Normalized centroid distance (alone) | 0.255 | — |
| NCDD (alone) | 0.450 | — |
| Random discard (≈ base failure rate) | 0.389 | 0.345 |

AUDP (↑): fused Prithvi 0.683 (mutual information alone: 0.690); fused
U-Net **0.770** (best single: 0.752).

### 10.4 Interpretation

- **The pipeline is model-agnostic in practice.** It ran unchanged on a
  frozen 300M-parameter foundation model and on a small network trained
  from scratch, degrading gracefully where a signal didn't apply (no
  embedding signal for the U-Net) rather than breaking.
- **The benefit of fusion depends on the quality of the underlying
  signals.** For the U-Net, fusion again clearly beats every individual
  signal (AURC ~22% below the best single signal), reproducing the burn
  scar result. For Prithvi, fusion is only marginally better than its best
  single signal on AURC and slightly worse on AUDP.
- **Test-time augmentation appears to be a weak substitute for a real
  ensemble.** Despite similar base failure rates, Prithvi's best fused risk
  score is about twice as poor as the U-Net's. A plausible explanation —
  not yet tested — is that five rotated or flipped views of a single model
  disagree far less, and less informatively, than five independently
  trained models. Notably, Prithvi's predictive entropy is no better than
  random, while its mutual information (disagreement between views) is
  informative.
- **The embedding-familiarity signal behaves differently here.** On burn
  scars it was no better than random (§8); with Prithvi's own fine-tuned
  representations it is among the best individual signals. This is
  consistent with the explanation in §11.2 that the burn scar result
  reflected a poor reference set rather than a flaw in the signal itself.

## 11. Deviations from the Original Study

This reproduction differs from the original SHRUG-FM study in several
ways, grouped below by what they affect. None of these were accidental —
each is a deliberate scoping choice made to get a working first
reproduction with accessible data and compute — but together they mean
**absolute numbers here should not be compared directly to the paper's
published results.**

### 11.1 Completeness of the uncertainty signal

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

### 11.2 Reference data for the embedding-familiarity signal

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

### 11.3 Model and training protocol

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

### 11.4 Dataset and evaluation scope

- **Independently drawn data split.** The public release of this dataset
  does not preserve the original paper's exact train/validation/test
  assignment, so a new one was drawn (§2.2). Even signals computed
  identically to the paper would not be evaluated on the same held-out
  chips.
- **Two of three tasks.** The original study evaluates burn scars, flood
  mapping, and landslide detection. Burn scars (§2–9) and flood mapping
  (§10) are covered; landslides are not. The flood mapping dataset also
  differs from the original study's (Sen1Floods11 rather than
  WorldFloods), so §10 is not a reproduction of the paper's flood result.
- **Model-agnosticism tested on two flood-mapping models** (§10), but the
  burn scar results (§7–9) rest on a single backbone (SSL4EO-S12 ViT-S/16).

### 11.5 Input preprocessing

- **Partial spectral coverage.** This dataset provides 6 of the 13
  spectral bands the pretrained backbone expects; the missing 7 were
  filled with zeros rather than any informed estimate.
- **Reduced spatial resolution.** Chips were downsampled from their
  native 512x512 to the backbone's required 224x224 input size.

### 11.6 Implementation provenance

The uncertainty-signal, fusion, and evaluation-metric computations in
this reproduction were implemented directly from the equations given in
the paper, because the authors' own code for these specific components
(as opposed to data loading and model training scaffolding) is not
publicly released. The implementation has therefore not been
cross-checked against an official reference implementation.

### 11.7 Conformal prediction design choices (§9)

- **Region definition is a coarse geographic quadrant split** (north/south
  x east/west of the median chip location), not a meaningful
  administrative or climatic boundary. It's a self-contained
  simplification for testing the *concept* of region-aware calibration,
  not an attempt to identify which real-world regions are under-served.
- **Coverage was checked on a random subsample of pixels per chip** (50
  per chip), not every pixel, to keep this check's computational cost
  modest. Reported coverage figures are therefore estimates, not an exact
  enumeration over every pixel.
- **Single calibration/evaluation split.** The §9 results reflect one
  random 50/50 split of the test set, not an average over repeated
  splits. Whether the 0.177-vs-0.113 disparity gap between split and
  Mondrian conformal is a stable effect or partly an artifact of this
  particular split has not been established.
- **The 90% coverage and 20% false-negative-rate targets are illustrative
  choices**, not figures drawn from a published SHRUG-FM conformal
  experiment — unlike the AURC/Risk@k comparison in §7, there is no
  equivalent published number to sanity-check §9's results against.

### 11.8 Model-agnostic check design choices (§10)

- **Test-time augmentation instead of an ensemble for Prithvi.** The
  Prithvi checkpoint is a single, already fine-tuned model, so the
  ensemble-disagreement signals come from five geometric views of one
  model rather than five independently trained models. This is a
  different — and, per §10.4, apparently weaker — source of uncertainty
  than the deep ensembles the original method uses. The Prithvi-vs-U-Net
  comparison therefore partly compares *uncertainty sources*, not only
  *models*.
- **Possible train/test overlap for Prithvi.** The Prithvi checkpoint was
  fine-tuned by its authors on Sen1Floods11, using their own data split.
  Some of the African chips used here as the held-out test set may have
  been part of its training data. Its test results are therefore not a
  clean out-of-distribution measurement, unlike the U-Net's, whose
  training never saw any African chip. (That Prithvi nonetheless fails
  slightly *more* often than the U-Net on these chips makes this less
  likely to flatter it, but the concern stands.)
- **Prithvi's segmentation quality is lower than expected.** A mean
  per-chip F1 of ~0.6 is low for a model fine-tuned on this exact dataset,
  and has not yet been explained — it may reflect chips with very little
  water dragging the per-chip average down, or a remaining mismatch
  between our input preparation and the model's training setup. Until
  this is ruled out, Prithvi's absolute numbers should be treated with
  extra caution.
- **Small test set.** 46 chips, from a single held-out continent. AURC
  differences of a few hundredths between signals (e.g. Prithvi's fused
  0.246 vs. 0.255 for its best single signal) are within what sampling
  noise alone could produce. The large differences — U-Net vs. Prithvi
  fused (0.120 vs. 0.246), and signals vs. random — are the robust ones.
- **Continent assignment is approximate**, based on simple latitude/
  longitude bounding boxes rather than country boundaries.
- **Single run, single held-out continent.** Results reflect one training
  run per model and one choice of held-out continent (Africa); other
  continents were not tried.

### Net effect

Taken together, these deviations mean the **internal comparisons in this
experiment — fused score vs. each individual signal vs. random (§8, §10),
and split vs. Mondrian vs. conformal-fusion conformal calibration (§9) — are
methodologically sound and directly meaningful**, since every row in a
given comparison was computed under the exact same conditions. The
cross-model comparison in §10 is meaningful in direction but confounded by
the different uncertainty sources and possible train/test overlap noted
in §11.8. **Comparing
the absolute numbers to the original paper's published results is not
meaningful** — the two experiments differ in signal completeness,
reference-set scale, training protocol, data split, and evaluation scope.

## 12. Suggested Next Steps

- Investigate *why* the worst-covered region in §9 is systematically
  under-served — e.g. whether it has fewer training chips, different burn
  characteristics, or different imaging conditions — now that a real
  disparity has been found rather than assumed.
- Define regions using actual geographic or administrative boundaries
  instead of a quadrant split, and check whether the same disparity
  pattern holds.
- Repeat the conformal calibration/evaluation split across multiple
  random seeds to establish how stable the §9 coverage-disparity
  estimates are.
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
- Explain Prithvi's lower-than-expected segmentation quality (§11.8)
  before relying on its absolute numbers.
- Separate the two confounds in §10: train a small *ensemble* on top of
  the frozen Prithvi encoder, so Prithvi and U-Net can be compared with
  the same kind of uncertainty source; and test whether TTA is weaker
  than ensembling on the U-Net side too.
- Hold out other continents in turn, to see whether the §10 findings
  depend on which region is unseen.
- Extend to landslide detection, the one task from the original study
  not yet covered.
- Apply the §9 conformal analysis to the flood mapping models, where
  holding out a continent gives a natural, real-world test of whether
  coverage guarantees survive geographic shift.
