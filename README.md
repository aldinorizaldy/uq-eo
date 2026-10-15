# eouq — Earth Observation Uncertainty Quantification

Model-agnostic, multi-layer uncertainty quantification for Earth observation
foundation models. The goal: carry uncertainty end-to-end from sensor input
to decision, work with any model (including frozen foundation models), say
*why* a prediction is uncertain, and give spatially fair guarantees —
coverage that holds per region, not just on average.

## Layered design

Every layer returns a standard `UncertaintyObject` (`eouq/core/uo.py`):
`{value, type, source, scale, guarantee, provenance}`, so fusion, reporting,
and the Uncertainty Card can treat all signals uniformly regardless of which
layer or method produced them.

| Layer | Question | Methods |
|---|---|---|
| L0 Input | Is the input unusual or noisy? | Geophysical covariate OOD: percentile rank + spatial KDE density (`eouq/layers/l0_input.py`) |
| L1 Representation | Is the sample familiar to the model? | Embedding OOD: k-means prototype distance, NCDD, kNN, Mahalanobis (`eouq/layers/l1_representation.py`) |
| L2 Prediction | How confident is the output? | Ensemble mean/entropy/mutual information, temperature scaling (`eouq/layers/l2_prediction.py`) |
| L3 Guarantee | What can we promise? | Split / Mondrian / localized conformal prediction, conformal risk control (`eouq/layers/l3_guarantee.py`) |
| L4 Aggregation | How uncertain is the area total? | Spatial block bootstrap / conformal on per-event totals (not yet implemented) |
| L5 Decision & governance | What happens and how is it reported? | Abstain/defer, attribution, Uncertainty Card (not yet implemented) |

The core takes probabilities (required), embeddings (optional), and
coordinates/region labels (optional) — it never touches model internals.
Model-specific code lives in thin adapters (`adapters/`): `ssl4eo_vit.py`,
`prithvi.py`, `unet.py`, all implementing the same `Adapter.extract()`
contract (`adapters/base.py`).

Pipeline stages: (1) train/fine-tune the decoder or ensemble, (2) cache
per-chip outputs (probs, embeddings, coords, F1) to disk, (3) run all UQ
layers, fusion, and metrics on the cache — cheap and CPU-only.

## SHRUG-FM reproduction

L0/L1/L2 and the fusion/metrics stage reimplement the method from
*SHRUG-FM: Reliability-Aware Foundation Models for Earth Observation*
(arXiv:[2511.10370](https://arxiv.org/abs/2511.10370), code reference:
[github.com/vishalned/SHRUG-FM](https://github.com/vishalned/SHRUG-FM),
reference only — no code copied, no license in that repo):

- `eouq/layers/l1_representation.py`: FAISS k-means (k=64), nearest-centroid
  distance, NCDD.
- `eouq/layers/l2_prediction.py`: deep ensemble mean prob / entropy / mutual
  information, averaged over the predicted region of interest.
- `eouq/fusion/tree_fusion.py`: correlation filter → RFE → depth-3 decision
  tree predicting `y_fail = 1{F1 < tau}`.
- `eouq/metrics/selective_risk.py`: risk–coverage curve, AURC,
  discard–performance curve, AUDP, Risk@coverage, tie-permutation averaging.

A local, untracked `NOTES.md` (gitignored, not in this repo) keeps the full
paper/repo reconciliation, confirmed numbers, discrepancies, and every
default/assumption made along the way.

## Repo layout

```
eouq/
  core/uo.py                  UncertaintyObject
  layers/l0_input.py          covariate OOD (percentile ranks, KDE)
  layers/l1_representation.py k-means/NCDD, kNN, Mahalanobis
  layers/l2_prediction.py     ensemble/MC-dropout stats, TTA, temperature scaling
  layers/l3_guarantee.py      split / Mondrian / localized conformal (stub)
  fusion/tree_fusion.py       SHRUG-FM-style tree fusion
  metrics/selective_risk.py   risk-coverage, AURC, AUDP
  metrics/fairness.py         coverage per group + disparity (stub)
  report/                     Uncertainty Card (stub)
adapters/                     model adapters: ssl4eo_vit.py, prithvi.py, unet.py
data/                         dataset loaders + covariate extraction (not yet built)
configs/                      YAML experiment configs
scripts/                      train/extract/evaluate entry points
slurm/                        sbatch templates
container/eouq.def            Apptainer image definition
tests/                        unit tests on synthetic data (CPU-only, fast)
```

(a local, gitignored `NOTES.md` at the repo root keeps the running log of
assumptions and paper-detail decisions — not part of this repo's history)

## Setup

Build the Apptainer image (login node only — pulls image layers and pip
installs, no GPU/heavy compute):

```
apptainer build --fakeroot container/eouq.sif container/eouq.def
```

Run the GPU smoke test (confirms torch sees the A100):

```
sbatch slurm/smoke_test.sbatch
```

Run the unit test suite (CPU-only, synthetic data):

```
sbatch slurm/run_tests.sbatch
```

Cluster specifics (partition, data paths, etc.) are in the local `NOTES.md`.

## Status

Setup and the SHRUG-FM-style L0/L1/L2/fusion/metrics logic are implemented
and unit-tested on synthetic data. Still open: decoder/ensemble training
code, the three dataset loaders, the SSL4EO adapter's `extract()`, and
sourcing real data (Burn Scars download, pretraining reference proxy) — see
the local `NOTES.md` for details and current decisions.
