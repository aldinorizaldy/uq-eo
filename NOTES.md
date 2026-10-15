# NOTES

Running log of environment assumptions, defaults, and paper-detail decisions.
Claude cannot execute Slurm/Apptainer/Python in this container — all entries
below are either confirmed by the user or a documented default pending
confirmation. Update this file whenever a new assumption is made.

## Cluster environment (confirmed by user, 2026-10-15)

- Partition: `gpu-a100`.
- Slurm account: not required on this cluster (no `--account` flag needed).
- Apptainer: `module load apptainer` (already in `slurm/smoke_test.sbatch`).
- Data/checkpoint storage: **no scratch dir** — stick to this repo directory.
  `data/` and `outputs/` under the repo root are bind-mounted directly
  (see `slurm/smoke_test.sbatch`, `container/eouq.def`). Do not introduce a
  scratch path.

## Open / assumed (not yet confirmed)

- **Compute-node internet access**: unconfirmed. Assuming NO internet access
  on compute nodes until told otherwise — datasets/HF checkpoints should be
  downloaded via a login/data-transfer step and bind-mounted, not fetched
  inside an sbatch job. Flag this again before writing any download step.
- **Max job wall-clock time**: unconfirmed cluster-wide limit. Smoke test
  uses `--time=00:10:00`, which is small enough to be safe regardless. Will
  ask again when a training job is scoped (Milestone 2).

## Paper-detail decisions (SHRUG-FM reproduction, Milestone 2)

Read against arXiv:2511.10370v2 (HTML) and github.com/vishalned/SHRUG-FM
(src/ tree, no license file — reference only, no code copied) on 2026-10-15.

Confirmed, matches original design exactly:
- Backbone: SSL4EO-S12 ViT-S/16, MoCo-pretrained, ~23M params, frozen,
  inputs resized to 224.
- Decoder: CNN decoder, ~41M params.
- U_E: FAISS k-means, k=64. NCDD formula:
  `U_E^NCDD(x) = sum_{j != c(x)} d_j(x) - (k-1) * d_c(x)(x)`
  where `d_j(x)` is distance to cluster j's centroid and `c(x)` is the
  nearest cluster. Low/negative value = OOD.
- U_T: mean prob / entropy / mutual information over ensemble members; ROI
  = pixels with mean prob >= threshold, threshold swept 0.5 down to 0.05.
- Fusion: correlation filter -> RFE -> decision tree, max_depth=3.
  y_fail = 1{F1 < tau}: tau=0.6 (burn scars), tau=0.6 (floods),
  tau=0.5 (landslides).
- Metrics: 100 tie-break permutations, 30 seeds for confidence bands.
- Burn scars splits: 804 chips, 485/54/263 train/val/test.
- Target numbers (sanity check once we have real results): burn scars
  Risk@0.5 = 0.090 +/- 0.038 (best single-score baseline, MI, = 0.137);
  WorldFloods AURC = 0.115 +/- 0.001; landslides AURC = 0.392 +/- 0.035
  (vs. aleatoric-entropy-alone baseline 0.526, not in original brief).

New details / corrections vs. the original brief:
- **Ensemble type is fixed per task, not a free choice**: paper uses
  bootstrap ensembles for burn scars, standard ensembles for floods and
  landslides (chosen by validation performance). Decision: default
  `ensemble_type="bootstrap"` for burn scars, `"standard"` for
  floods/landslides in config.
- **Burn Scars band count corrected**: original brief/initial paper-summary
  pass said "66 spectral bands" for ExEBench Burn Scars — this is a misread
  of "6 bands". Cross-checked against the repo's `dataset.toml`
  (`wavelengths = [0.48, 0.56, 0.65, 0.865, 1.61, 2.2]`, 6 values) and the
  known public HLS Burn Scars band set. Decision: use 6 HLS bands
  (blue/green/red/NIR-narrow/SWIR1/SWIR2), not 66.
- **U_I covariates are more specific**: HydroATLAS at 500m resolution +
  Copernicus DEM GLO-30 at 30m, 21 features total (paper Table 5), not a
  generic "elevation/slope/land cover" list. Decision: start with a small
  subset we can source easily (DEM elevation/slope only) and expand later;
  full 21-feature HydroATLAS join is deferred until we scope data access.
  Flagged, not blocking Milestone 2 code.
- **WorldFloods split is in 256x256 patches, not scenes**: 509 global S2
  L1C scenes tile into 39,060 / 1,854 / 1,936 train/val/test patches.
  Relevant for cache sizing if/when we reach Milestone 4 (Prithvi check
  uses Sen1Floods11, but WorldFloods numbers are needed earlier for
  sanity-checking AURC against the paper).
- **The public repo does not contain the core methodology code.** `src/`
  has embedding extraction, dataset loaders, decoder+ensemble
  model/training/testing scaffolding, and a calibration-diagram plotter —
  but no k-means/NCDD, no percentile-rank/KDE, no correlation-filter+RFE+
  tree fusion, and no risk-coverage/AURC/AUDP code (only 5 commits on
  main). `environment.yml` lists faiss-cpu/scikit-learn/hdbscan/h3,
  consistent with the method, but those modules were never pushed. All of
  eouq's L0/L1/L2/fusion/metrics logic is implemented from the paper's
  formulas directly, not adapted from repo code.

## Dataset sourcing (found while reading the repo, 2026-10-15)

- **Burn scars** (Milestone 2 target, smallest dataset): repo comment
  points to the HuggingFace dataset `ibm-nasa-geospatial/hls_burn_scars`
  (public, no auth needed). Estimated size: ~1-3 GB for 804 chips of
  512x512, 6 bands. Not yet downloaded — need your go-ahead on exact size
  and target path (per working rule: no multi-GB download without asking).
- **WorldFloods**: repo reads from a public GCS bucket
  `gs://ml4cc_data_lake/2_PROD/2_Mart/worldfloods_v1_0/...` (ml4floods
  project). Likely tens of GB for the full train/val/test patch set.
  Needed only from Milestone 4 onward in the original plan (and Milestone
  4 actually targets Sen1Floods11 + Prithvi, not WorldFloods) — deferring
  download entirely until that milestone is scoped.
- **Landslide**: no public URL found in the repo; "28 single events" reads
  like a restricted/unreleased research dataset. Lowest priority (not in
  Milestone 2 or 4) — unresolved, will ask again if/when we get there.
- **Pretraining reference set — DECIDED (2026-10-15)**: needed for U_I
  percentile/KDE and U_E k-means, which score a sample against the
  *pretraining* distribution, not the downstream dataset. The real
  SSL4EO-S12 pretraining corpus (~1M S1/S2 patches, hundreds of GB) is
  infeasible to download here. Default: build the reference/proxy set from
  the **pooled train splits of our own three downstream datasets** (burn
  scars + floods + landslides train sets, ~740 chips combined) — run the
  same frozen SSL4EO-S12 ViT-S/16 encoder over them for embeddings (U_E)
  and pull whatever DEM/covariate values we can source for those locations
  (U_I). This is not the true pretraining distribution, so OOD scores will
  be systematically smaller/noisier than the paper's (less diverse
  reference set) — acceptable for Milestone 2 since we're reproducing the
  *method*, not claiming the exact paper numbers. Revisit with real
  SSL4EO-S12 metadata (if a lightweight location/embedding index is ever
  published separately from the full imagery) only if this proxy turns out
  to track the paper's AURC/Risk@k numbers too poorly to be useful.

## Implementation status (2026-10-15)

L0/L1/L2 layers, fusion, and selective-risk metrics are implemented
(pure NumPy/scikit-learn/FAISS, CPU-only, no execution needed to write
them) with synthetic-data unit tests in `tests/`. Not yet run — no python
available in this container; run `pytest` yourself.

One stub signature changed from the original scaffold, both in
`eouq/layers/l1_representation.py`:
- `ncdd_score(query_embeddings, centroids)` — dropped the unused
  `reference_distances` param. The paper's NCDD formula is self-contained
  (only needs distances to all centroids of the query point itself), so
  there was nothing for that argument to normalise against.
- Added `normalized_centroid_distance(query_embeddings, centroids,
  cluster_mean_distance)` — this is U_E^nd from the paper (nearest-centroid
  distance normalised by the reference mean distance within the assigned
  cluster), a distinct signal from NCDD that the original stub's module
  docstring mentioned but had no dedicated function for.

Still open before Milestone 2 can run for real: the decoder/ensemble
training code, the three dataset loaders, the SSL4EO adapter's `extract()`,
and the Burn Scars download (pretraining-proxy approach is now decided
above, just not built yet).
