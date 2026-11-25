# Uncertainty Card — Flood mapping, Prithvi-EO-2.0 (frozen, test-time augmentation), West Africa (Ghana, Nigeria) held out

## Recommendation (abstain / defer)
- Accepting every prediction: expected failure rate ~0.435.
- Accepting only the 50% most-confident predictions (deferring the rest to review): ~0.237.
- Accepting the 75% most-confident: ~0.229.

## Selective-risk summary
- **aurc**: 0.246
- **risk_at_0.5**: 0.237
- **risk_at_0.75**: 0.229
- **audp**: 0.683

## Signal attribution
- Best single signal: **roi_mutual_information**
- Fusion improves over best single signal: **True**

| Signal | AURC |
|---|---|
| ncdd | 0.450 |
| normalized_centroid_distance | 0.255 |
| roi_entropy | 0.401 |
| roi_mutual_information | 0.255 |

## Coverage under geographic shift
- Target coverage: 0.9

| Calibrated on | Coverage (overall) | Event disparity |
|---|---|---|
| on other continents | 0.946 | 0.007 |
| in distribution | 0.945 | 0.006 |

## Conformal risk control
- Target: pixel false-negative rate <= 0.2
- calibrated_on_other_continents: lambda = 0.25, mean FNR on evaluation set = 0.326
- calibrated_in_distribution: lambda = 0.15, mean FNR on evaluation set = 0.261

