# Uncertainty Card — Burn scars (SSL4EO-S12 ViT-S/16, 5-member bootstrap ensemble)

## Recommendation (abstain / defer)
- Accepting every prediction: expected failure rate ~0.281.
- Accepting only the 50% most-confident predictions (deferring the rest to review): ~0.057.
- Accepting the 75% most-confident: ~0.110.

## Selective-risk summary
- **aurc**: 0.096
- **risk_at_0.5**: 0.057
- **risk_at_0.75**: 0.110
- **audp**: 0.765

## Signal attribution
- Best single signal: **roi_entropy**
- Fusion improves over best single signal: **True**

| Signal | AURC |
|---|---|
| ncdd | 0.387 |
| normalized_centroid_distance | 0.264 |
| roi_entropy | 0.214 |
| roi_mutual_information | 0.286 |

## Coverage guarantees
### split_conformal
- Target coverage: 0.9
- Empirical coverage (overall): 0.887
- Coverage disparity across regions: 0.177

| Region | Coverage |
|---|---|
| NE | 0.917 |
| NW | 0.864 |
| SE | 0.944 |
| SW | 0.768 |

### mondrian_conformal
- Target coverage: 0.9
- Empirical coverage (overall): 0.888
- Coverage disparity across regions: 0.113

| Region | Coverage |
|---|---|
| NE | 0.909 |
| NW | 0.891 |
| SE | 0.911 |
| SW | 0.798 |

### conformal_fusion
- Target coverage: 0.9
- Empirical coverage (overall): 0.887
- Coverage disparity across regions: 0.175

| Region | Coverage |
|---|---|
| NE | 0.916 |
| NW | 0.860 |
| SE | 0.947 |
| SW | 0.772 |

### localized_conformal
- Target coverage: 0.9
- Empirical coverage (overall): 0.891
- Coverage disparity across regions: 0.153

| Region | Coverage |
|---|---|
| NE | 0.912 |
| NW | 0.879 |
| SE | 0.935 |
| SW | 0.782 |

## Area totals (L4)
- Per-district conformal intervals (2.0-degree cells): coverage 0.911 over 79 districts (half-width 7563 px vs. mean district total 11219 px).
- Grand total: predicted 586125 px, true 886296 px, block-bootstrap interval [428466, 755230] -- true total inside: False.

## Conformal risk control
- Target: pixel false-negative rate <= 0.2
- Selected threshold (lambda): 0.25
- Mean FNR on evaluation set: 0.155

