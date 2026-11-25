# Uncertainty Card — Flood mapping, U-Net (from scratch, 5-member ensemble), West Africa (Ghana, Nigeria) held out

## Recommendation (abstain / defer)
- Accepting every prediction: expected failure rate ~0.348.
- Accepting only the 50% most-confident predictions (deferring the rest to review): ~0.068.
- Accepting the 75% most-confident: ~0.159.

## Selective-risk summary
- **aurc**: 0.120
- **risk_at_0.5**: 0.068
- **risk_at_0.75**: 0.159
- **audp**: 0.770

## Signal attribution
- Best single signal: **roi_entropy**
- Fusion improves over best single signal: **True**

| Signal | AURC |
|---|---|
| roi_entropy | 0.155 |
| roi_mutual_information | 0.216 |

## Coverage under geographic shift
- Target coverage: 0.9

| Calibrated on | Coverage (overall) | Event disparity |
|---|---|---|
| on other continents | 0.953 | 0.003 |
| in distribution | 0.939 | 0.020 |

## Conformal risk control
- Target: pixel false-negative rate <= 0.2
- calibrated_on_other_continents: lambda = 0.25, mean FNR on evaluation set = 0.359
- calibrated_in_distribution: lambda = 0.15, mean FNR on evaluation set = 0.309

