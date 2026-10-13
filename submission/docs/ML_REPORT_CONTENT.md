# ML Report Content

## Dataset

Primary dataset is `data/processed/shots_enriched.parquet` with 1,368,990 NBA shot attempts, 7,737 games, 1,086 shooters, seasons 2020-21, 2021-22, 2022-23, 2023-24, 2024-25, 2025-26, regular season plus playoffs.

## Problem Definition

Binary probability estimation: predict whether a shot is made before the outcome is known. The decision layer converts probability into expected points per shot.

## Data Preparation

The project uses chronological train/validation/test splits, reconstructed pre-shot scores, canonical shot geometry, categorical shot zones/actions, and leakage-safe historical shooter priors.

## Model Experiments

Tested baselines, Logistic Regression, HistGradientBoosting, raw player ID, historical shooter profile, and calibrated variants.

## Final Model

HistGradientBoosting with historical shooter profile plus raw player ID and sigmoid calibration.

## Metrics Table

| Model | Feature config | Calibration | Log Loss | Brier | ROC AUC |
| --- | --- | --- | ---: | ---: | ---: |
| hist_gradient_boosting | historical_shooter_profile | sigmoid | 0.63579 | 0.22343 | 0.66653 |
| hist_gradient_boosting | historical_profile_plus_raw_id | sigmoid | 0.63585 | 0.22346 | 0.66647 |
| hist_gradient_boosting | generic | sigmoid | 0.63813 | 0.22450 | 0.65964 |
| hist_gradient_boosting | raw_player_id | sigmoid | 0.63880 | 0.22479 | 0.65874 |
| baseline_shot_value | generic | sigmoid | 0.67364 | 0.24036 | 0.59276 |
| baseline_global | generic | sigmoid | 0.69147 | 0.24916 | 0.50000 |

Final 2025-26 regular season: Log Loss 0.63585, Brier 0.22346, ROC AUC 0.66647, actual FG% 0.4710, predicted FG% 0.4720.

## Main Graphs

See `submission/report_assets/01_target_distribution.png` through `12_confusion_matrix.png`.

## Final Observations

The model is best evaluated as a calibrated probability model, not as a hard classifier. The TAKE/REVIEW/DEFER layer is an expected-value policy; observed PPS by decision is descriptive rather than causal.
