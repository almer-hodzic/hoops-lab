# Shot Quality Benchmark

## Research Questions
Estimate pre-shot field-goal make probability, assess clutch calibration, and test regular-season to playoff generalization.

## Training Data
Primary models train only on 2020-21 through 2023-24 regular-season rows. Validation is 2024-25 regular season; final test is 2025-26 regular season. Playoffs are stress tests only.

## Feature Sets
- Generic Shot Quality: shot geometry, shot zones, action/shot type, period/time, home/away, pre-shot score context, season.
- Shooter-Aware Shot Quality: Generic features plus categorical `player_id` with unknown-player handling.

Leakage controls exclude `shot_made`, `points_scored`, post-shot scores, descriptions, PBP result fields, future events, and `is_clutch` from primary model inputs.

## Models
Global empirical baseline, shot-value empirical baseline, shot-zone empirical baseline, logistic regression, and histogram gradient boosting. XGBoost/LightGBM/CatBoost were not added because sklearn models already establish the benchmark without extra dependencies.

## Validation Results
| Model                  | Feature config | Calibration | Log loss | Brier   | ROC AUC | ECE     |
| ---------------------- | -------------- | ----------- | -------- | ------- | ------- | ------- |
| baseline_global        | generic        | none        | 0.69100  | 0.24893 | 0.50000 | 0.00213 |
| baseline_global        | generic        | sigmoid     | 0.69100  | 0.24893 | 0.50000 | 0.00000 |
| baseline_global        | generic        | isotonic    | 0.69100  | 0.24893 | 0.50000 | 0.00000 |
| baseline_shot_value    | generic        | none        | 0.67415  | 0.24062 | 0.59051 | 0.00386 |
| baseline_shot_value    | generic        | sigmoid     | 0.67411  | 0.24060 | 0.59051 | 0.00003 |
| baseline_shot_value    | generic        | isotonic    | 0.67411  | 0.24060 | 0.59051 | 0.00000 |
| baseline_zone_value    | generic        | none        | 0.65767  | 0.23266 | 0.63296 | 0.00437 |
| baseline_zone_value    | generic        | sigmoid     | 0.65762  | 0.23264 | 0.63296 | 0.00214 |
| baseline_zone_value    | generic        | isotonic    | 0.65760  | 0.23263 | 0.63299 | 0.00000 |
| logistic_regression    | generic        | none        | 0.64161  | 0.22612 | 0.65236 | 0.00975 |
| logistic_regression    | generic        | sigmoid     | 0.64147  | 0.22606 | 0.65236 | 0.00580 |
| logistic_regression    | generic        | isotonic    | 0.64041  | 0.22562 | 0.65283 | 0.00000 |
| hist_gradient_boosting | generic        | none        | 0.63683  | 0.22408 | 0.65818 | 0.00203 |
| hist_gradient_boosting | generic        | sigmoid     | 0.63681  | 0.22407 | 0.65818 | 0.00149 |
| hist_gradient_boosting | generic        | isotonic    | 0.63642  | 0.22393 | 0.65860 | 0.00000 |
| baseline_global        | shooter_aware  | none        | 0.69100  | 0.24893 | 0.50000 | 0.00213 |
| baseline_global        | shooter_aware  | sigmoid     | 0.69100  | 0.24893 | 0.50000 | 0.00000 |
| baseline_global        | shooter_aware  | isotonic    | 0.69100  | 0.24893 | 0.50000 | 0.00000 |
| baseline_shot_value    | shooter_aware  | none        | 0.67415  | 0.24062 | 0.59051 | 0.00386 |
| baseline_shot_value    | shooter_aware  | sigmoid     | 0.67411  | 0.24060 | 0.59051 | 0.00003 |
| baseline_shot_value    | shooter_aware  | isotonic    | 0.67411  | 0.24060 | 0.59051 | 0.00000 |
| baseline_zone_value    | shooter_aware  | none        | 0.65767  | 0.23266 | 0.63296 | 0.00437 |
| baseline_zone_value    | shooter_aware  | sigmoid     | 0.65762  | 0.23264 | 0.63296 | 0.00214 |
| baseline_zone_value    | shooter_aware  | isotonic    | 0.65760  | 0.23263 | 0.63299 | 0.00000 |
| logistic_regression    | shooter_aware  | none        | 0.64032  | 0.22553 | 0.65646 | 0.01129 |
| logistic_regression    | shooter_aware  | sigmoid     | 0.64008  | 0.22542 | 0.65646 | 0.00688 |
| logistic_regression    | shooter_aware  | isotonic    | 0.63940  | 0.22516 | 0.65686 | 0.00000 |
| hist_gradient_boosting | shooter_aware  | none        | 0.63745  | 0.22434 | 0.65752 | 0.00960 |
| hist_gradient_boosting | shooter_aware  | sigmoid     | 0.63724  | 0.22424 | 0.65752 | 0.00139 |
| hist_gradient_boosting | shooter_aware  | isotonic    | 0.63684  | 0.22410 | 0.65790 | 0.00000 |

## Selected Model
hist_gradient_boosting / generic / calibration=isotonic. Selection prioritized validation log loss, Brier score, calibration, and stress-test practicality.

## Final / Stress-Test Metrics For Selected Model
| Dataset                 | Log loss | Brier   | ROC AUC | Actual FG% | Pred FG% | Actual PPP | Pred xPPS |
| ----------------------- | -------- | ------- | ------- | ---------- | -------- | ---------- | --------- |
| validation_regular      | 0.63642  | 0.22393 | 0.65860 | 0.4672     | 0.4672   | 1.0861     | 1.0890    |
| test_regular            | 0.63831  | 0.22442 | 0.66000 | 0.4710     | 0.4698   | 1.0913     | 1.0929    |
| validation_clutch       | 0.62759  | 0.21965 | 0.66710 | 0.4376     | 0.4390   | 1.0107     | 1.0166    |
| test_clutch             | 0.62733  | 0.21869 | 0.67776 | 0.4375     | 0.4403   | 1.0031     | 1.0162    |
| test_non_clutch         | 0.63868  | 0.22461 | 0.65917 | 0.4721     | 0.4708   | 1.0943     | 1.0954    |
| playoffs_2024_25        | 0.64217  | 0.22606 | 0.64740 | 0.4551     | 0.4584   | 1.0556     | 1.0664    |
| playoffs_2025_26        | 0.63747  | 0.22415 | 0.65502 | 0.4517     | 0.4729   | 1.0438     | 1.0947    |
| playoffs_2024_25_clutch | 0.63204  | 0.22201 | 0.63661 | 0.4311     | 0.4234   | 1.0155     | 0.9866    |
| playoffs_2025_26_clutch | 0.60488  | 0.20943 | 0.69073 | 0.3992     | 0.4624   | 0.9055     | 1.0501    |

## Calibration
Sigmoid and isotonic calibration were fit on validation predictions only and evaluated on held-out test/stress datasets. Figures are under `reports/modeling/figures/`.

## Segment Findings
Segment tables are stored under `data/analysis/segment_*.csv`. The most interpretable slices are shot value, distance bucket, shot zone, home/away, score state, and Q4/OT context.

## Player Analysis
`data/analysis/player_actual_vs_expected_test_regular.csv` includes players with at least 250 test regular-season attempts.

## Feature Importance
`data/analysis/feature_importance_selected_model.csv` contains permutation importance on a validation sample. Interpret as predictive utility, not causality.

## Limitations
The benchmark uses event-level pre-shot features only; no defender distance, lineup context, fatigue, play design, or shot-clock features are available. Player identity improves fit but may reduce interpretability of pure shot quality. Playoff rows are stress tests, not training data.

## Recommendation
Use the selected calibrated probability model as the expected-points engine for the next decision-agent stage. The decision layer should consume `predicted_make_probability` and `expected_points`, not hard classifications.