# Shooter Skill v1.1 Experiment

## Method
Historical shooter-profile features are computed chronologically from shots strictly before the current attempt. The canonical dataset is read-only. Training uses 2020-21 through 2023-24 regular season only. The 2024-25 regular season is split chronologically: the earlier half is used for model selection, and the later half is used only for probability calibration. 2025-26 regular season and playoffs are final evaluations.

## Feature Configurations
- Generic: pre-shot geometry/context only.
- Raw player ID: Generic plus categorical `player_id`.
- Historical shooter profile: Generic plus prior FGA/FG%, 2P/3P priors, zone priors, distance priors, shrunk/Bayesian priors, season-to-date priors, previous-season profile, and cold-start flag.
- Historical profile plus raw ID: profile features plus categorical player ID.

## Model-Selection Results
| Model                  | Feature config                 | Calibration | Log loss | Brier   | AUC     | ECE     |
| ---------------------- | ------------------------------ | ----------- | -------- | ------- | ------- | ------- |
| baseline_global        | generic                        | none        | 0.69072  | 0.24879 | 0.50000 | 0.00442 |
| baseline_global        | generic                        | sigmoid     | 0.69073  | 0.24879 | 0.50000 | 0.00457 |
| baseline_global        | generic                        | isotonic    | 0.69073  | 0.24879 | 0.50000 | 0.00457 |
| baseline_shot_value    | generic                        | none        | 0.67429  | 0.24068 | 0.58931 | 0.00220 |
| baseline_shot_value    | generic                        | sigmoid     | 0.67432  | 0.24070 | 0.58931 | 0.00403 |
| baseline_shot_value    | generic                        | isotonic    | 0.67432  | 0.24070 | 0.58931 | 0.00400 |
| hist_gradient_boosting | generic                        | none        | 0.63620  | 0.22380 | 0.65765 | 0.00329 |
| hist_gradient_boosting | generic                        | sigmoid     | 0.63632  | 0.22386 | 0.65765 | 0.00769 |
| hist_gradient_boosting | generic                        | isotonic    | 0.63648  | 0.22392 | 0.65753 | 0.00752 |
| hist_gradient_boosting | raw_player_id                  | none        | 0.63616  | 0.22377 | 0.65850 | 0.00658 |
| hist_gradient_boosting | raw_player_id                  | sigmoid     | 0.63622  | 0.22380 | 0.65850 | 0.00771 |
| hist_gradient_boosting | raw_player_id                  | isotonic    | 0.63646  | 0.22390 | 0.65817 | 0.00751 |
| hist_gradient_boosting | historical_shooter_profile     | none        | 0.63450  | 0.22301 | 0.66383 | 0.01058 |
| hist_gradient_boosting | historical_shooter_profile     | sigmoid     | 0.63430  | 0.22292 | 0.66383 | 0.00448 |
| hist_gradient_boosting | historical_shooter_profile     | isotonic    | 0.63453  | 0.22301 | 0.66362 | 0.00547 |
| hist_gradient_boosting | historical_profile_plus_raw_id | none        | 0.63422  | 0.22288 | 0.66441 | 0.00775 |
| hist_gradient_boosting | historical_profile_plus_raw_id | sigmoid     | 0.63413  | 0.22284 | 0.66441 | 0.00449 |
| hist_gradient_boosting | historical_profile_plus_raw_id | isotonic    | 0.63450  | 0.22295 | 0.66426 | 0.00559 |

## Selected Model
hist_gradient_boosting / historical_profile_plus_raw_id / calibration=sigmoid

## Selected Model Metrics
| Dataset                          | Log loss | Brier   | AUC     | Actual FG% | Pred FG% | Actual PPP | Pred xPPS |
| -------------------------------- | -------- | ------- | ------- | ---------- | -------- | ---------- | --------- |
| selection_regular_2024_25_early  | 0.63413  | 0.22284 | 0.66441 | 0.4649     | 0.4690   | 1.0819     | 1.0913    |
| calibration_regular_2024_25_late | 0.63556  | 0.22349 | 0.66494 | 0.4695     | 0.4695   | 1.0903     | 1.0915    |
| full_validation_regular_2024_25  | 0.63485  | 0.22317 | 0.66470 | 0.4672     | 0.4692   | 1.0861     | 1.0914    |
| test_regular_2025_26             | 0.63585  | 0.22346 | 0.66647 | 0.4710     | 0.4720   | 1.0913     | 1.0956    |
| test_clutch_2025_26              | 0.62412  | 0.21782 | 0.68176 | 0.4375     | 0.4460   | 1.0031     | 1.0275    |
| test_non_clutch_2025_26          | 0.63624  | 0.22364 | 0.66576 | 0.4721     | 0.4728   | 1.0943     | 1.0978    |
| playoffs_2024_25                 | 0.64094  | 0.22580 | 0.65046 | 0.4551     | 0.4753   | 1.0556     | 1.1018    |
| playoffs_2025_26                 | 0.63723  | 0.22394 | 0.66073 | 0.4517     | 0.4849   | 1.0438     | 1.1189    |
| playoffs_2025_26_clutch          | 0.61281  | 0.21286 | 0.69451 | 0.3992     | 0.4779   | 0.9055     | 1.0805    |

## Cold Start
Cold-start rows in 2025-26 test regular season: 104; selected-model log loss on cold-start rows: 0.6608884722366312.

## Representative Player Sanity Checks
`data/analysis/shooter_skill_v11_player_sanity_checks.csv` contains high/mid/low predicted-probability representative players with attempts, actual FG%, predicted FG%, and prior profile summaries.

## Feature Importance
`data/analysis/shooter_skill_v11_feature_importance.csv` contains permutation importance. Interpret as predictive utility, not causality.

## Decision-Safe Conclusion
Historical shooter-profile features are decision-safe because each value is based on prior shots only. Raw player ID is included only as a comparison condition, not recommended as the primary decision-agent feature source unless maintainability and cold-start behavior are acceptable.