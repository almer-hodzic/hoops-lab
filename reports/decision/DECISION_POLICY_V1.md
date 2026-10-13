# Decision Policy v1

## Method
Uses the existing Shooter Skill v1.1 `historical_shooter_profile` artifact as the shot-value engine. The shooter model is not retrained. A sigmoid probability calibrator is fit on the 2024-25 late validation split because only the base historical-profile artifact exists on disk.

Decision quantities: `p_make`, `shot_xpps = p_make * shot_value`, `replacement_xpps`, and `decision_margin = shot_xpps - replacement_xpps`.

## Replacement Approaches
| Approach      | Validation MAE | Validation RMSE | Threshold | Objective |
| ------------- | -------------- | --------------- | --------- | --------- |
| global        | 1.1570         | 1.2032          | 0.15      | 0.6839    |
| game_context  | 1.1557         | 1.2026          | 0.15      | 0.6780    |
| team_context  | 1.1556         | 1.2037          | 0.15      | 0.6412    |
| context_model | 1.1540         | 1.2015          | 0.15      | 0.6768    |

Selected replacement approach: `global`
Selected REVIEW threshold: `0.15`

## Frozen Evaluation
| Dataset                | Decision | Rows   | Share | Mean shot xPPS | Mean repl xPPS | Observed PPS | Mean margin |
| ---------------------- | -------- | ------ | ----- | -------------- | -------------- | ------------ | ----------- |
| 2025_26_regular        | TAKE     | 44043  | 0.201 | 1.514          | 1.081          | 1.503        | 0.433       |
| 2025_26_regular        | REVIEW   | 118680 | 0.542 | 1.077          | 1.081          | 1.073        | -0.004      |
| 2025_26_regular        | DEFER    | 56435  | 0.258 | 0.803          | 1.081          | 0.808        | -0.278      |
| 2025_26_regular_clutch | TAKE     | 1109   | 0.156 | 1.518          | 1.081          | 1.508        | 0.437       |
| 2025_26_regular_clutch | REVIEW   | 3468   | 0.489 | 1.049          | 1.081          | 1.023        | -0.032      |
| 2025_26_regular_clutch | DEFER    | 2522   | 0.355 | 0.786          | 1.081          | 0.754        | -0.295      |
| 2025_26_playoffs       | TAKE     | 3286   | 0.227 | 1.504          | 1.081          | 1.416        | 0.423       |
| 2025_26_playoffs       | REVIEW   | 7889   | 0.545 | 1.083          | 1.081          | 1.016        | 0.002       |
| 2025_26_playoffs       | DEFER    | 3298   | 0.228 | 0.820          | 1.081          | 0.739        | -0.261      |
| 2025_26_playoff_clutch | TAKE     | 109    | 0.229 | 1.553          | 1.081          | 1.321        | 0.473       |
| 2025_26_playoff_clutch | REVIEW   | 209    | 0.439 | 1.048          | 1.081          | 0.938        | -0.033      |
| 2025_26_playoff_clutch | DEFER    | 158    | 0.332 | 0.802          | 1.081          | 0.576        | -0.279      |

## Legacy Q/Bandit Comparison
Online epsilon-greedy Q-table over binned clutch observations; cold start randomizes Shoot/DontShoot until minimum visits; DontShoot reward uses fixed baseline PPP.

Decision Policy v1 estimates expected value directly from pre-shot probability and contextual replacement, while the legacy bandit learns action values from historical reward observations and exploration.

## Possession Continuation Assessment
PBP has ordered events, scores, teams, players, shot/free-throw/rebound/turnover/foul/substitution/timeout events, so a future possession-continuation model is feasible.

Limitations: Possession boundaries require explicit derivation from rebounds, turnovers, made shots, free throws, period boundaries, and team changes.; The current canonical shot table is event-level; it does not yet include possession IDs or next-shot continuation outcomes.; Some PBP score correction behavior is documented and should remain quality-flagged in any possession model.