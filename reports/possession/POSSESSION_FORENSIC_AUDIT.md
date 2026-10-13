# Possession Forensic Audit

## Target Semantics
Decision Policy v2 trains the continuation model on `possession_points` from `shot_possession_links.parquet`, which is copied from the full parsed possession's `points_scored` value. That target includes the current evaluated shot when the shot scores, includes free throws kept in that possession, and includes points after offensive rebounds if the parser keeps the possession open.

Classification: `CURRENT_POSSESSION_OUTCOME_MODEL`. It is not a true defer/continue counterfactual and should not be described as expected value of deferring the shot.

Future information used as target: final possession score, events after the shot, offensive-rebound continuations, and any free throws assigned to the possession. Future leakage into v2 features is limited but not zero-risk: `shot_attempt_number_in_possession` is produced by grouping final shot links and can encode whether this is a later attempt in a possession; the model does not use final possession points, final FGA count, or post-shot elapsed time as features.

## Possession Count Sanity
Mean inferred possessions per team/game: `87.06`; median `87.00`; p05 `77.00`; p95 `98.00`.
Independent estimate mean per team/game: `98.57`; median `98.28`; p05 `90.36`; p95 `107.89`.
Mean absolute difference: `11.52` possessions per team/game.

| Season | Type | Game | Team | Inferred | Estimate | Diff |
| --- | --- | --- | --- | --- | --- | --- |
| 2024-25 | regular_season | 22400867 | 1610612757 | 83.0 | 112.4 | -29.4 |
| 2021-22 | regular_season | 22100078 | 1610612738 | 89.0 | 117.0 | -28.0 |
| 2023-24 | regular_season | 22301113 | 1610612744 | 71.0 | 99.0 | -28.0 |
| 2022-23 | regular_season | 22200398 | 1610612761 | 62.0 | 89.7 | -27.7 |
| 2022-23 | regular_season | 22200570 | 1610612751 | 78.0 | 105.2 | -27.2 |
| 2025-26 | regular_season | 22501074 | 1610612742 | 76.0 | 103.2 | -27.2 |
| 2024-25 | regular_season | 22400254 | 1610612754 | 86.0 | 113.0 | -27.0 |
| 2020-21 | regular_season | 22000616 | 1610612761 | 71.0 | 98.0 | -27.0 |
| 2025-26 | regular_season | 22500260 | 1610612743 | 79.0 | 105.9 | -26.9 |
| 2021-22 | regular_season | 22100436 | 1610612747 | 79.0 | 105.8 | -26.8 |
| 2025-26 | regular_season | 22500135 | 1610612762 | 81.0 | 107.6 | -26.6 |
| 2020-21 | regular_season | 22000707 | 1610612754 | 86.0 | 112.5 | -26.5 |
| 2024-25 | regular_season | 22400479 | 1610612750 | 73.0 | 99.4 | -26.4 |
| 2024-25 | playoffs | 42400406 | 1610612760 | 73.0 | 99.4 | -26.4 |
| 2025-26 | regular_season | 22500649 | 1610612762 | 83.0 | 109.3 | -26.3 |

## Anomalies
Negative durations: `7698`. Primary causes: period transition closure using the first row of the next period, and clock increases within preserved action order from corrections/replay/same-clock administrative events.
Ordering errors: `2015`. Primary cause: action-number reset/period transition closure with the next-period row.
>8 point possessions: `168`. Categories: `{"multiple_field_goal_attempts": 3, "many_free_throw_attempts": 4, "ended_at_period_boundary": 19, "had_offensive_rebound": 38}`.

## Free Throws and Rebounds
The current parser does not explicitly model final free throw possession endings, and it closes made field goals before checking for same-clock and-one free throws. Team rebounds represented with `teamId == 0` and team ID in `personId` require special handling. These are parser-design issues, not just noisy rows.

Offensive-rebound validation from raw PBP: {"miss_dreb_other_team": 515014, "miss_oreb_same_team": 212053}

## Counterfactual Limitation
NBA PBP records events that happened. It does not record passes, pump fakes, declined open shots, or comparable unchosen shot opportunities. Therefore true empirical defer labels are not observable from this PBP alone.

## Continuation Model Value
HGB MAE improvement over global: `0.0088` absolute, `0.75%` relative.
HGB RMSE improvement over global: `0.0043`.
Ridge MAE `1.1520` vs HGB MAE `1.1521`; ridge is simpler and essentially tied.

## V1 vs V2 Methodology
V1 is a transparent shot xPPS versus replacement benchmark and does not claim to observe a defer counterfactual. V2 adds contextual possession information, but the current target is full possession outcome including the current shot, so V2 is not scientifically defensible as a defer policy.

## Decision
`USE V2 AS CONTEXTUAL BENCHMARK ONLY` until a corrected possession parser and honestly named contextual target are used. Do not adopt v2 as a production defer/continue policy.
