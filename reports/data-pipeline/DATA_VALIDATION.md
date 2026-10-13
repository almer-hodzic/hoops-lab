# Data Validation

Passed: `True`

## Hard Checks
```json
{
  "duplicate_identity": 0,
  "shot_value_invalid": 0,
  "points_invalid": 0,
  "score_negative": 0,
  "score_diff_invalid": 0,
  "score_mapping_invalid": 0,
  "clutch_invalid": 0,
  "split_invalid": 0,
  "critical_nulls": {
    "season": 0,
    "season_type": 0,
    "dataset_split": 0,
    "game_id": 0,
    "event_id": 0,
    "player_id": 0,
    "team_id": 0,
    "home_team_id": 0,
    "away_team_id": 0,
    "shot_value": 0,
    "home_score_before": 0,
    "away_score_before": 0,
    "score_diff_before": 0
  }
}
```

Coordinate extreme rows, report only: `1310`
Overall match rate: `99.999708%`
Unmatched rows: `4`
PBP shot value disagreements: `0`
Made FG score reconcile failures: `2265`
Miss FG score reconcile failures: `0`
Score decrease events: `1910`

## Pair Validation
| season  | season_type    | shot_rows | canonical_rows | unmatched_rows | match_rate_pct | pbp_shot_value_disagreements | made_fg_score_reconcile_failures | miss_fg_score_reconcile_failures | score_decrease_events | games_with_home_team_id | games_with_away_team_id |
| ------- | -------------- | --------- | -------------- | -------------- | -------------- | ---------------------------- | -------------------------------- | -------------------------------- | --------------------- | ----------------------- | ----------------------- |
| 2020-21 | regular_season | 190983    | 190983         | 0              | 100.0          | 0                            | 410                              | 0                                | 321                   | 1080                    | 1080                    |
| 2020-21 | playoffs       | 14771     | 14771          | 0              | 100.0          | 0                            | 37                               | 0                                | 36                    | 85                      | 85                      |
| 2021-22 | regular_season | 216722    | 216722         | 0              | 100.0          | 0                            | 384                              | 0                                | 309                   | 1230                    | 1230                    |
| 2021-22 | playoffs       | 14571     | 14571          | 0              | 100.0          | 0                            | 27                               | 0                                | 22                    | 87                      | 87                      |
| 2022-23 | regular_season | 217218    | 217217         | 1              | 99.99954       | 0                            | 288                              | 0                                | 258                   | 1230                    | 1230                    |
| 2022-23 | playoffs       | 14651     | 14651          | 0              | 100.0          | 0                            | 21                               | 0                                | 20                    | 84                      | 84                      |
| 2023-24 | regular_season | 218701    | 218700         | 1              | 99.999543      | 0                            | 367                              | 0                                | 307                   | 1230                    | 1230                    |
| 2023-24 | playoffs       | 13840     | 13840          | 0              | 100.0          | 0                            | 14                               | 0                                | 6                     | 82                      | 82                      |
| 2024-25 | regular_season | 219527    | 219527         | 0              | 100.0          | 0                            | 319                              | 0                                | 269                   | 1230                    | 1230                    |
| 2024-25 | playoffs       | 14377     | 14377          | 0              | 100.0          | 0                            | 10                               | 0                                | 12                    | 84                      | 84                      |
| 2025-26 | regular_season | 219160    | 219158         | 2              | 99.999087      | 0                            | 371                              | 0                                | 328                   | 1230                    | 1230                    |
| 2025-26 | playoffs       | 14473     | 14473          | 0              | 100.0          | 0                            | 17                               | 0                                | 22                    | 85                      | 85                      |