# Canonical Dataset Report

## Methodology
The pipeline reads immutable compressed raw archives from `data/raw/`, reconstructs running scores from full Play-by-Play V3 event streams, matches Shot Detail rows by `GAME_ID + GAME_EVENT_ID` to PBP `gameId + actionNumber`, and writes generated Parquet outputs under `data/processed/` and `data/interim/`.

## Output Files
- `data/processed/shots_enriched.parquet`
- `data/processed/clutch_shots.parquet`
- `data/interim/unmatched_shots.parquet`
- `data/metadata/canonical_dataset_summary.json`
- `data/metadata/feature_policy.json`

## Score Reconstruction
PBP events are sorted by game and action number. The current home/away score is captured before each event, then updated from available post-event score fields. This includes free throws and other scoring events, not only field goals.

## Clutch Definition
`is_clutch` is true for Q4 with 5:00 or less remaining or any overtime period, with absolute pre-shot score differential <= 5.

## Split Methodology
TRAIN: 2020-21 through 2023-24; VALIDATION: 2024-25; TEST: 2025-26. Both regular season and playoff rows are preserved with `season_type`.

## Match Rates
| season  | season_type    | shot_rows | canonical_rows | unmatched_rows | match_rate_pct |
| ------- | -------------- | --------- | -------------- | -------------- | -------------- |
| 2020-21 | regular_season | 190983    | 190983         | 0              | 100.0          |
| 2020-21 | playoffs       | 14771     | 14771          | 0              | 100.0          |
| 2021-22 | regular_season | 216722    | 216722         | 0              | 100.0          |
| 2021-22 | playoffs       | 14571     | 14571          | 0              | 100.0          |
| 2022-23 | regular_season | 217218    | 217217         | 1              | 99.99954       |
| 2022-23 | playoffs       | 14651     | 14651          | 0              | 100.0          |
| 2023-24 | regular_season | 218701    | 218700         | 1              | 99.999543      |
| 2023-24 | playoffs       | 13840     | 13840          | 0              | 100.0          |
| 2024-25 | regular_season | 219527    | 219527         | 0              | 100.0          |
| 2024-25 | playoffs       | 14377     | 14377          | 0              | 100.0          |
| 2025-26 | regular_season | 219160    | 219158         | 2              | 99.999087      |
| 2025-26 | playoffs       | 14473     | 14473          | 0              | 100.0          |

## Clutch Counts
| season  | season_type    | rows |
| ------- | -------------- | ---- |
| 2020-21 | playoffs       | 428  |
| 2020-21 | regular_season | 6696 |
| 2021-22 | playoffs       | 452  |
| 2021-22 | regular_season | 7562 |
| 2022-23 | playoffs       | 510  |
| 2022-23 | regular_season | 8188 |
| 2023-24 | playoffs       | 479  |
| 2023-24 | regular_season | 7232 |
| 2024-25 | playoffs       | 515  |
| 2024-25 | regular_season | 7176 |
| 2025-26 | playoffs       | 476  |
| 2025-26 | regular_season | 7099 |

## Leakage Considerations
The canonical table contains labels/outcomes such as `shot_made` and `points_scored`; future modeling must exclude them from input features. Event descriptions, post-shot scores, and PBP shot result fields are intentionally not included in the canonical feature table.

## Known Limitations
4 Shot Detail events are unmatched and excluded from canonical rows requiring verified pre-shot game state.