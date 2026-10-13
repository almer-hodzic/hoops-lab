# Defensive Context Data Audit

## Observed Shot-Level Data

- 2014/15 shot-log rows: 128,069
- Exact closest-defender rows: 128,069
- Closest-defender-distance coverage: 100.00%
- Shot-clock coverage: 95.65%
- Dribbles coverage: 100.00%
- Touch-time coverage: 100.00%

## Aggregated Defender Profile Data

- Matchup rows: 2,143,437
- Matchup seasons: 2017-18, 2018-19, 2019-20, 2020-21, 2021-22, 2022-23, 2023-24, 2024-25, 2025-26
- Matchup games: 11,391
- Matchup unique defenders: 1,396
- PT Defend rows: 53,129
- PT Defend successful seasons: 2014-15, 2015-16, 2016-17, 2017-18, 2018-19, 2019-20, 2020-21, 2021-22, 2022-23, 2023-24, 2024-25, 2025-26
- PT Defend successful categories: 2 Pointers, 3 Pointers, Greater Than 15Ft, Less Than 10Ft, Less Than 6Ft, Overall
- PT Defend failed downloads: 0
- PT Defend empty downloads: 0

## Inferred / Derived Features

- Leakage-safe defender prior rows: 2,143,437
- Prior formula: shifted expanding defender history; FG% and 3P% use Bayesian shrinkage with strength 50.0.

## Defender ID Matching

- Direct tracking defender ID match rate: 99.16%
- IDs in multiple sources: 1,579

## Data-Quality Problems

- Matchup duplicate canonical keys: 0
- Matchup FGM > FGA rows: 0
- Matchup 3PM > 3PA rows: 0
- Matchup FTM > FTA rows: 0
- Matchup time-share percentage values > 1.0: 2,491
- Shot-clock out-of-bounds rows: 0
- Negative touch-time rows: 312

## Feasibility

- Closest defender distance can be learned only from the 2014/15 event-level tracking shot logs.
- Modern defender profiles can be built from Player Matchups and LeagueDashPtDefend when available.
- Modern exact defender-per-shot data was not found; matchup archives are game-level player matchup aggregates, not shot-level closest-defender labels.
- Shot clock, dribbles, and touch time are available in the 2014/15 tracking dataset and prepared for a future experiment.
- Domain shift is material: tracking data is 2014/15, while production shot-quality data covers 2020/21-2025/26 with different shot mix, spacing, and player population.
- Safe next modeling experiment: compare 2014/15 shot-log models with and without closest defender distance, shot clock, dribbles, and touch time before any production integration.