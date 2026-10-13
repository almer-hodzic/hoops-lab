# Full Historical NBA Ingestion

## Status

PASS. The full historical PlayerSeason, PlayerTeamSplit, TeamSeason, and LeagueSeason datasets were built for NBA seasons 1979-80 through 2025-26.

## Source And Acquisition

- Source: Basketball-Reference league season pages.
- Programmatic endpoints:
  - `https://www.basketball-reference.com/leagues/NBA_{year}_totals.html`
  - `https://www.basketball-reference.com/leagues/NBA_{year}_advanced.html`
  - `https://www.basketball-reference.com/leagues/NBA_{year}.html`
- Raw HTML is cached under `data/historical/raw/basketball-reference/`.
- Raw cache files remain ignored by Git.
- Re-running without `--refresh` uses the cache and does not require live network access if all cached HTML files are present.
- Re-running with `--refresh` re-downloads source HTML using request delays, retry attempts, and local caching.

## Outputs

| Output | Rows |
| --- | ---: |
| `data/historical/processed/player_seasons.csv` | 20,526 |
| `data/historical/processed/player_team_splits.csv` | 4,803 |
| `data/historical/processed/team_seasons.csv` | 1,314 |
| `data/historical/processed/league_seasons.csv` | 47 |
| Unique players | 3,818 |

Sample files:

- `data/historical/processed/player_seasons_sample.csv`
- `data/historical/processed/team_seasons_sample.csv`
- `data/historical/processed/league_seasons_sample.csv`

Metadata:

- `data/historical/metadata/full_historical_ingestion_manifest.json`
- `data/historical/metadata/team_identity_map.json`

## PlayerSeason Construction

Primary player-season rows use one row per `player_id` and season. For players traded during a season, Basketball-Reference aggregate rows are used as the primary season row when available. Modern aggregate rows are marked as `TOT`; older historical aggregate rows such as `2TM` and `3TM` are also treated as season totals. Team-specific split rows are preserved separately in `player_team_splits.csv`.

The player table includes identity metadata, team/franchise metadata, box-score totals, Basketball-Reference advanced fields, league/team context, estimated possessions, per-75 rates, shooting rates, relative true shooting, sample flags, and era-relative percentile/z-score fields.

## Derived Metrics

- `estimated_possessions_minutes_team_pace = minutes / 48 * team_pace`
- `points_per_75 = points / estimated_possessions * 75`
- `assists_per_75 = assists / estimated_possessions * 75`
- `rebounds_per_75 = total_rebounds / estimated_possessions * 75`
- `turnovers_per_75 = turnovers / estimated_possessions * 75`
- `steals_per_75 = steals / estimated_possessions * 75`
- `blocks_per_75 = blocks / estimated_possessions * 75`
- `three_point_attempt_rate = 3PA / FGA`
- `free_throw_rate = FTA / FGA`
- `relative_ts = player_true_shooting_pct - league_ts_pct`

Nulls are preserved where source fields or denominators are unavailable.

## Era Context

Season-relative percentiles and z-scores are added for:

- `points_per_75`
- `assists_per_75`
- `rebounds_per_75`
- `relative_ts`
- `usage_pct`
- `assist_pct`
- `total_rebound_pct`
- `steals_per_75`
- `blocks_per_75`
- `turnovers_per_75`

Era context uses players with at least 500 minutes in the same season. Percentiles/z-scores are skipped when a season-metric population has fewer than 50 valid players.

## Sample Flags

`comparison_sample` values:

| Flag | Rows |
| --- | ---: |
| `qualified_rotation` | 11,268 |
| `comparison_population` | 3,370 |
| `low_sample` | 5,888 |

Rules:

- `qualified_rotation`: at least 1,000 minutes
- `comparison_population`: at least 500 minutes
- `low_sample`: below 500 minutes

## Missingness

Selected player-season missingness:

| Field | Missingness |
| --- | ---: |
| `usage_pct` | 0.01% |
| `turnover_pct` | 0.26% |
| `box_plus_minus` | 0.01% |
| `three_point_attempt_rate` | 0.35% |
| `free_throw_rate` | 0.35% |
| `relative_ts` | 0.31% |

The small shooting-rate missingness is caused by zero-FGA seasons or unavailable denominators.

## Star Sanity Checks

| Player | Season | Team | Minutes | Points/75 | Relative TS | USG% |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Kareem Abdul-Jabbar | 1979-80 | LAL | 3,143 | 22.38 | 0.108 | 24.1 |
| Larry Bird | 1985-86 | BOS | 3,113 | 24.17 | 0.039 | 27.6 |
| Michael Jordan | 1990-91 | CHI | 3,034 | 32.02 | 0.071 | 32.9 |
| Shaquille O'Neal | 1999-00 | LAL | 3,163 | 28.59 | 0.055 | 31.2 |
| Tim Duncan | 2002-03 | SAS | 3,181 | 23.69 | 0.045 | 28.0 |
| LeBron James | 2012-13 | MIA | 2,877 | 28.09 | 0.105 | 30.2 |
| Stephen Curry | 2015-16 | GSW | 2,700 | 31.89 | 0.128 | 32.6 |
| Nikola Jokic | 2022-23 | DEN | 2,323 | 26.70 | 0.120 | 27.2 |

## Validation

Command:

```powershell
python scripts\historical\validate_full_historical_dataset.py
```

Result:

```json
{
  "status": "pass",
  "errors": [],
  "row_counts": {
    "player_seasons": 20526,
    "player_team_splits": 4803,
    "team_seasons": 1314,
    "league_seasons": 47,
    "unique_players": 3818
  },
  "season_range": {
    "start": "1979-80",
    "end": "2025-26"
  }
}
```

Validation checks include season coverage, duplicate primary keys, summary-row leakage, formula spot checks, league true-shooting reconstruction, percentile bounds, Unicode name integrity, traded-player split linkage, franchise continuity, manifest row counts, and representative star seasons.

## Known Limitations

- Possessions are estimated from minutes and team pace, not reconstructed from play-by-play.
- Team pace and advanced player fields are sourced from Basketball-Reference where available and should be treated as source-derived features.
- Very low-FGA seasons can produce high free-throw rates; these are preserved because they are mathematically correct.
- This is regular-season historical player/team context. It does not ingest historical playoff player-season rows.

## Next Step

This dataset is ready to support the historical simulator design phase, but no simulator, frontend, or production model changes were made in this task.
