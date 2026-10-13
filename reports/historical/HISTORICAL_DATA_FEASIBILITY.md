# Historical Player Data Feasibility Spike

## Summary

Recommendation: **READY TO SCALE** for the next ingestion step, with a controlled scope.

The prototype now fixes the data-foundation issues found in audit:

- League context is derived from team-level season tables, not player rows.
- Basketball-Reference source player IDs are extracted from cached HTML links.
- Unicode player names survive processed CSV output.
- Team identity separates team-season identity from franchise identity.
- Sample-relative scoring has been removed.
- Possession estimates are explicitly marked as estimates.
- Field provenance and missingness metadata are written to the manifest.

The recommended HoopsLab V1 historical start remains **1979-80**, the first NBA season with the 3-point line. Research mode can still include 1976-77 through 1978-79 with null-aware handling and visible caveats.

## Outputs

- `data/historical/processed/player_season_prototype.csv`
- `data/historical/processed/player_season_comparison.csv`
- `data/historical/processed/team_season_prototype.csv`
- `data/historical/metadata/historical_player_data_manifest.json`
- `data/historical/metadata/team_identity_map.json`

Raw source HTML is cached under ignored path:

- `data/historical/raw/basketball-reference/`

## Source Acquisition

Source: Basketball-Reference season pages.

URL patterns:

- `https://www.basketball-reference.com/leagues/NBA_{season_end_year}_totals.html`
- `https://www.basketball-reference.com/leagues/NBA_{season_end_year}_advanced.html`
- `https://www.basketball-reference.com/leagues/NBA_{season_end_year}.html`

The pipeline is cached-first. It reads existing cached HTML unless `--refresh` is passed. If live acquisition is needed, it uses:

- 45 second request timeout
- 3 download attempts
- bounded exponential backoff
- 3 second conservative request pacing after successful downloads
- source URL/cache status recorded in the manifest

The current prototype uses 10 season pages groups and 30 possible HTML requests on a cold cache. The latest validation run used cache hits.

## Actual Historical Data

Directly sourced from Basketball-Reference player totals:

- player display name
- season
- team abbreviation
- age
- position
- games
- games started when available
- minutes
- points
- FGA/FG
- FTA/FT
- 2P/2PA where tracked
- 3P/3PA where tracked
- ORB/DRB/TRB
- assists
- steals
- blocks
- turnovers when available
- personal fouls

Directly sourced from Basketball-Reference team tables:

- wins/losses
- team pace
- ORtg/DRtg/NRtg
- team TS%
- team 3PAr
- team TOV%
- team ORB%
- team points/assists/rebounds per game

## Source-Derived Advanced Metrics

The following are Basketball-Reference-derived model/stat metrics. HoopsLab does not independently calculate them in this spike:

- PER
- WS
- BPM
- USG%
- AST%
- TOV%
- TRB%

They are kept with provenance:

```json
"source_derived_basketball_reference"
```

Older eras can still be unavailable or partially unavailable depending on source inputs. Missing values remain null.

## Reconstructed League Context

League context is now reconstructed from team-level tables, not player-level rows. This prevents traded-player `TOT` plus team-split rows from affecting league environment metrics.

Fields:

- `league_points`
- `league_fga`
- `league_fta`
- `league_ts_pct`
- `league_pace`
- `league_points_per_75`

League TS formula:

```text
league_TS = league_points / (2 * (league_fga + 0.44 * league_fta))
```

Inputs:

- league points = sum of team season PTS
- league FGA = sum of team season FGA
- league FTA = sum of team season FTA

League pace:

```text
league_pace = sum(team_pace * team_games) / sum(team_games)
```

League scoring environment:

```text
league_points_per_75 = league_points / sum(team_pace * team_games) * 75
```

This is team scoring environment per 75 estimated team possessions. It is not subtracted from individual player points per 75 because that would mix player scoring with team scoring units.

## Derived Metrics

Estimated possessions:

```text
estimated_possessions_minutes_team_pace = (minutes / 48) * team_pace
```

Provenance:

```text
estimated_minutes_team_pace
```

This is not measured player on-court possessions and must not be treated as tracking/play-by-play possession data.

Per-75 player statistics:

```text
stat_per_75 = stat / estimated_possessions_minutes_team_pace * 75
```

Implemented:

- `points_per_75`
- `assists_per_75`
- `rebounds_per_75`
- `turnovers_per_75`
- `steals_per_75`
- `blocks_per_75`

Efficiency and shot mix:

```text
relative_ts = player_ts - league_ts
three_point_attempt_rate = 3PA / FGA
free_throw_rate = FTA / FGA
```

Sample-based percentiles, z-scores, and sample-relative scoring are intentionally not implemented.

## Identity Model

### Player Identity

Basketball-Reference player IDs are extracted from cached HTML player links such as:

```text
/players/j/jokicni01.html
```

Canonical fields:

- `player_id`: `br:{source_player_id}`
- `source_player_id`: Basketball-Reference ID, e.g. `jokicni01`
- `player_name`: HoopsLab canonical display name
- `source_player_name`: source display name with Unicode preserved

Validation checks that each representative player resolves to the expected stable source ID.

### Team / Franchise Identity

The prototype writes an explicit team identity map:

- `team_id`
- `franchise_id`
- `display_name`
- source abbreviation

This distinguishes team-season identity from franchise identity. Historical relocation/name-change examples represented in the map include:

- Seattle SuperSonics / Oklahoma City Thunder
- New Jersey Nets / Brooklyn Nets
- Charlotte Hornets/Bobcats historical cases
- Vancouver Grizzlies / Memphis Grizzlies
- New Orleans Hornets/Pelicans naming cases

## Missingness / Availability Model

Missing historical fields remain null.

Availability metadata distinguishes:

- `recorded`
- `derived`
- `estimated`
- `not_tracked_in_era`
- `source_unavailable_or_not_tracked`

Kareem Abdul-Jabbar 1976-77 still demonstrates this behavior:

- 3PAr is null because the NBA 3-point line did not exist.
- turnover-derived fields are null where source inputs are unavailable.

Zero is used only when the stat existed and was recorded as zero.

## Prototype Coverage

Representative player-seasons loaded:

| Player | Season | Player ID | Team | Status |
|---|---:|---|---|---|
| Kareem Abdul-Jabbar | 1976-77 | br:abdulka01 | LAL | loaded |
| Larry Bird | 1985-86 | br:birdla01 | BOS | loaded |
| Michael Jordan | 1990-91 | br:jordami01 | CHI | loaded |
| Shaquille O'Neal | 1999-00 | br:onealsh01 | LAL | loaded |
| LeBron James | 2012-13 | br:jamesle01 | MIA | loaded |
| Stephen Curry | 2015-16 | br:curryst01 | GSW | loaded |
| Nikola Jokic | 2022-23 | br:jokicni01 | DEN | loaded |

Representative team-seasons loaded:

| Team | Season | Team ID | Franchise ID | Status |
|---|---:|---|---|---|
| Boston Celtics | 1985-86 | team:boston-celtics | franchise:boston-celtics | loaded |
| Chicago Bulls | 1995-96 | team:chicago-bulls | franchise:chicago-bulls | loaded |
| Los Angeles Lakers | 2000-01 | team:los-angeles-lakers | franchise:lakers | loaded |
| Miami Heat | 2012-13 | team:miami-heat | franchise:miami-heat | loaded |
| Golden State Warriors | 2016-17 | team:golden-state-warriors | franchise:warriors | loaded |

## Validation

Commands:

```powershell
python scripts\historical\build_player_season_prototype.py
python scripts\historical\validate_player_season_prototype.py
```

Result:

```json
{
  "ok": true,
  "player_rows": 7,
  "team_rows": 5,
  "schema_version": "historical_player_season_prototype_v2"
}
```

Validation covers:

- Representative player rows loaded.
- Representative team rows loaded.
- Stable Basketball-Reference player IDs.
- Unicode source names.
- No duplicate PlayerSeason rows.
- No duplicate TeamSeason rows.
- League TS formula from team totals.
- Protection from traded-player duplicate rows affecting league TS.
- Possession estimate formula.
- Per-75 formula.
- Missing historical stats remain null.
- Invalid sample-relative scoring field removed.
- Team/franchise mapping includes relocation/name-change cases.
- Field provenance metadata exists.

## Remaining Limitations

- Basketball-Reference HTML is still not a formal API. The cache/retry/schema checks reduce risk but do not remove it.
- Full player identity across all seasons still needs a complete cross-season crosswalk audit.
- Team identity mapping should be expanded to all historical abbreviations before full ingestion.
- League-relative percentiles/z-scores require complete season populations and are intentionally deferred.
- The possession estimate remains approximate and should later be replaced or supplemented when play-by-play possessions are available.

## Next Step

Scale ingestion for 1979-80 through the latest completed season only after adding the full-season target list and broadening team identity coverage. The next implementation should still be data-only:

1. Fetch/cache all required season pages.
2. Build complete `Player`, `PlayerSeason`, `Team`, and `TeamSeason` tables.
3. Add full player-season identity audit.
4. Compute full-season league-relative distributions.
5. Add all-season validation reports.

