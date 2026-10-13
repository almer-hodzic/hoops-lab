# Historical Player Data Feasibility Spike

## Summary

Recommendation: **GO WITH LIMITATIONS**.

HoopsLab can build a reliable historical `PlayerSeason` foundation for cross-era analysis and early simulation prototypes, provided the product preserves both raw historical production and era-relative context. The minimum practical V1 scope should start with the **1979-80 season**, the first NBA season with the 3-point line. A broader research mode can include **1976-77 through 1978-79**, but comparisons from those seasons must clearly mark 3-point and some possession/usage inputs as unavailable or structurally different.

This spike produced a reproducible prototype from Basketball-Reference season pages:

- `data/historical/processed/player_season_prototype.csv`
- `data/historical/processed/player_season_comparison.csv`
- `data/historical/processed/team_season_prototype.csv`
- `data/historical/metadata/historical_player_data_manifest.json`

Raw source HTML is cached under ignored path `data/historical/raw/basketball-reference/`.

## Sources Investigated

### Basketball-Reference

Status: **Used for prototype**.

Source pattern:

- `https://www.basketball-reference.com/leagues/NBA_{season_end_year}_totals.html`
- `https://www.basketball-reference.com/leagues/NBA_{season_end_year}_advanced.html`
- `https://www.basketball-reference.com/leagues/NBA_{season_end_year}.html`

Coverage:

- Broad NBA player season totals, team season tables, and advanced metrics.
- Pace is documented as available since 1973-74 in the Basketball-Reference glossary.
- Usage percentage is documented as available since 1977-78.
- True shooting percentage is available in advanced tables and can also be derived from PTS, FGA, and FTA.

Strengths:

- Broad historical coverage.
- Programmatically usable with `pandas.read_html`.
- Contains player totals, team pace/context, and advanced rate stats.
- Good fit for a portfolio feasibility spike.

Limitations:

- HTML table layout is not a formal API contract.
- Raw pages should be cached locally and regenerated rather than committed.
- Stable cross-source player IDs are not available from these tables alone.
- Older seasons have real stat availability gaps.

### Official NBA Stats / nba_api

Status: **Not used as the primary historical source**.

The `nba_api` project documents official NBA Stats endpoints such as `LeagueDashPlayerStats`, but official endpoints are more fragile operationally, often require NBA-specific headers, can rate-limit or block automation, and are not as convenient for full historical coverage. This source remains useful later for modern-era enrichment, not as the sole historical foundation.

### Public Open Datasets / Kaggle-Style Archives

Status: **Investigated conceptually, not used**.

Many public NBA stat archives exist, but common options require Kaggle/API credentials, manual license acceptance, or have unclear provenance and update behavior. Those can be optional import paths later, but they should not be required for HoopsLab V1 unless the user manually provides files.

### Existing Project NBA Shot/PBP Data

Status: **Not used for this spike**.

The current project data is modern shot-quality data and is valuable for Shot Lab. It does not provide the historical player-season coverage needed for all-time team building or cross-era simulation.

## Prototype Coverage

Representative player-seasons loaded:

| Player | Season | Team | Status |
|---|---:|---|---|
| Kareem Abdul-Jabbar | 1976-77 | LAL | loaded |
| Larry Bird | 1985-86 | BOS | loaded |
| Michael Jordan | 1990-91 | CHI | loaded |
| Shaquille O'Neal | 1999-00 | LAL | loaded |
| LeBron James | 2012-13 | MIA | loaded |
| Stephen Curry | 2015-16 | GSW | loaded |
| Nikola Jokic | 2022-23 | DEN | loaded |

Representative team-seasons loaded:

| Team | Season | Status |
|---|---:|---|
| Boston Celtics | 1985-86 | loaded |
| Chicago Bulls | 1995-96 | loaded |
| Los Angeles Lakers | 2000-01 | loaded |
| Miami Heat | 2012-13 | loaded |
| Golden State Warriors | 2016-17 | loaded |

## Canonical PlayerSeason Schema

### Raw / Source Fields

Current prototype fields:

- `player_id`: null for Basketball-Reference HTML prototype; future resolver needed.
- `player_name`
- `source_player_name`
- `season`
- `season_end_year`
- `team`
- `team_name`
- `age`
- `position`
- `games`
- `games_started`
- `minutes`
- `points`
- `field_goals`
- `field_goal_attempts`
- `two_point_makes`
- `two_point_attempts`
- `three_point_makes`
- `three_point_attempts`
- `free_throws`
- `free_throw_attempts`
- `offensive_rebounds`
- `defensive_rebounds`
- `total_rebounds`
- `assists`
- `steals`
- `blocks`
- `turnovers`
- `personal_fouls`
- `true_shooting_pct`
- `assist_pct`
- `turnover_pct`
- `total_rebound_pct`
- `usage_pct`
- `player_efficiency_rating`
- `win_shares`
- `box_plus_minus`
- `team_pace`
- `league_pace`
- `league_ts_pct`
- `source_status`
- `source`

### Derived Fields

Implemented:

- `estimated_on_court_possessions`
- `points_per_75`
- `assists_per_75`
- `rebounds_per_75`
- `turnovers_per_75`
- `steals_per_75`
- `blocks_per_75`
- `three_point_attempt_rate`
- `free_throw_rate`
- `relative_ts`
- `league_relative_scoring_per_75`

The final field is prototype-only and calculated relative to the representative sample mean. It should be replaced with a full-season league/player distribution calculation before production use.

## Derived Metric Formulas

Estimated on-court possessions:

```text
estimated_on_court_possessions = (minutes / 48) * team_pace
```

If team pace is unavailable, the prototype falls back to league pace. This is a transparent approximation, not exact play-by-play possession counting.

Per-75 counting stats:

```text
stat_per_75 = stat / estimated_on_court_possessions * 75
```

True shooting percentage:

```text
TS% = PTS / (2 * (FGA + 0.44 * FTA))
```

When Basketball-Reference advanced TS% is available, the prototype uses the source value.

Relative true shooting:

```text
relative_TS = player_TS - league_TS_for_that_season
```

League TS is derived from all player totals for that season:

```text
league_TS = sum(PTS) / (2 * (sum(FGA) + 0.44 * sum(FTA)))
```

Shot mix:

```text
three_point_attempt_rate = 3PA / FGA
free_throw_rate = FTA / FGA
```

Missing inputs produce nulls, not zeroes.

## Era Normalization Approach

The prototype preserves raw production and adds interpretable era context:

1. **Possession normalization**: counting stats per estimated 75 possessions.
2. **Efficiency relative to league**: player TS% minus season league TS%.
3. **Environment context**: team and league pace retained as explicit fields.
4. **Volume context**: current prototype includes `points_per_75`; production should add full-season percentile/z-score context only after validating distributions.

This does not pretend eras are identical. It gives users both what happened historically and how that production sat inside its own environment.

## Historical Availability Matrix

| Era | Basic Box | Minutes | TOV | STL/BLK | ORB/DRB | 3PT | Pace | Usage | Advanced | Play-by-play | Tracking |
|---|---|---|---|---|---|---|---|---|---|---|---|
| pre-1976 / early NBA | partial/broad | often available | weak/unavailable | unavailable before 1973-74 | weaker | unavailable | estimated/limited | unavailable | limited | unavailable | unavailable |
| 1976-1979 | good | available | begins but inconsistent by season | available | available | unavailable until 1979-80 | available from 1973-74 | from 1977-78 | partial | unavailable | unavailable |
| 1980s | good | available | available | available | available | available | available | available | available | limited/unavailable | unavailable |
| 1990s | good | available | available | available | available | available | available | available | available | partial late-era only | unavailable |
| 2000s | good | available | available | available | available | available | available | available | available | better availability | unavailable/limited |
| 2010s | strong | available | available | available | available | available | available | available | strong | available from modern sources | partial/modern |
| 2020s | strong | available | available | available | available | available | available | available | strong | available | available for some sources |

Fields that create unfair comparisons if treated as universal:

- 3-point makes/attempts before 1979-80.
- Usage percentage before 1977-78.
- Turnovers and turnover-derived metrics in older seasons.
- Steals/blocks before 1973-74.
- Tracking/defender/player movement data before the modern tracking era.

## Missing Data Rules

The simulator must distinguish:

- `null`: unavailable, not tracked, not applicable, or not derivable.
- `0`: the event/stat was tracked and the player/team recorded zero.
- `estimated`: derived from a documented formula.
- `source`: directly supplied by a source table.

Examples:

- Kareem 1976-77 has null 3PA because the NBA had no 3-point line.
- Kareem 1976-77 has null turnover-derived fields where the source does not provide the input.
- Shaq 1999-00 has 0 made threes and 1 attempted three because the stat existed and was tracked.

## Sanity Check Results

Observed prototype outputs are directionally sensible:

- Curry 2015-16: very high `relative_ts` and extreme `three_point_attempt_rate`.
- Jordan 1990-91: elite `points_per_75` with strong positive `relative_ts`.
- Shaq 1999-00: elite scoring volume, strong rebounding rate, near-zero 3PAr, high FTr.
- LeBron 2012-13: high efficiency, high usage, high assist percentage.
- Jokic 2022-23: extremely high TS and assist percentage for a center profile.
- Kareem 1976-77: strong scoring/efficiency/rebounding without being penalized for unavailable 3PT/usage inputs.

These are sanity checks, not ratings.

## Validation

Commands run:

```powershell
python scripts\historical\build_player_season_prototype.py
python scripts\historical\validate_player_season_prototype.py
```

Validation result:

```json
{
  "ok": true,
  "player_rows": 7,
  "team_rows": 5
}
```

Validation checks:

- All representative player-seasons present.
- All representative team-seasons present.
- No duplicate player-season rows.
- No duplicate team-season rows.
- Required loaded fields are non-null.
- TS% is between 0 and 1.
- `points_per_75` is positive.
- Manifest reports no validation issues.

## Caveats

- Basketball-Reference HTML is not a formal API. The pipeline uses cached raw HTML and column-signature selection to reduce brittleness, but this remains a source risk.
- Player identity resolution needs a real crosswalk before production. The prototype has no stable universal `player_id`.
- Team abbreviation/name mapping must be expanded and audited for historical franchise changes.
- `estimated_on_court_possessions` uses season team pace, not exact lineup/player possession counts.
- Older seasons should expose nulls rather than imputed modern values.
- Full league-relative percentiles/z-scores require complete season-level player distributions, not just representative rows.

## V1 Recommendation

Minimum supported HoopsLab V1 historical season: **1979-80**.

Reason:

- 3-point line exists.
- Pace/team context is available.
- Turnovers and usage are meaningfully available from the late 1970s onward.
- This avoids a large class of structural missingness while still covering modern and classic eras.

Research mode can support **1976-77 onward** with explicit caveats and null-aware UI/model behavior.

## Next Step

Build a full historical ingestion pipeline for all seasons from 1979-80 through current completed season:

1. Fetch/cache all required Basketball-Reference season pages.
2. Build complete `Player`, `PlayerSeason`, `Team`, and `TeamSeason` tables.
3. Add player identity resolution and franchise/team mapping.
4. Compute league-wide era-relative percentiles/z-scores.
5. Add tests for historical missingness and representative legends.

