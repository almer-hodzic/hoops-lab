# Simulation Engine Design Spike

## Verdict

**MODEL FOUNDATION STILL NEEDS WORK**

The current historical data is strong enough to build a transparent V1 team-strength profile for offense and pace. It is not strong enough to present defense or game-level win probability as definitive without additional game-level and stronger defensive data. The recommended next build should expose uncertainty clearly and keep the simulator labeled as an experimental historical translation.

## Reconstruction Dataset

- Team-season rows: 1314
- Seasons: 1979-80 through 2025-26
- Team roster source: `player_seasons.csv` plus `player_team_splits.csv`
- Team target source: `team_seasons.csv`
- League context source: `league_seasons.csv`

Rows use actual historical player minutes for validation. Traded players use team-specific split rows for the teams they actually played for.

## Feature Set

Features are minutes-weighted, not equal-player averages:

- Per-75 production: points, assists, rebounds, turnovers, steals, blocks, offensive rebounds, defensive rebounds
- Efficiency and style: relative TS, TS%, 3PAr, FTr
- Creation/ball control: usage, AST%, TOV%, top-three usage pressure
- Rebounding and defense proxies: TRB%, defensive rebounds, steals, blocks, PER-based box proxy
- Rotation structure: top 1/3/5 minute share, minutes HHI, players above 500/1000 minutes
- Era context: league estimated ORtg, league pace, league TS%

No 0-100 ratings, chemistry, or arbitrary overalls were created.

## Split

- Train: 1979-80 through 2013-14
- Validation: 2014-15 through 2018-19
- Test: 2019-20 through 2025-26

This is temporal validation to reduce season leakage.

## Model Results

| Target | Model | Val MAE | Val RMSE | Val R2 | Test MAE | Test RMSE | Test R2 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ortg | formula | 3.17 | 3.70 | -0.005 | 3.95 | 4.45 | -0.365 |
| ortg | linear | 0.07 | 0.08 | 0.999 | 0.11 | 0.13 | 0.999 |
| ortg | ridge | 0.07 | 0.08 | 0.999 | 0.11 | 0.13 | 0.999 |
| ortg | hist_gradient_boosting | 0.12 | 0.25 | 0.995 | 1.28 | 2.22 | 0.661 |
| drtg | formula | 2.17 | 2.63 | 0.311 | 2.27 | 2.82 | 0.300 |
| drtg | linear | 3.49 | 3.86 | -0.481 | 4.30 | 4.67 | -0.911 |
| drtg | ridge | 3.44 | 3.82 | -0.447 | 4.22 | 4.59 | -0.849 |
| drtg | hist_gradient_boosting | 2.59 | 3.02 | 0.096 | 6.22 | 6.64 | -2.867 |
| pace | formula | 1.63 | 2.00 | 0.504 | 1.56 | 1.89 | 0.094 |
| pace | linear | 1.63 | 2.06 | 0.473 | 2.08 | 2.56 | -0.651 |
| pace | ridge | 1.62 | 2.06 | 0.478 | 2.05 | 2.52 | -0.604 |
| pace | hist_gradient_boosting | 1.75 | 2.23 | 0.387 | 2.32 | 2.81 | -0.996 |

Recommended selected approaches:

- ORtg reconstruction: `ridge`
- DRtg prototype baseline: `formula`
- Pace prototype baseline: `formula`

Ridge is acceptable for historical ORtg reconstruction because it is stable on validation/test and remains interpretable. DRtg and Pace use formula baselines for the prototype recommendation because the learned models either underperform the temporal test split or extrapolate poorly for custom all-time rosters.

## Leakage And Deployability Audit

No production application code is changed by this spike. The audit found no target team outcome columns in the model feature matrix: `actual_ortg`, `actual_drtg`, `actual_pace`, `actual_net_rtg`, wins, and losses are stored only as labels/context outputs, not predictors.

The suspicious ORtg Ridge result is still not deployable evidence. The strongest issue is `w_points_per_75`: it is computed from realized historical player scoring and actual historical minutes, so it can almost directly reconstruct team points per possession. `w_per` is also treated as reconstruction-only because it is a broad realized box-score composite. Both are available for selected PlayerSeason records, but they should not drive the production custom-team projection model without role/usage recalibration.

Team-level leakage check: the Ridge feature matrix does not use actual team TS%, actual team TOV%, actual team ORB%, actual team FTr, actual offensive possessions, actual team points, actual team ORtg, net rating, or wins/losses. The `team_3par_from_players`, `team_ftr_from_players`, and `team_ast_to_tov_from_players` columns are rebuilt from player rows; for custom rosters the script scales player counting inputs to projected minutes before recomputing those ratios.

| Feature | Source Type | Category | Synthetic Available | Can Encode ORtg |
| --- | --- | --- | ---: | ---: |
| league_ortg_estimate | league | SAFE FOR CUSTOM TEAM PROJECTION | True | False |
| league_pace | league | SAFE FOR CUSTOM TEAM PROJECTION | True | False |
| league_ts_pct | league | SAFE FOR CUSTOM TEAM PROJECTION | True | False |
| w_points_per_75 | player | RECONSTRUCTION-ONLY | True | True |
| w_assists_per_75 | player | SAFE FOR CUSTOM TEAM PROJECTION | True | False |
| w_rebounds_per_75 | player | SAFE FOR CUSTOM TEAM PROJECTION | True | False |
| w_turnovers_per_75 | player | SAFE FOR CUSTOM TEAM PROJECTION | True | False |
| w_steals_per_75 | player | SAFE FOR CUSTOM TEAM PROJECTION | True | False |
| w_blocks_per_75 | player | SAFE FOR CUSTOM TEAM PROJECTION | True | False |
| w_offensive_rebounds_per_75 | player | SAFE FOR CUSTOM TEAM PROJECTION | True | False |
| w_defensive_rebounds_per_75 | player | SAFE FOR CUSTOM TEAM PROJECTION | True | False |
| w_relative_ts | player | SAFE FOR CUSTOM TEAM PROJECTION | True | False |
| w_true_shooting_pct | player | SAFE FOR CUSTOM TEAM PROJECTION | True | False |
| w_usage_pct | player | SAFE FOR CUSTOM TEAM PROJECTION | True | False |
| w_assist_pct | player | SAFE FOR CUSTOM TEAM PROJECTION | True | False |
| w_turnover_pct | player | SAFE FOR CUSTOM TEAM PROJECTION | True | False |
| w_total_rebound_pct | player | SAFE FOR CUSTOM TEAM PROJECTION | True | False |
| w_three_point_attempt_rate | player | SAFE FOR CUSTOM TEAM PROJECTION | True | False |
| w_free_throw_rate | player | SAFE FOR CUSTOM TEAM PROJECTION | True | False |
| w_age | player | SAFE FOR CUSTOM TEAM PROJECTION | True | False |
| w_per | player | RECONSTRUCTION-ONLY | True | True |
| team_3par_from_players | player | SAFE FOR CUSTOM TEAM PROJECTION | True | False |
| team_ftr_from_players | player | SAFE FOR CUSTOM TEAM PROJECTION | True | False |
| team_ast_to_tov_from_players | player | SAFE FOR CUSTOM TEAM PROJECTION | True | False |
| minutes_top1_share | rotation | SAFE FOR CUSTOM TEAM PROJECTION | True | False |
| minutes_top3_share | rotation | SAFE FOR CUSTOM TEAM PROJECTION | True | False |
| minutes_top5_share | rotation | SAFE FOR CUSTOM TEAM PROJECTION | True | False |
| minutes_hhi | rotation | SAFE FOR CUSTOM TEAM PROJECTION | True | False |
| rotation_players_500 | rotation | SAFE FOR CUSTOM TEAM PROJECTION | True | False |
| rotation_players_1000 | rotation | SAFE FOR CUSTOM TEAM PROJECTION | True | False |
| weighted_usage_top3 | player+rotation | SAFE FOR CUSTOM TEAM PROJECTION | True | False |
| usage_pressure | player+rotation | SAFE FOR CUSTOM TEAM PROJECTION | True | False |
| spacing_proxy | player | SAFE FOR CUSTOM TEAM PROJECTION | True | False |
| rebounding_proxy | player | SAFE FOR CUSTOM TEAM PROJECTION | True | False |
| defense_box_proxy | player | SAFE FOR CUSTOM TEAM PROJECTION | True | False |

Leakage-safe ORtg retrain excludes `w_points_per_75` and `w_per` from learned features, and the safe formula excludes direct scoring volume.

| Safe ORtg Model | Val MAE | Val RMSE | Val R2 | Test MAE | Test RMSE | Test R2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| transparent_formula_safe | 4.00 | 4.40 | -0.418 | 4.91 | 5.29 | -0.930 |
| linear_safe | 0.38 | 0.49 | 0.982 | 0.48 | 0.61 | 0.975 |
| ridge_safe | 0.37 | 0.49 | 0.982 | 0.47 | 0.60 | 0.975 |
| hist_gradient_boosting_safe | 1.00 | 1.34 | 0.868 | 3.08 | 3.93 | -0.065 |

This is materially worse than the original `0.11` test MAE and should be reported as the more honest deployability check. The remaining strong Ridge result still benefits from historical player-season role context, so it is useful for reconstruction but not sufficient proof for arbitrary synthetic rosters.

Custom-roster availability proof: the proposed production path uses only canonical PlayerSeason fields, explicit roster slots, explicit target-era league context, and the rotation algorithm below. Any feature that cannot be computed from those inputs is excluded from the production projection path.

## Defense Audit

DRtg inputs do not include actual team DRtg, opponent scoring, net rating, wins/losses, or other realized defensive outcome columns. The genuinely derivable defensive evidence is limited to player defensive rebounds, steals, blocks, total rebound rate, age, minutes/role, and the current `defense_box_proxy`. That is not enough for confident individual defensive projection, so the formula baseline remains the safest V1 approach and should carry wide uncertainty.

## Pace Audit

Projected pace does not copy the custom roster's actual team pace because no such team exists. The custom profile uses target-era `league_pace`, roster concentration, and player stat proxies. Historical validation still uses actual historical player minutes, so learned pace reconstruction remains descriptive; the deployable V1 path should keep pace anchored to target era plus transparent role/style adjustments.

## Coefficient / Feature Sanity

```json
{
  "actual_ortg": {
    "top_coefficients": [
      {
        "feature": "w_points_per_75",
        "coefficient": 3.534553228412425
      },
      {
        "feature": "w_turnovers_per_75",
        "coefficient": -0.12417177610204355
      },
      {
        "feature": "w_assists_per_75",
        "coefficient": 0.09359556114314582
      },
      {
        "feature": "w_per",
        "coefficient": 0.09358382581214642
      },
      {
        "feature": "w_usage_pct",
        "coefficient": -0.0899979157598636
      },
      {
        "feature": "team_3par_from_players",
        "coefficient": 0.08324934961283441
      },
      {
        "feature": "league_pace",
        "coefficient": 0.05870120325040529
      },
      {
        "feature": "w_assist_pct",
        "coefficient": -0.058223310674373696
      },
      {
        "feature": "w_relative_ts",
        "coefficient": 0.05751370002263112
      },
      {
        "feature": "w_offensive_rebounds_per_75",
        "coefficient": 0.05455806948009467
      },
      {
        "feature": "w_true_shooting_pct",
        "coefficient": 0.05431535980939195
      },
      {
        "feature": "league_ortg_estimate",
        "coefficient": 0.04903192117103172
      }
    ],
    "flags": [
      "Ridge assigns positive offensive coefficient to turnover percentage."
    ]
  },
  "actual_drtg": {
    "top_coefficients": [
      {
        "feature": "league_ortg_estimate",
        "coefficient": 1.7921123244702688
      },
      {
        "feature": "w_defensive_rebounds_per_75",
        "coefficient": -1.4809324600204263
      },
      {
        "feature": "team_3par_from_players",
        "coefficient": 1.4206461079449921
      },
      {
        "feature": "league_pace",
        "coefficient": -1.237828313657541
      },
      {
        "feature": "defense_box_proxy",
        "coefficient": -1.230605587851187
      },
      {
        "feature": "w_offensive_rebounds_per_75",
        "coefficient": 1.1729790266057307
      },
      {
        "feature": "w_steals_per_75",
        "coefficient": -1.1601085956862107
      },
      {
        "feature": "minutes_hhi",
        "coefficient": -1.0345488841913322
      },
      {
        "feature": "w_three_point_attempt_rate",
        "coefficient": -0.8532878097100363
      },
      {
        "feature": "w_turnovers_per_75",
        "coefficient": -0.8398931449195675
      },
      {
        "feature": "league_ts_pct",
        "coefficient": -0.8130391967041515
      },
      {
        "feature": "w_per",
        "coefficient": 0.7366920166770683
      }
    ],
    "flags": []
  },
  "actual_pace": {
    "top_coefficients": [
      {
        "feature": "league_pace",
        "coefficient": 3.6728989355970634
      },
      {
        "feature": "w_per",
        "coefficient": 2.411243686285246
      },
      {
        "feature": "w_points_per_75",
        "coefficient": -2.25469466671694
      },
      {
        "feature": "w_usage_pct",
        "coefficient": 1.2842303943005713
      },
      {
        "feature": "league_ortg_estimate",
        "coefficient": 1.2620713033464526
      },
      {
        "feature": "team_3par_from_players",
        "coefficient": 1.042755063419368
      },
      {
        "feature": "w_assists_per_75",
        "coefficient": 0.9004154023768368
      },
      {
        "feature": "league_ts_pct",
        "coefficient": -0.7369762080593185
      },
      {
        "feature": "minutes_top3_share",
        "coefficient": 0.6678748476309283
      },
      {
        "feature": "w_age",
        "coefficient": -0.6585044601574666
      },
      {
        "feature": "w_assist_pct",
        "coefficient": -0.6430872805137638
      },
      {
        "feature": "minutes_top5_share",
        "coefficient": -0.5640487573003738
      }
    ],
    "flags": []
  }
}
```

Main interpretation:

- Offensive estimates are meaningfully informed by league environment, shooting efficiency, creation, turnover pressure, and spacing.
- Pace estimates are fairly stable because league pace provides a strong era anchor.
- Defensive estimates are materially weaker. Box-score defensive proxies have limited signal and should be treated as uncertainty-heavy.

## Famous-Team Sanity Checks

| Season | Team | Actual ORtg | Pred ORtg | Actual DRtg | Pred DRtg | Actual Pace | Pred Pace |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1985-86 | Boston Celtics | 111.8 | 111.8 | 102.6 | 107.8 | 101.2 | 102.2 |
| 1995-96 | Chicago Bulls | 115.2 | 115.1 | 101.8 | 108.2 | 91.1 | 92.2 |
| 2000-01 | Los Angeles Lakers | 108.4 | 108.5 | 104.8 | 103.7 | 91.7 | 91.3 |
| 2012-13 | Miami Heat | 112.3 | 112.3 | 103.7 | 106.3 | 90.7 | 92.3 |
| 2016-17 | Golden State Warriors | 115.6 | 115.6 | 104.0 | 108.9 | 99.8 | 96.9 |

These examples are sanity checks only. Models were trained and selected on broad temporal splits, not tuned to famous teams.

## Rotation Recommendation

Historical validation uses actual minutes.

For custom V1 rosters, use a transparent default:

1. Start from each selected PlayerSeason's historical MPG.
2. Assign role-aware floors and soft caps: starters start near 28-40 minutes, bench players near 8-24.
3. Assign hard caps: five-player-only rosters may reach 48 per player; rosters with bench depth cap starters near 42 and bench near 30.
4. If initial minutes exceed 240, scale down proportionally.
5. If initial minutes are below 240, distribute remaining minutes by historical MPG weight until hard caps are reached.
6. Round to one decimal and apply any rounding drift to the player with remaining headroom.
7. The final rotation must equal exactly 240.0 minutes; otherwise the roster is invalid.

This avoids equal-player weighting while keeping assumptions visible.

## Usage Normalization

The spike uses `weighted_usage_top3` and `usage_pressure` as transparent diminishing-return signals. A future roster with Jordan, LeBron, and Curry cannot assume each keeps full historical usage. V1 should normalize expected usage so total lineup/team usage is coherent, then apply a modest penalty when too much usage is concentrated among high-usage creators. This is not chemistry; it is a usage accounting constraint.

## Custom All-Time Roster Prototype

Target environment: 2016-17 neutral NBA environment

| Slot | Player | Season | Historical Team | Historical MPG | Projected Minutes |
| --- | --- | --- | --- | ---: | ---: |
| PG | Stephen Curry | 2015-16 | GSW | 34.2 | 48.0 |
| SG | Michael Jordan | 1990-91 | CHI | 37.0 | 48.0 |
| SF | LeBron James | 2012-13 | MIA | 37.9 | 48.0 |
| PF | Tim Duncan | 2002-03 | SAS | 39.3 | 48.0 |
| C | Shaquille O'Neal | 1999-00 | LAL | 40.0 | 48.0 |

Intermediate profile:

```json
{
  "w_points_per_75": 28.8571,
  "w_relative_ts": 0.0807,
  "w_usage_pct": 30.98,
  "weighted_usage_top3": 19.14,
  "usage_pressure": 1.14,
  "team_3par_from_players": 0.1726,
  "team_ftr_from_players": 0.3857,
  "spacing_proxy": 0.0139,
  "rebounding_proxy": 10.4195,
  "defense_box_proxy": 5.9574
}
```

Projected profile:

```json
{
  "projected_ortg": 118.38,
  "projected_drtg": 109.53,
  "projected_pace": 96.4
}
```

Total projected minutes: 240.0

Raw selected-model extrapolation, shown for audit only:

```json
{
  "projected_ortg": 169.16,
  "projected_drtg": 109.53,
  "projected_pace": 96.4
}
```

Unassigned bench minutes: 0.0

Caveats:

- Actual high-usage stars cannot all retain original usage; usage pressure is included as a penalty/proxy, not chemistry.
- Individual defense is weakly observed in current historical PlayerSeason data.
- Only five players were supplied; V1 allocates all 240 minutes among them instead of silently inventing replacement players.
- Learned reconstruction models can extrapolate badly on all-star custom rosters, so the transparent formula projection is shown as the safer prototype output.
- Projection is a team-strength profile, not a simulated win total.

## Cross-Era Methodology

Recommended V1 default: **user-selectable target era with neutral modern default**.

For the first simulator build, use a target league environment such as 2016-17 or a selectable season. Translate player/team profiles through era-relative features:

- Use relative TS and season-relative percentiles/z-scores for player quality signals.
- Use the target era's league ORtg, league pace, and league TS as anchors.
- Preserve style indicators like 3PAr rather than pretending a 1980 shot profile automatically becomes modern spacing.

Avoid modeling every rule/officiating change in V1.

## Game Simulation Design

Do not productionize yet, but the recommended structure is:

1. Estimate Team A offensive strength in target environment.
2. Estimate Team B defensive strength in target environment.
3. Blend expected points per 100 possessions from A offense vs B defense.
4. Repeat for B offense vs A defense.
5. Estimate game pace from both teams' pace profiles and target era.
6. Convert expected points per 100 possessions to expected score.
7. Sample game outcomes from calibrated score/efficiency variance.

The current processed historical season data does not contain game-level margins, home/away splits, or team-game pace. Therefore game variance and home-court should not be calibrated from the current dataset alone.

## Home Court

Current historical processed data is not sufficient to estimate home-court effect. Required data: historical team game logs with home/away, points for/against, possessions or pace proxy, date, and season. Likely sources include Basketball-Reference team game logs or other public game-log datasets. Codex can likely acquire Basketball-Reference game logs programmatically with the same caching discipline, subject to access reliability.

## Additional Data Needed

- Team game logs: calibrate score variance, home-court, game-to-game efficiency spread.
- Opponent-adjusted team/player context: reduce bias from teammate/team environment.
- Stronger individual defense data: DBPM/DRB/STL/BLK alone is not enough for confident defensive simulation.
- Lineup/minutes data if future versions model fit more precisely.

## Uncertainty

Biggest uncertainty sources:

- Individual defense
- Mixed-era translation
- Usage and role fit
- Bench strength/minute allocation
- Lack of lineup data
- Historical missing fields
- Rule/officiating differences
- No game-level variance calibration in current processed data

## Recommendation

**MODEL FOUNDATION STILL NEEDS WORK**

The model foundation still needs work before game-log calibration is the only blocker. The immediate fixes are to keep reconstruction-only scoring/composite features out of the production projection path, formalize usage normalization, and improve the defensive feature story. After that, add game-log data for variance/home-court and win-probability calibration.

## Counterfactual / Deployability Validation

Reproduce offline: `python scripts/simulation/validate_deployability.py`. The earlier same-season experiments remain above as reconstruction references, not deployment evidence.

Outputs: [frozen metrics, exact equation coefficients and feature audit](../../data/simulation/deployability_summary.json), [lagged inputs](../../data/simulation/lagged_roster_features.csv), [coverage](../../data/simulation/lagged_roster_coverage.csv), [predictions](../../data/simulation/lagged_roster_predictions.csv). The 23 deployable covariates are audited below; three additional explicit prediction offsets (`anchor_ortg`, `anchor_drtg`, `anchor_pace`) derive only from the prior league season: league_points_per_75/75*100 for ratings and league_pace for pace. They require no target-team outcome.

### Population and temporal boundaries

{
  "eligible_team_seasons": 1291,
  "rows_including_both_modes": 2582,
  "splits_per_mode": {
    "train": 931,
    "validation": 150,
    "test": 210
  },
  "seasons": [
    "1980-81",
    "1981-82",
    "1982-83",
    "1983-84",
    "1984-85",
    "1985-86",
    "1986-87",
    "1987-88",
    "1988-89",
    "1989-90",
    "1990-91",
    "1991-92",
    "1992-93",
    "1993-94",
    "1994-95",
    "1995-96",
    "1996-97",
    "1997-98",
    "1998-99",
    "1999-00",
    "2000-01",
    "2001-02",
    "2002-03",
    "2003-04",
    "2004-05",
    "2005-06",
    "2006-07",
    "2007-08",
    "2008-09",
    "2009-10",
    "2010-11",
    "2011-12",
    "2012-13",
    "2013-14",
    "2014-15",
    "2015-16",
    "2016-17",
    "2017-18",
    "2018-19",
    "2019-20",
    "2020-21",
    "2021-22",
    "2022-23",
    "2023-24",
    "2024-25",
    "2025-26"
  ],
  "coverage_threshold": 0.8
}

Train targets: 1980-81 through 2013-14; validation: 2014-15 through 2018-19; test: 2019-20 through 2025-26. First target lacks a cached prior league anchor and is excluded. All players use profiles strictly before the target year. Historical future trades/injuries and actual season-long membership are known in hindsight; this is a roster-conditional experiment, not a preseason roster forecast.

### Coverage and missing players

Compare immediate previous season against latest available profile within three years. Select the latter structurally for returning players, without choosing based on final-test error. Returner MPG decays by 1/(1+0.15*(lag-1)). Unknown entrants use fixed training-only pooled profile medians, 8 MPG preference and an explicit unknown flag, never their target-year role. Known-profile missing fields use training-only prior-position medians. These are population priors, not fabricated individual observations. Primary eligibility requires >=80% known projected minutes in deployable mode; actual-minute coverage is diagnostic only. Coverage is optimistic for unknown stars because the rotation gives unknown entrants low preference. No international/rookie skill evidence is available.

| max_profile_lag | candidate_teams | eligible_teams | mean_projected_coverage | mean_actual_coverage | projected_coverage_p10 | actual_coverage_p10 | teams_actual_coverage_below_80 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1292 | 1291 | 0.996 | 0.888 | 1.000 | 0.757 | 235 |
| 3 | 1292 | 1291 | 0.997 | 0.900 | 1.000 | 0.774 | 183 |

### Shared rotation and usage builder

Oracle rotation is NOT DEPLOYABLE: target actual minutes normalized to 240. All performance remains lagged. Deployable rotation ranks prior MPG only: first five floor24/cap40, next five floor6/cap26, others zero. Allocate the remaining minutes proportionally to prior MPG (minimum preference1), iterating at caps. For 6-player capacity shortage caps become48; five players necessarily play48 each. No target-season starts, usage, MPG, efficiency or team pace enter features. No anonymous players. These are ranking roles, not verified starter labels, and no positional lineup constraints are modeled.

With m_i projected minutes, e_i=m_i/48, u_i=prior USG%, q_i=clip(u_i,1,60)^alpha, normalized usage v_i=100*q_i/sum(e_j*q_j); possession share s_i=e_i*v_i/100. Thus sum(s_i)=1 and sum(e_i*v_i)=100. Compression=sum((m_i/240)*max(u_i-v_i,0)); concentration=sum(s_i^2). Alpha is selected among 0.5/0.75/1 using training teams only to minimize projected-minute-weighted error against their realized, normalized usage (calibration targets only). This calibrates role matching, not a causal efficiency response. Fit coefficients on compression and other effects on training teams only. Normalization conserves usage and retains larger roles for high-usage players at equal minutes; an efficiency penalty is not assumed unless its training coefficient supports one. No chemistry score. Training calibration: `{"selected_exponent": 1.0, "training_scores": {"0.5": {"weighted_usage_mae": 2.507943451270574, "teams": 931}, "0.75": {"weighted_usage_mae": 2.050626178461826, "teams": 931}, "1.0": {"weighted_usage_mae": 1.9298649552993505, "teams": 931}}}`

### Production feature audit

All inputs below derive from prior/selected PlayerSeason profiles, roster membership and proposed minutes. No PER, target scoring-volume aggregate, target-team outcome or PER-based composite is used. Prior per75 rates inherit prior team-pace estimation/context; they are not context-free player talent. League anchors use the preceding season, which avoids requiring future league outcomes. Reconstruction and deployable feature sets are separately exported in JSON.

| feature | source | target_performance_required | synthetic_available |
| --- | --- | --- | --- |
| w_relative_ts | sum(minutes/240 * prior.relative_ts) over non-null source values | False | True |
| w_assist_pct | sum(minutes/240 * prior.assist_pct) over non-null source values | False | True |
| w_turnover_pct | sum(minutes/240 * prior.turnover_pct) over non-null source values | False | True |
| w_usage_pct | sum(minutes/240 * prior.usage_pct) over non-null source values | False | True |
| w_three_point_attempt_rate | sum(minutes/240 * prior.three_point_attempt_rate) over non-null source values | False | True |
| w_free_throw_rate | sum(minutes/240 * prior.free_throw_rate) over non-null source values | False | True |
| w_offensive_rebounds_per_75 | sum(minutes/240 * prior.offensive_rebounds_per_75) over non-null source values | False | True |
| w_defensive_rebounds_per_75 | sum(minutes/240 * prior.defensive_rebounds_per_75) over non-null source values | False | True |
| w_steals_per_75 | sum(minutes/240 * prior.steals_per_75) over non-null source values | False | True |
| w_blocks_per_75 | sum(minutes/240 * prior.blocks_per_75) over non-null source values | False | True |
| w_defensive_box_plus_minus | sum(minutes/240 * prior.defensive_box_plus_minus) over non-null source values | False | True |
| w_steal_pct | sum(minutes/240 * prior.steal_pct) over non-null source values | False | True |
| w_block_pct | sum(minutes/240 * prior.block_pct) over non-null source values | False | True |
| w_defensive_rebound_pct | sum(minutes/240 * prior.defensive_rebound_pct) over non-null source values | False | True |
| usage_weighted_relative_ts | sum(normalized possession share * prior.relative_ts) | False | True |
| usage_weighted_3par | sum(normalized possession share * prior.three_point_attempt_rate) | False | True |
| usage_weighted_ftr | sum(normalized possession share * prior.free_throw_rate) | False | True |
| usage_compression | sum(minutes/240 * max(prior.USG - normalized USG,0)) | False | True |
| usage_concentration | sum(normalized possession share squared) | False | True |
| rotation_hhi | sum((proposed minutes/240) squared) | False | True |
| prior_pace_delta | sum(minutes/240 * (prior.team_pace - prior.league_pace)) | False | True |
| known_prior_minutes_share | sum(proposed minutes/240 for sourced prior profiles) | False | True |
| returner_minutes_share | sum(proposed minutes/240 for prior lag >1) | False | True |
| prior_age_years | sum(minutes/240 * sourced prior profile lag in years); unknown=0 | False | True |

### Model comparisons

Predict residuals relative to prior league ORtg/pace. The transparent formula is a small additive OLS equation using the explicitly listed fit terms, not arbitrary hand coefficients. Full linear and fixed Ridge(alpha30) compare against the league anchor. One small HGB offense candidate checks nonlinear usage/spacing interactions; defense and pace remain additive. Imputation/scaling fit on training only. Select minimum validation MAE separately per target/mode. No test tuning or refitting. Oracle and deployable are distinct experimental fits; only the latter serves the custom lineup.

| mode | target | model | split | mae | rmse | r2 |
| --- | --- | --- | --- | --- | --- | --- |
| oracle | ortg | league_anchor | validation | 2.775 | 3.506 | 0.098 |
| oracle | ortg | league_anchor | test | 2.789 | 3.536 | 0.138 |
| oracle | ortg | transparent_formula | validation | 1.857 | 2.507 | 0.539 |
| oracle | ortg | transparent_formula | test | 2.040 | 2.546 | 0.553 |
| oracle | ortg | linear | validation | 2.109 | 2.681 | 0.472 |
| oracle | ortg | linear | test | 2.418 | 2.977 | 0.389 |
| oracle | ortg | ridge | validation | 2.073 | 2.648 | 0.485 |
| oracle | ortg | ridge | test | 2.358 | 2.904 | 0.418 |
| oracle | ortg | nonlinear | validation | 1.812 | 2.298 | 0.612 |
| oracle | ortg | nonlinear | test | 2.208 | 2.680 | 0.505 |
| oracle | drtg | league_anchor | validation | 2.410 | 2.954 | 0.133 |
| oracle | drtg | league_anchor | test | 2.451 | 3.088 | 0.163 |
| oracle | drtg | transparent_formula | validation | 2.335 | 2.772 | 0.237 |
| oracle | drtg | transparent_formula | test | 2.424 | 2.906 | 0.259 |
| oracle | drtg | linear | validation | 2.206 | 2.674 | 0.289 |
| oracle | drtg | linear | test | 2.322 | 2.814 | 0.305 |
| oracle | drtg | ridge | validation | 2.165 | 2.621 | 0.318 |
| oracle | drtg | ridge | test | 2.259 | 2.736 | 0.343 |
| oracle | pace | league_anchor | validation | 1.981 | 2.541 | 0.203 |
| oracle | pace | league_anchor | test | 1.694 | 2.034 | -0.044 |
| oracle | pace | transparent_formula | validation | 1.498 | 1.975 | 0.518 |
| oracle | pace | transparent_formula | test | 1.684 | 2.062 | -0.073 |
| oracle | pace | linear | validation | 1.498 | 1.939 | 0.536 |
| oracle | pace | linear | test | 1.810 | 2.186 | -0.206 |
| oracle | pace | ridge | validation | 1.483 | 1.935 | 0.537 |
| oracle | pace | ridge | test | 1.702 | 2.076 | -0.087 |
| deployable | ortg | league_anchor | validation | 2.775 | 3.506 | 0.098 |
| deployable | ortg | league_anchor | test | 2.789 | 3.536 | 0.138 |
| deployable | ortg | transparent_formula | validation | 2.200 | 2.849 | 0.404 |
| deployable | ortg | transparent_formula | test | 2.450 | 3.075 | 0.348 |
| deployable | ortg | linear | validation | 2.352 | 2.994 | 0.342 |
| deployable | ortg | linear | test | 2.723 | 3.393 | 0.206 |
| deployable | ortg | ridge | validation | 2.302 | 2.948 | 0.362 |
| deployable | ortg | ridge | test | 2.562 | 3.217 | 0.286 |
| deployable | ortg | nonlinear | validation | 2.328 | 2.909 | 0.379 |
| deployable | ortg | nonlinear | test | 2.506 | 3.139 | 0.321 |
| deployable | drtg | league_anchor | validation | 2.410 | 2.954 | 0.133 |
| deployable | drtg | league_anchor | test | 2.451 | 3.088 | 0.163 |
| deployable | drtg | transparent_formula | validation | 2.275 | 2.744 | 0.252 |
| deployable | drtg | transparent_formula | test | 2.392 | 2.875 | 0.274 |
| deployable | drtg | linear | validation | 2.350 | 2.859 | 0.188 |
| deployable | drtg | linear | test | 2.854 | 3.448 | -0.044 |
| deployable | drtg | ridge | validation | 2.307 | 2.807 | 0.217 |
| deployable | drtg | ridge | test | 2.713 | 3.277 | 0.057 |
| deployable | pace | league_anchor | validation | 1.981 | 2.541 | 0.203 |
| deployable | pace | league_anchor | test | 1.694 | 2.034 | -0.044 |
| deployable | pace | transparent_formula | validation | 1.555 | 2.068 | 0.472 |
| deployable | pace | transparent_formula | test | 1.582 | 1.957 | 0.033 |
| deployable | pace | linear | validation | 1.535 | 1.981 | 0.515 |
| deployable | pace | linear | test | 1.700 | 2.098 | -0.111 |
| deployable | pace | ridge | validation | 1.519 | 1.969 | 0.521 |
| deployable | pace | ridge | test | 1.688 | 2.084 | -0.096 |

Selected on validation: `{"oracle": {"ortg": "nonlinear", "drtg": "ridge", "pace": "ridge"}, "deployable": {"ortg": "transparent_formula", "drtg": "transparent_formula", "pace": "ridge"}}`

The fitted deployable offensive compression coefficient is `0.12535994118408056`. It is positive: normalization conserves usage and reduces individual star roles, but these historical data do NOT establish an efficiency penalty for star crowding. Do not interpret this association as a causal fit bonus or claim validated efficiency diminishing returns. This is a specific acceptance blocker requiring targeted role/transfer evidence.

### Defensive source provenance

Offline ingestion extension `scripts/historical/export_cached_defensive_profiles.py` reuses the canonical cache and numeric parser with row-level source IDs (names are not unique), exporting cached Basketball-Reference `NBA_YYYY_advanced.html` DBPM/DWS/STL%/BLK%/DRB% fields to a supplemental analysis file. Source URLs: https://www.basketball-reference.com/leagues/NBA_YYYY_advanced.html. No downloads, canonical schema or production application changes. DBPM and DWS are source-derived estimates, not observed individual defensive impact; percentage fields are source-computed box rates. DWS is audited but excluded from predictors because cumulative wins/playing time are context-heavy. Absent source entries remain null in the sidecar; model imputation is explicit. Older eras are supported only where the cached source records the metric. Availability by era is in JSON.

Canonical profiles without a matching row-level source ID/team or with inconsistent source minutes are quarantined from prior histories; canonical datasets are not rewritten. This detects an inherited Charles Jones same-name identity collision and other source-join inconsistencies. It does not repair all canonical identity issues, and target membership can still inherit them. Quarantined rows: `[{"player_id": "br:johnsge03", "player_name": "George Johnson", "season": "1980-81", "team": "SAS", "minutes": 1935.0, "source_minutes": null}, {"player_id": "br:johnsed02", "player_name": "Eddie Johnson", "season": "1982-83", "team": "KCK", "minutes": 2933.0, "source_minutes": null}, {"player_id": "br:johnsge02", "player_name": "George Johnson", "season": "1982-83", "team": "IND", "minutes": 2297.0, "source_minutes": null}, {"player_id": "br:johnsed02", "player_name": "Eddie Johnson", "season": "1984-85", "team": "KCK", "minutes": 3029.0, "source_minutes": null}, {"player_id": "br:jonesch02", "player_name": "Charles Jones", "season": "1984-85", "team": "2TM", "minutes": 667.0, "source_minutes": null}, {"player_id": "br:johnsed03", "player_name": "Eddie Johnson", "season": "1985-86", "team": "2TM", "minutes": 1477.0, "source_minutes": null}, {"player_id": "br:johnsge03", "player_name": "George Johnson", "season": "1985-86", "team": "SEA", "minutes": 264.0, "source_minutes": null}, {"player_id": "br:johnsed02", "player_name": "Eddie Johnson", "season": "1986-87", "team": "SAC", "minutes": 2457.0, "source_minutes": null}, {"player_id": "br:jonesch02", "player_name": "Charles Jones", "season": "1987-88", "team": "WSB", "minutes": 1313.0, "source_minutes": null}, {"player_id": "br:jonesch02", "player_name": "Charles Jones", "season": "1988-89", "team": "WSB", "minutes": 1154.0, "source_minutes": 516.0}, {"player_id": "br:smithch02", "player_name": "Charles Smith", "season": "1989-90", "team": "LAC", "minutes": 2732.0, "source_minutes": null}, {"player_id": "br:smithch02", "player_name": "Charles Smith", "season": "1990-91", "team": "LAC", "minutes": 2703.0, "source_minutes": null}, {"player_id": "br:smithmi01", "player_name": "Michael Smith", "season": "1994-95", "team": "SAC", "minutes": 1736.0, "source_minutes": null}, {"player_id": "br:willima04", "player_name": "Marcus Williams", "season": "2008-09", "team": "GSW", "minutes": 54.0, "source_minutes": null}, {"player_id": "br:johnsch04", "player_name": "Chris Johnson", "season": "2012-13", "team": "MIN", "minutes": 284.0, "source_minutes": null}, {"player_id": "br:mitchto03", "player_name": "Tony Mitchell", "season": "2013-14", "team": "DET", "minutes": 79.0, "source_minutes": null}]`

### Era, quality and error distribution

#### oracle

| era | split | n | target | mae | rmse | r2 |
| --- | --- | --- | --- | --- | --- | --- |
| (1979, 1989] | train/in-sample | 208 | ortg | 1.351 | 1.738 | 0.756 |
| (1979, 1989] | train/in-sample | 208 | drtg | 1.926 | 2.451 | 0.344 |
| (1979, 1989] | train/in-sample | 208 | pace | 1.915 | 2.422 | 0.324 |
| (1989, 1999] | train/in-sample | 278 | ortg | 1.557 | 1.983 | 0.752 |
| (1989, 1999] | train/in-sample | 278 | drtg | 2.052 | 2.659 | 0.458 |
| (1989, 1999] | train/in-sample | 278 | pace | 1.642 | 2.170 | 0.744 |
| (1999, 2009] | train/in-sample | 295 | ortg | 1.549 | 1.931 | 0.728 |
| (1999, 2009] | train/in-sample | 295 | drtg | 2.157 | 2.679 | 0.457 |
| (1999, 2009] | train/in-sample | 295 | pace | 1.762 | 2.380 | 0.123 |
| (2009, 2014] | train/in-sample | 150 | ortg | 1.459 | 1.775 | 0.733 |
| (2009, 2014] | train/in-sample | 150 | drtg | 2.126 | 2.647 | 0.367 |
| (2009, 2014] | train/in-sample | 150 | pace | 1.426 | 1.883 | 0.375 |
| (2014, 2019] | validation | 150 | ortg | 1.812 | 2.298 | 0.612 |
| (2014, 2019] | validation | 150 | drtg | 2.165 | 2.621 | 0.318 |
| (2014, 2019] | validation | 150 | pace | 1.483 | 1.935 | 0.537 |
| (2019, 2022] | test | 90 | ortg | 2.031 | 2.460 | 0.469 |
| (2019, 2022] | test | 90 | drtg | 2.068 | 2.478 | 0.212 |
| (2019, 2022] | test | 90 | pace | 2.024 | 2.397 | -0.306 |
| (2022, 2026] | test | 120 | ortg | 2.340 | 2.833 | 0.308 |
| (2022, 2026] | test | 120 | drtg | 2.402 | 2.915 | 0.063 |
| (2022, 2026] | test | 120 | pace | 1.460 | 1.797 | 0.104 |

Test quality tiers use actual net rating only for diagnosis: poor<-3, average[-3,3], elite>3.

| tier | n | target | mae | rmse | r2 |
| --- | --- | --- | --- | --- | --- |
| poor | 58 | ortg | 2.528 | 2.904 | 0.032 |
| poor | 58 | drtg | 2.406 | 2.836 | 0.002 |
| poor | 58 | pace | 1.558 | 1.928 | -0.125 |
| average | 97 | ortg | 1.782 | 2.220 | 0.229 |
| average | 97 | drtg | 2.108 | 2.621 | -0.187 |
| average | 97 | pace | 1.741 | 2.154 | -0.200 |
| elite | 55 | ortg | 2.621 | 3.136 | -0.354 |
| elite | 55 | drtg | 2.369 | 2.827 | 0.039 |
| elite | 55 | pace | 1.786 | 2.084 | -0.010 |

| target | 0.1 | 0.5 | 0.9 | 0.95 | 1.0 |
| --- | --- | --- | --- | --- | --- |
| ortg | 0.466 | 1.989 | 4.196 | 5.277 | 7.481 |
| drtg | 0.360 | 2.110 | 4.585 | 5.093 | 6.569 |
| pace | 0.295 | 1.519 | 3.378 | 3.752 | 5.480 |

Defensive stability against the anchor (no post-test reselection):

| model | era | mae | rmse | r2 |
| --- | --- | --- | --- | --- |
| league_anchor | test_2020_2022 | 2.273 | 2.834 | -0.031 |
| league_anchor | test_2023_2026 | 2.584 | 3.266 | -0.176 |
| league_anchor | validation_2015_2019 | 2.410 | 2.954 | 0.133 |
| transparent_formula | test_2020_2022 | 2.207 | 2.610 | 0.126 |
| transparent_formula | test_2023_2026 | 2.587 | 3.110 | -0.066 |
| transparent_formula | validation_2015_2019 | 2.335 | 2.772 | 0.237 |
| linear | test_2020_2022 | 2.134 | 2.545 | 0.169 |
| linear | test_2023_2026 | 2.464 | 3.000 | 0.008 |
| linear | validation_2015_2019 | 2.206 | 2.674 | 0.289 |
| ridge | test_2020_2022 | 2.068 | 2.478 | 0.212 |
| ridge | test_2023_2026 | 2.402 | 2.915 | 0.063 |
| ridge | validation_2015_2019 | 2.165 | 2.621 | 0.318 |

#### deployable

| era | split | n | target | mae | rmse | r2 |
| --- | --- | --- | --- | --- | --- | --- |
| (1979, 1989] | train/in-sample | 208 | ortg | 2.390 | 2.955 | 0.295 |
| (1979, 1989] | train/in-sample | 208 | drtg | 2.103 | 2.608 | 0.257 |
| (1979, 1989] | train/in-sample | 208 | pace | 1.948 | 2.448 | 0.310 |
| (1989, 1999] | train/in-sample | 278 | ortg | 2.311 | 2.919 | 0.464 |
| (1989, 1999] | train/in-sample | 278 | drtg | 2.297 | 2.955 | 0.331 |
| (1989, 1999] | train/in-sample | 278 | pace | 1.693 | 2.261 | 0.722 |
| (1999, 2009] | train/in-sample | 295 | ortg | 2.468 | 3.116 | 0.292 |
| (1999, 2009] | train/in-sample | 295 | drtg | 2.272 | 2.866 | 0.378 |
| (1999, 2009] | train/in-sample | 295 | pace | 1.821 | 2.408 | 0.102 |
| (2009, 2014] | train/in-sample | 150 | ortg | 2.394 | 3.053 | 0.210 |
| (2009, 2014] | train/in-sample | 150 | drtg | 2.273 | 2.778 | 0.303 |
| (2009, 2014] | train/in-sample | 150 | pace | 1.450 | 1.913 | 0.355 |
| (2014, 2019] | validation | 150 | ortg | 2.200 | 2.849 | 0.404 |
| (2014, 2019] | validation | 150 | drtg | 2.275 | 2.744 | 0.252 |
| (2014, 2019] | validation | 150 | pace | 1.519 | 1.969 | 0.521 |
| (2019, 2022] | test | 90 | ortg | 2.268 | 2.806 | 0.309 |
| (2019, 2022] | test | 90 | drtg | 2.166 | 2.627 | 0.114 |
| (2019, 2022] | test | 90 | pace | 2.000 | 2.397 | -0.306 |
| (2022, 2026] | test | 120 | ortg | 2.587 | 3.261 | 0.083 |
| (2022, 2026] | test | 120 | drtg | 2.560 | 3.048 | -0.024 |
| (2022, 2026] | test | 120 | pace | 1.455 | 1.815 | 0.086 |

Test quality tiers use actual net rating only for diagnosis: poor<-3, average[-3,3], elite>3.

| tier | n | target | mae | rmse | r2 |
| --- | --- | --- | --- | --- | --- |
| poor | 58 | ortg | 2.722 | 3.314 | -0.261 |
| poor | 58 | drtg | 3.069 | 3.617 | -0.623 |
| poor | 58 | pace | 1.559 | 1.965 | -0.167 |
| average | 97 | ortg | 2.177 | 2.797 | -0.224 |
| average | 97 | drtg | 2.091 | 2.533 | -0.109 |
| average | 97 | pace | 1.704 | 2.083 | -0.122 |
| elite | 55 | ortg | 2.645 | 3.274 | -0.476 |
| elite | 55 | drtg | 2.207 | 2.539 | 0.224 |
| elite | 55 | pace | 1.798 | 2.206 | -0.132 |

| target | 0.1 | 0.5 | 0.9 | 0.95 | 1.0 |
| --- | --- | --- | --- | --- | --- |
| ortg | 0.456 | 2.053 | 5.213 | 6.234 | 8.478 |
| drtg | 0.365 | 2.370 | 4.546 | 5.487 | 7.663 |
| pace | 0.252 | 1.439 | 3.378 | 4.033 | 5.485 |

Defensive stability against the anchor (no post-test reselection):

| model | era | mae | rmse | r2 |
| --- | --- | --- | --- | --- |
| league_anchor | test_2020_2022 | 2.273 | 2.834 | -0.031 |
| league_anchor | test_2023_2026 | 2.584 | 3.266 | -0.176 |
| league_anchor | validation_2015_2019 | 2.410 | 2.954 | 0.133 |
| transparent_formula | test_2020_2022 | 2.166 | 2.627 | 0.114 |
| transparent_formula | test_2023_2026 | 2.560 | 3.048 | -0.024 |
| transparent_formula | validation_2015_2019 | 2.275 | 2.744 | 0.252 |
| linear | test_2020_2022 | 2.587 | 3.127 | -0.255 |
| linear | test_2023_2026 | 3.054 | 3.671 | -0.485 |
| linear | validation_2015_2019 | 2.350 | 2.859 | 0.188 |
| ridge | test_2020_2022 | 2.479 | 2.980 | -0.139 |
| ridge | test_2023_2026 | 2.888 | 3.483 | -0.338 |
| ridge | validation_2015_2019 | 2.307 | 2.807 | 0.217 |

### Major roster changes

| season | season_end_year | team_name | mode | known_prior_minutes_share | actual_ortg | actual_drtg | actual_pace | predicted_ortg | predicted_drtg | predicted_pace |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2007-08 | 2008 | Boston Celtics | oracle | 0.948 | 110.200 | 98.900 | 90.900 | 109.361 | 103.910 | 92.438 |
| 2010-11 | 2011 | Miami Heat | oracle | 0.999 | 111.700 | 103.500 | 90.900 | 111.734 | 102.624 | 91.308 |
| 2016-17 | 2017 | Golden State Warriors | oracle | 0.941 | 115.600 | 104.000 | 99.800 | 111.425 | 103.221 | 98.641 |
| 2019-20 | 2020 | Los Angeles Lakers | oracle | 0.994 | 112.000 | 106.300 | 100.900 | 114.756 | 107.821 | 103.564 |
| 2007-08 | 2008 | Boston Celtics | deployable | 1.000 | 110.200 | 98.900 | 90.900 | 107.025 | 104.982 | 92.897 |
| 2010-11 | 2011 | Miami Heat | deployable | 1.000 | 111.700 | 103.500 | 90.900 | 112.338 | 105.021 | 91.698 |
| 2016-17 | 2017 | Golden State Warriors | deployable | 1.000 | 115.600 | 104.000 | 99.800 | 112.707 | 102.680 | 97.867 |
| 2019-20 | 2020 | Los Angeles Lakers | deployable | 1.000 | 112.000 | 106.300 | 100.900 | 111.490 | 108.777 | 103.010 |

These misses combine roster/injury availability, role changes and coaching effects absent from player box profiles. Boston's new defensive system, Miami's role adjustment, Durant's arrival and the Lakers' defensive personnel are stress cases, not cases used to tune coefficients. Train-period examples are in-sample sanity checks, not held-out evidence. Specific residuals and coverage appear in the table.

### Five-player semantics and all-time lineup

Recommend minimum eight named players for a normal V1 full-game roster. Permit five-player lineups only as explicitly labeled ironman hypotheticals with high, uncalibrated uncertainty; 48 minutes each is unavoidable. No fatigue/bench/replacement simulation is implied.

| player | season | minutes | historical_usage_pct | normalized_usage_pct | team_possession_share_pct | relative_ts | assist_pct | turnover_pct | usage_pct | three_point_attempt_rate | free_throw_rate | offensive_rebounds_per_75 | defensive_rebounds_per_75 | steals_per_75 | blocks_per_75 | defensive_box_plus_minus | steal_pct | block_pct | defensive_rebound_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Stephen Curry | 2015-16 | 48.000 | 32.600 | 21.046 | 21.046 | 0.128 | 33.700 | 12.900 | 32.600 | 0.554 | 0.250 | 0.913 | 4.861 | 2.269 | 0.201 | 1.600 | 3.000 | 0.400 | 13.600 |
| Michael Jordan | 1990-91 | 48.000 | 32.900 | 21.240 | 21.240 | 0.071 | 25.200 | 8.700 | 32.900 | 0.051 | 0.365 | 1.465 | 4.642 | 2.768 | 1.030 | 3.200 | 3.700 | 1.700 | 14.300 |
| LeBron James | 2012-13 | 48.000 | 30.200 | 19.496 | 19.496 | 0.105 | 36.400 | 12.400 | 30.200 | 0.188 | 0.395 | 1.338 | 7.077 | 1.780 | 0.924 | 2.400 | 2.400 | 1.900 | 20.800 |
| Tim Duncan | 2002-03 | 48.000 | 28.000 | 18.076 | 18.076 | 0.045 | 19.500 | 12.900 | 28.000 | 0.016 | 0.455 | 3.257 | 9.859 | 0.692 | 2.980 | 2.500 | 0.900 | 5.200 | 27.300 |
| Shaquille O'Neal | 1999-00 | 48.000 | 31.200 | 20.142 | 20.142 | 0.055 | 19.300 | 9.900 | 31.200 | 0.001 | 0.495 | 4.099 | 9.052 | 0.439 | 2.916 | 2.000 | 0.600 | 5.300 | 24.800 |

Target-era assumptions: 2016-17 preseason, 2015-16 league anchors; no game logs, fatigue or bench model

Shared aggregate inputs: `{"w_relative_ts": 0.08071694658270726, "w_assist_pct": 26.82, "w_turnover_pct": 11.360000000000001, "w_usage_pct": 30.980000000000004, "w_three_point_attempt_rate": 0.16181331837191504, "w_free_throw_rate": 0.39221251422318043, "w_offensive_rebounds_per_75": 2.214305410813706, "w_defensive_rebounds_per_75": 7.098033251001157, "w_steals_per_75": 1.589494678276071, "w_blocks_per_75": 1.6103301901598672, "w_defensive_box_plus_minus": 2.3400000000000003, "w_steal_pct": 2.1200000000000006, "w_block_pct": 2.9000000000000004, "w_defensive_rebound_pct": 20.160000000000004, "usage_weighted_relative_ts": 0.08161744265482868, "usage_weighted_3par": 0.16699157614480717, "usage_weighted_ftr": 0.38930891886330393, "usage_compression": 10.98, "usage_concentration": 0.2006604981826714, "rotation_hhi": 0.20000000000000004, "prior_pace_delta": -0.1573850163890767, "known_prior_minutes_share": 1.0, "returner_minutes_share": 0.0, "prior_age_years": 1.0, "anchor_ortg": 107.20165285182661, "anchor_drtg": 107.20165285182661, "anchor_pace": 95.77419354838707}`

Projections (diagnostic, not accepted production ratings): `{"ortg": 121.16117001789252, "drtg": 97.75440657649936, "pace": 96.76497266841987}`

Validation ordinary-team uncertainty reference: `{"ortg": {"validation_abs_error_p90": 4.556722899384564, "reference_interval": [116.60444711850795, 125.71789291727708]}, "drtg": {"validation_abs_error_p90": 4.332467927292531, "reference_interval": [93.42193864920684, 102.08687450379189]}, "pace": {"validation_abs_error_p90": 3.311451616155041, "reference_interval": [93.45352105226483, 100.07642428457491]}}`

Out-of-training-range features: `{"w_relative_ts": {"value": 0.08071694658270726, "min": -0.05696158253832797, "max": 0.05301263433019937}, "w_assist_pct": {"value": 26.82, "min": 8.22820478204788, "max": 22.194652673595886}, "w_usage_pct": {"value": 30.980000000000004, "min": 15.119824646915042, "max": 24.626230103992192}, "w_defensive_rebounds_per_75": {"value": 7.098033251001157, "min": 3.216545545748502, "max": 6.432660172098833}, "w_defensive_box_plus_minus": {"value": 2.3400000000000003, "min": -1.495591533600204, "max": 1.5061422451366184}, "w_defensive_rebound_pct": {"value": 20.160000000000004, "min": 10.09726210110326, "max": 19.459467259050285}, "usage_weighted_relative_ts": {"value": 0.08161744265482868, "min": -0.05661050528347352, "max": 0.058107770890613406}, "usage_compression": {"value": 10.98, "min": 0.0, "max": 4.626230103992194}, "usage_concentration": {"value": 0.2006604981826714, "min": 0.11615244991309345, "max": 0.1722496872704493}, "rotation_hhi": {"value": 0.20000000000000004, "min": 0.11618904269661952, "max": 0.1329968909255105}}`

### Acceptance and next step

**B. NEEDS TARGETED IMPROVEMENT**. Lagged validation removes same-season performance dependence and tests the whole eligible league, but it cannot establish causal transport to cross-era star lineups. Defense lacks matchup/lineup coverage; usage shrinkage/efficiency response and role/minute allocation remain hypotheses. Ordinary-team errors are not synthetic-roster calibration. Era-level defensive stability must be assessed against the league-anchor baseline (also exported per era); a learned model is only an experimental candidate until it improves consistently. No readiness claim follows from a high R-squared.

Next: validate rotation against held-out actual minutes diagnostically, conduct held-out transfer/roster-turnover and normalized-usage sensitivity tests, and establish explicit defensive acceptance tolerances using validation only. Keep final-test results frozen. Do not acquire game logs or start Monte Carlo calibration yet.

## Rotation Validation

Focused improvement protocol: `python scripts/simulation/improve_projection.py --validate`, then tests, then `--final` once. `--report` only renders stored results. Training targets end 2013-14; validation is 2014-15 through 2018-19. Final 2019-20 through 2025-26 retains all 210 original team keys. The frozen protocol fingerprints code, canonical data, prior membership, model bundle and validation results; an exclusive ledger prevents a second final evaluation. No test outcomes inform selection/tolerances.

Detailed artifacts: `data/simulation/targeted_projection_validation.json`, `targeted_projection_protocol.json`, validation rotation/transfer/features CSVs, and final JSON/predictions CSV. Frozen fitted models are cached under ignored `artifacts/simulation/`.

MPG requires two denominators: actual per-appearance minutes/games does not sum to 240 because injuries, partial seasons, trades and player absences overlap. Primary rotation error uses team-equivalent MPG = 240*player season minutes/sum(team player minutes), the allocation the model must conserve. Per-appearance errors are separately reported; they are not interchangeable. Target starts label starter/bench diagnostics only; they never enter rotation inputs.

| model | equivalent_mae | equivalent_rmse | appearance_mae | appearance_rmse | starter_mae | bench_mae | starter_appearance_mae | bench_appearance_mae | top5_share_error | top8_share_error | hhi_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| old | 8.770 | 11.667 | 10.170 | 12.466 | 11.394 | 7.489 | 9.261 | 10.613 | 0.171 | 0.134 | 0.033 |
| balanced_prior | 6.213 | 8.236 | 7.813 | 9.796 | 7.408 | 5.630 | 9.448 | 7.014 | 0.052 | 0.057 | 0.009 |
| ridge_minutes | 6.178 | 7.753 | 7.873 | 9.498 | 7.379 | 5.591 | 10.735 | 6.477 | 0.092 | 0.104 | 0.017 |

Selected rotation: `balanced_prior`.

The balanced prior uses 0.6*last MPG + 0.4*three-year MPG (history weights1,0.5,0.25), multiplied by 0.6+0.4*prior availability. The learned candidate is a fixed Ridge(alpha100) predicting team-equivalent minutes from prior MPG/history, games availability, projected age, prior starts fraction, prior rank, roster depth, known-profile and absence flags. Training outcomes are labels only. No selected rotation uses target MPG/starts. Select lower validation equivalent-MPG MAE, preferring the transparent balanced rule within0.10 MPG. This validation-only practical-equivalence choice avoids selecting a nine-feature minutes regression for a tiny MAE gain while its top-five share/concentration errors are worse. retain old rotation if neither candidate improves. Historical caps38 for prior-ranked first five, 28 for others; floors4/0.5 preserve depth. A scalar shift with clipping projects onto exactly240 minutes. Unlike the old hard24 starter floor, it permits prior stars to lose roles.

Unknown entrants receive a training-only mixture minutes prior, not fabricated NBA skill observations. When legitimate Team Builder slots are supplied, training-calibrated starter/primary/deep-bench minutes priors are available. Historical new-entrant roles are unknown; target starts/rank are never used to assign them. Statistical profile medians remain explicitly uncertain population imputations.

Training entrant priors: `{"roles": {"deep_bench": {"n": 1873, "equivalent_mpg": 3.7740286994721277, "appearance_mpg": 9.397720151606396}, "primary_bench": {"n": 372, "equivalent_mpg": 16.77555177002215, "appearance_mpg": 19.620338380114525}, "starter": {"n": 321, "equivalent_mpg": 27.624919664703363, "appearance_mpg": 29.392787851301645}}, "unknown_role_mixture_mpg": 8.642579978538315, "n": 2566}`

| model | group | n | mae | rmse | bias |
| --- | --- | --- | --- | --- | --- |
| old | absence_return | 65 | 4.527 | 7.295 | -1.313 |
| old | no_prior | 517 | 6.725 | 10.365 | -6.725 |
| old | returning | 1358 | 9.466 | 11.869 | 0.734 |
| old | transferred | 932 | 9.187 | 12.285 | 2.752 |
| balanced_prior | absence_return | 65 | 4.827 | 6.546 | 1.448 |
| balanced_prior | no_prior | 517 | 5.850 | 8.912 | -4.094 |
| balanced_prior | returning | 1358 | 6.093 | 7.861 | -0.791 |
| balanced_prior | transferred | 932 | 6.685 | 8.481 | 3.322 |

Per-appearance MPG errors by the same prior-history categories:

| model | group | n | mae | rmse | bias |
| --- | --- | --- | --- | --- | --- |
| old | absence_return | 65 | 10.065 | 12.237 | -8.776 |
| old | no_prior | 517 | 13.353 | 15.528 | -13.353 |
| old | returning | 1358 | 8.892 | 11.152 | -4.933 |
| old | transferred | 932 | 10.272 | 12.389 | -5.947 |
| balanced_prior | absence_return | 65 | 7.648 | 9.941 | -6.015 |
| balanced_prior | no_prior | 517 | 10.899 | 13.397 | -10.722 |
| balanced_prior | returning | 1358 | 7.218 | 8.683 | -6.458 |
| balanced_prior | transferred | 932 | 6.978 | 8.909 | -5.377 |

Prior games-played coverage (fraction of prior league schedule):

| model | availability | denominator | n | mae | rmse | bias |
| --- | --- | --- | --- | --- | --- | --- |
| old | 50-80% | equivalent | 645 | 9.423 | 12.062 | -0.235 |
| old | 50-80% | appearance | 645 | 10.395 | 12.527 | -7.261 |
| old | <50% | equivalent | 525 | 6.753 | 9.952 | -1.594 |
| old | <50% | appearance | 525 | 11.199 | 13.285 | -9.040 |
| old | >=80% | equivalent | 1185 | 10.200 | 12.648 | 3.768 |
| old | >=80% | appearance | 1185 | 8.202 | 10.385 | -2.855 |
| old | no_prior | equivalent | 517 | 6.725 | 10.365 | -6.725 |
| old | no_prior | appearance | 517 | 13.353 | 15.528 | -13.353 |
| balanced_prior | 50-80% | equivalent | 645 | 6.706 | 8.279 | 0.189 |
| balanced_prior | 50-80% | appearance | 645 | 7.834 | 9.447 | -6.837 |
| balanced_prior | <50% | equivalent | 525 | 5.047 | 7.037 | -1.257 |
| balanced_prior | <50% | appearance | 525 | 9.255 | 11.261 | -8.703 |
| balanced_prior | >=80% | equivalent | 1185 | 6.619 | 8.396 | 2.240 |
| balanced_prior | >=80% | appearance | 1185 | 5.815 | 7.035 | -4.383 |
| balanced_prior | no_prior | equivalent | 517 | 5.850 | 8.912 | -4.094 |
| balanced_prior | no_prior | appearance | 517 | 10.899 | 13.397 | -10.722 |

Largest errors after selection:

| season | team | player | category | starter | starter_observed | prior_games_fraction | actual_mpg | actual_equivalent_mpg | projected_mpg | error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2014-15 | Minnesota Timberwolves | Andrew Wiggins | no_prior | True | True | 0.500 | 36.207 | 35.977 | 1.829 | 34.148 |
| 2017-18 | Philadelphia 76ers | Ben Simmons | no_prior | True | True | 0.500 | 33.728 | 33.147 | 0.500 | 32.647 |
| 2014-15 | Indiana Pacers | Paul George | returning | False | True | 0.976 | 15.167 | 1.100 | 33.437 | 32.337 |
| 2018-19 | Cleveland Cavaliers | Collin Sexton | no_prior | True | True | 0.500 | 31.768 | 31.651 | 0.500 | 31.151 |
| 2017-18 | Boston Celtics | Gordon Hayward | transferred | True | True | 0.890 | 5.000 | 0.061 | 29.112 | 29.051 |
| 2017-18 | Utah Jazz | Donovan Mitchell | no_prior | True | True | 0.500 | 33.392 | 32.045 | 3.015 | 29.030 |
| 2015-16 | Indiana Pacers | Ty Lawson | transferred | False | True | 0.915 | 18.077 | 2.837 | 31.562 | 28.725 |
| 2018-19 | Phoenix Suns | Mikal Bridges | no_prior | True | True | 0.500 | 29.476 | 29.175 | 0.500 | 28.675 |
| 2017-18 | Los Angeles Lakers | Kyle Kuzma | no_prior | False | True | 0.500 | 31.182 | 28.948 | 0.500 | 28.448 |
| 2015-16 | Minnesota Timberwolves | Karl-Anthony Towns | no_prior | True | True | 0.500 | 32.037 | 31.711 | 3.975 | 27.736 |
| 2018-19 | Dallas Mavericks | Luka Dončić | no_prior | True | True | 0.500 | 32.194 | 28.124 | 0.500 | 27.624 |
| 2017-18 | San Antonio Spurs | Kawhi Leonard | returning | True | True | 0.902 | 23.333 | 2.554 | 29.706 | 27.152 |

Structural limits: actual season-long rosters contain more players than an active game roster; membership is known in hindsight, but signing/trade dates and injury availability are not modeled. A single240-minute allocation cannot reproduce per-appearance roles or predict unknown rookie talent. Prior teammate continuity and prior team membership classify returning/transfer cases; multi-team prior profiles do not identify exact offseason transfer dates.

Product rule: normal full-game mode requires8-12 named players, exactly240 minutes; provided starter slots receive floors20/caps38 and bench floors4/caps28. Five-player mode is explicit ironman only,48 each, high uncertainty. Six/seven-player full-game rosters are rejected. Historical research can retain larger season-long rosters; no anonymous players are invented. Team Builder UI is unchanged.

## Player Transfer Stability

Correlations and mean absolute/change statistics below use validation transfers only, with verified source identities and sourced values. Star transfers: prior USG>=27 and MPG>=28; ordinary transfers: MPG>=10 excluding stars. Large role changes (diagnostic labels only): absolute USG change>=5 or appearance MPG change>=8. Subsets overlap and player/team rows are not independent. Absence returns are identified separately in rotation diagnostics.

Stability samples include low-minute stints: correlations for all transfers are noisy, not estimates of intrinsic skill persistence. Inspect the separately reported star/ordinary subsets and per-player raw attempts/minutes before assigning strong causal weights. Changes in TS and 3PAr use fractions; usage/AST/TOV and defensive percentages use percentage points.

| subset | field | n | correlation | mean_change | mae_change |
| --- | --- | --- | --- | --- | --- |
| all_transfers | usage_pct | 997 | 0.539 | -0.541 | 3.469 |
| all_transfers | true_shooting_pct | 991 | 0.211 | -0.006 | 0.078 |
| all_transfers | relative_ts | 991 | 0.205 | -0.010 | 0.078 |
| all_transfers | assist_pct | 997 | 0.720 | -0.376 | 4.622 |
| all_transfers | turnover_pct | 991 | 0.424 | 0.007 | 4.213 |
| all_transfers | three_point_attempt_rate | 991 | 0.801 | 0.032 | 0.094 |
| all_transfers | minutes_per_game | 997 | 0.595 | -1.838 | 5.918 |
| all_transfers | defensive_box_plus_minus | 997 | 0.217 | -0.095 | 1.247 |
| all_transfers | defensive_win_shares | 997 | 0.530 | -0.359 | 0.669 |
| all_transfers | steal_pct | 997 | 0.217 | -0.018 | 0.671 |
| all_transfers | block_pct | 997 | 0.669 | 0.009 | 0.739 |
| all_transfers | defensive_rebound_pct | 997 | 0.579 | -0.022 | 3.823 |
| all_transfers | steals_per_75 | 997 | 0.219 | -0.013 | 0.502 |
| all_transfers | blocks_per_75 | 997 | 0.670 | -0.015 | 0.340 |
| all_transfers | defensive_rebounds_per_75 | 997 | 0.570 | -0.025 | 1.297 |
| all_transfers | age | 997 | 0.996 | 1.086 | 1.086 |
| all_transfers | defensive_environment | 997 | 0.080 | -0.005 | 2.591 |
| star_transfers | usage_pct | 26 | 0.647 | -3.258 | 3.488 |
| star_transfers | true_shooting_pct | 26 | 0.538 | -0.023 | 0.038 |
| star_transfers | relative_ts | 26 | 0.585 | -0.027 | 0.037 |
| star_transfers | assist_pct | 26 | 0.753 | -3.015 | 5.154 |
| star_transfers | turnover_pct | 26 | 0.516 | 0.104 | 2.427 |
| star_transfers | three_point_attempt_rate | 26 | 0.798 | 0.033 | 0.073 |
| star_transfers | minutes_per_game | 26 | 0.402 | -4.275 | 4.471 |
| star_transfers | defensive_box_plus_minus | 26 | 0.662 | 0.108 | 0.908 |
| star_transfers | defensive_win_shares | 26 | 0.401 | -0.438 | 1.154 |
| star_transfers | steal_pct | 26 | 0.756 | -0.065 | 0.365 |
| star_transfers | block_pct | 26 | 0.923 | 0.342 | 0.504 |
| star_transfers | defensive_rebound_pct | 26 | 0.921 | 0.223 | 2.438 |
| star_transfers | steals_per_75 | 26 | 0.767 | -0.042 | 0.257 |
| star_transfers | blocks_per_75 | 26 | 0.932 | 0.129 | 0.204 |
| star_transfers | defensive_rebounds_per_75 | 26 | 0.926 | 0.081 | 0.769 |
| star_transfers | age | 26 | 1.000 | 1.000 | 1.000 |
| star_transfers | defensive_environment | 26 | 0.099 | -0.999 | 2.658 |
| ordinary_transfers | usage_pct | 829 | 0.558 | -0.407 | 3.221 |
| ordinary_transfers | true_shooting_pct | 824 | 0.242 | -0.009 | 0.067 |
| ordinary_transfers | relative_ts | 824 | 0.235 | -0.013 | 0.068 |
| ordinary_transfers | assist_pct | 829 | 0.761 | -0.410 | 4.246 |
| ordinary_transfers | turnover_pct | 824 | 0.523 | -0.110 | 3.618 |
| ordinary_transfers | three_point_attempt_rate | 824 | 0.830 | 0.031 | 0.088 |
| ordinary_transfers | minutes_per_game | 829 | 0.577 | -2.932 | 5.801 |
| ordinary_transfers | defensive_box_plus_minus | 829 | 0.284 | -0.080 | 1.103 |
| ordinary_transfers | defensive_win_shares | 829 | 0.473 | -0.435 | 0.719 |
| ordinary_transfers | steal_pct | 829 | 0.289 | 0.002 | 0.565 |
| ordinary_transfers | block_pct | 829 | 0.707 | -0.023 | 0.651 |
| ordinary_transfers | defensive_rebound_pct | 829 | 0.691 | -0.012 | 3.334 |
| ordinary_transfers | steals_per_75 | 829 | 0.290 | 0.002 | 0.422 |
| ordinary_transfers | blocks_per_75 | 829 | 0.711 | -0.026 | 0.299 |
| ordinary_transfers | defensive_rebounds_per_75 | 829 | 0.696 | -0.014 | 1.131 |
| ordinary_transfers | age | 829 | 0.997 | 1.060 | 1.060 |
| ordinary_transfers | defensive_environment | 829 | 0.050 | -0.056 | 2.602 |
| large_role_changes | usage_pct | 414 | 0.315 | -0.790 | 5.394 |
| large_role_changes | true_shooting_pct | 408 | 0.133 | -0.014 | 0.110 |
| large_role_changes | relative_ts | 408 | 0.128 | -0.017 | 0.110 |
| large_role_changes | assist_pct | 414 | 0.593 | -1.119 | 5.941 |
| large_role_changes | turnover_pct | 408 | 0.329 | -0.441 | 5.320 |
| large_role_changes | three_point_attempt_rate | 408 | 0.678 | 0.042 | 0.122 |
| large_role_changes | minutes_per_game | 414 | 0.291 | -3.354 | 9.562 |
| large_role_changes | defensive_box_plus_minus | 414 | 0.112 | -0.163 | 1.715 |
| large_role_changes | defensive_win_shares | 414 | 0.426 | -0.372 | 0.692 |
| large_role_changes | steal_pct | 414 | 0.110 | -0.017 | 0.916 |
| large_role_changes | block_pct | 414 | 0.590 | 0.012 | 0.926 |
| large_role_changes | defensive_rebound_pct | 414 | 0.429 | -0.014 | 5.283 |
| large_role_changes | steals_per_75 | 414 | 0.113 | -0.014 | 0.687 |
| large_role_changes | blocks_per_75 | 414 | 0.587 | -0.010 | 0.426 |
| large_role_changes | defensive_rebounds_per_75 | 414 | 0.418 | -0.023 | 1.785 |
| large_role_changes | age | 414 | 0.994 | 1.133 | 1.133 |
| large_role_changes | defensive_environment | 414 | 0.031 | -0.184 | 2.629 |
| transfers_500_minutes_both_seasons | usage_pct | 471 | 0.724 | -0.539 | 2.718 |
| transfers_500_minutes_both_seasons | true_shooting_pct | 471 | 0.428 | 0.002 | 0.038 |
| transfers_500_minutes_both_seasons | relative_ts | 471 | 0.406 | -0.001 | 0.037 |
| transfers_500_minutes_both_seasons | assist_pct | 471 | 0.839 | -0.214 | 3.700 |
| transfers_500_minutes_both_seasons | turnover_pct | 471 | 0.751 | -0.048 | 2.224 |
| transfers_500_minutes_both_seasons | three_point_attempt_rate | 471 | 0.894 | 0.025 | 0.071 |
| transfers_500_minutes_both_seasons | minutes_per_game | 471 | 0.655 | -0.805 | 4.297 |
| transfers_500_minutes_both_seasons | defensive_box_plus_minus | 471 | 0.600 | 0.033 | 0.740 |
| transfers_500_minutes_both_seasons | defensive_win_shares | 471 | 0.459 | -0.295 | 0.733 |
| transfers_500_minutes_both_seasons | steal_pct | 471 | 0.740 | -0.002 | 0.323 |
| transfers_500_minutes_both_seasons | block_pct | 471 | 0.874 | 0.042 | 0.461 |
| transfers_500_minutes_both_seasons | defensive_rebound_pct | 471 | 0.876 | 0.123 | 2.344 |
| transfers_500_minutes_both_seasons | steals_per_75 | 471 | 0.744 | -0.004 | 0.242 |
| transfers_500_minutes_both_seasons | blocks_per_75 | 471 | 0.873 | 0.001 | 0.215 |
| transfers_500_minutes_both_seasons | defensive_rebounds_per_75 | 471 | 0.872 | 0.027 | 0.812 |
| transfers_500_minutes_both_seasons | age | 471 | 1.000 | 1.000 | 1.000 |
| transfers_500_minutes_both_seasons | defensive_environment | 471 | 0.049 | 0.084 | 2.696 |
| returning | usage_pct | 1358 | 0.789 | 0.082 | 2.463 |
| returning | true_shooting_pct | 1352 | 0.270 | -0.002 | 0.048 |
| returning | relative_ts | 1352 | 0.257 | -0.006 | 0.048 |
| returning | assist_pct | 1358 | 0.873 | 0.474 | 3.237 |
| returning | turnover_pct | 1352 | 0.531 | -0.278 | 2.619 |
| returning | three_point_attempt_rate | 1352 | 0.898 | 0.016 | 0.062 |
| returning | minutes_per_game | 1358 | 0.789 | -0.259 | 4.108 |
| returning | defensive_box_plus_minus | 1358 | 0.386 | -0.061 | 0.856 |
| returning | defensive_win_shares | 1358 | 0.679 | -0.082 | 0.700 |
| returning | steal_pct | 1358 | 0.605 | -0.046 | 0.418 |
| returning | block_pct | 1358 | 0.795 | -0.027 | 0.584 |
| returning | defensive_rebound_pct | 1358 | 0.794 | 0.106 | 2.650 |
| returning | steals_per_75 | 1358 | 0.605 | -0.034 | 0.314 |
| returning | blocks_per_75 | 1358 | 0.799 | -0.033 | 0.268 |
| returning | defensive_rebounds_per_75 | 1358 | 0.793 | 0.032 | 0.899 |
| returning | age | 1358 | 1.000 | 1.000 | 1.000 |
| returning | defensive_environment | 1358 | 0.520 | 0.041 | 2.028 |

Position-role stability: `{"validation_transfer_exact_position_match": 0.8495486459378134}`


Defensive transfer stability by historical era (training and validation only):

| era | field | n | correlation | mean_change | mae_change |
| --- | --- | --- | --- | --- | --- |
| (1979, 1989] | defensive_box_plus_minus | 805 | 0.367 | -0.075 | 1.113 |
| (1979, 1989] | defensive_win_shares | 805 | 0.554 | -0.329 | 0.733 |
| (1979, 1989] | steal_pct | 805 | 0.346 | -0.024 | 0.673 |
| (1979, 1989] | block_pct | 805 | 0.687 | 0.013 | 0.590 |
| (1979, 1989] | defensive_rebound_pct | 805 | 0.727 | -0.094 | 3.172 |
| (1989, 1999] | defensive_box_plus_minus | 1437 | 0.170 | -0.221 | 1.567 |
| (1989, 1999] | defensive_win_shares | 1437 | 0.547 | -0.273 | 0.693 |
| (1989, 1999] | steal_pct | 1437 | 0.304 | -0.046 | 0.766 |
| (1989, 1999] | block_pct | 1437 | 0.291 | -0.137 | 0.823 |
| (1989, 1999] | defensive_rebound_pct | 1437 | 0.601 | -0.007 | 3.622 |
| (1999, 2009] | defensive_box_plus_minus | 1733 | 0.199 | 0.031 | 1.358 |
| (1999, 2009] | defensive_win_shares | 1733 | 0.467 | -0.226 | 0.733 |
| (1999, 2009] | steal_pct | 1733 | 0.263 | -0.025 | 0.698 |
| (1999, 2009] | block_pct | 1733 | 0.674 | -0.035 | 0.721 |
| (1999, 2009] | defensive_rebound_pct | 1733 | 0.650 | 0.055 | 3.332 |
| (2009, 2014] | defensive_box_plus_minus | 983 | 0.318 | -0.089 | 1.276 |
| (2009, 2014] | defensive_win_shares | 983 | 0.485 | -0.322 | 0.708 |
| (2009, 2014] | steal_pct | 983 | 0.324 | 0.020 | 0.634 |
| (2009, 2014] | block_pct | 983 | 0.687 | -0.033 | 0.754 |
| (2009, 2014] | defensive_rebound_pct | 983 | 0.634 | 0.160 | 3.531 |
| (2014, 2019] | defensive_box_plus_minus | 997 | 0.217 | -0.095 | 1.247 |
| (2014, 2019] | defensive_win_shares | 997 | 0.530 | -0.359 | 0.669 |
| (2014, 2019] | steal_pct | 997 | 0.217 | -0.018 | 0.671 |
| (2014, 2019] | block_pct | 997 | 0.669 | 0.009 | 0.739 |
| (2014, 2019] | defensive_rebound_pct | 997 | 0.579 | -0.022 | 3.823 |

## Usage Normalization Validation

e_i=m_i/48; q_i=clip((1-shrink)*prior_USG_i+shrink*training_role_mean,1,60)^alpha; v_i=100*q_i/sum(e_j*q_j); s_i=e_i*v_i/100. Thus sum(s_i)=1. Compare alpha1, alpha0.75 and10% shrinkage to training-only rank-role means. Keep alpha1 unless validation raw-USG MAE improves by at least0.05 percentage points. Target rank/starts never determine predictor role; prior-ranked role is used. Raw-USG errors compare actual observations; distribution/concentration errors use actual USG normalized under the same projected minutes to avoid a denominator artifact. This latter reference is diagnostic, not a predictor.

| candidate | raw_usage_mae | conserved_usage_mae | top1 | top3_share | concentration |
| --- | --- | --- | --- | --- | --- |
| alpha1 | 3.104 | 3.063 | 3.296 | 0.016 | 0.003 |
| alpha075 | 3.131 | 3.089 | 4.619 | 0.027 | 0.006 |
| role_shrink10 | 3.053 | 3.013 | 3.545 | 0.018 | 0.004 |

Selected usage: `{"name": "role_shrink10", "alpha": 1.0, "shrink": 0.1}`

Star combinations (prior USG>=27, projected MPG>=20):

| cohort | n | change_usage_pct | change_true_shooting_pct | change_relative_ts | change_assist_pct | change_turnover_pct |
| --- | --- | --- | --- | --- | --- | --- |
| 0-1 high-usage players | 123 | 0.099 | -0.006 | -0.010 | 0.254 | -0.081 |
| 2+ high-usage players | 27 | -1.068 | 0.003 | -0.002 | -1.100 | -0.055 |

| season | season_end_year | team | high_usage_players | change_usage_pct | change_true_shooting_pct | change_relative_ts | change_assist_pct | change_turnover_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2007-08 | 2008 | Boston Celtics | 3 | -2.276 | 0.010 | 0.011 | -1.162 | -0.131 |
| 2010-11 | 2011 | Miami Heat | 3 | -2.541 | -0.004 | -0.003 | -5.041 | 0.929 |
| 2016-17 | 2017 | Golden State Warriors | 2 | -1.577 | 0.004 | -0.007 | -0.158 | 0.538 |

Automatically identified validation cases:

| season | season_end_year | team | high_usage_players | change_usage_pct | change_true_shooting_pct | change_relative_ts | change_assist_pct | change_turnover_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2015-16 | 2016 | Los Angeles Lakers | 2 | -2.867 | 0.006 | -0.000 | -2.192 | -0.448 |
| 2015-16 | 2016 | Miami Heat | 2 | -2.486 | null | null | 0.805 | null |
| 2015-16 | 2016 | Oklahoma City Thunder | 2 | -2.417 | 0.007 | -0.000 | -1.971 | 1.415 |
| 2017-18 | 2018 | Minnesota Timberwolves | 2 | -1.869 | 0.005 | 0.001 | -2.353 | -1.355 |
| 2017-18 | 2018 | Oklahoma City Thunder | 3 | -1.836 | 0.001 | -0.003 | -2.209 | -0.091 |
| 2016-17 | 2017 | Golden State Warriors | 2 | -1.577 | 0.004 | -0.007 | -0.158 | 0.538 |
| 2014-15 | 2015 | Cleveland Cavaliers | 3 | -1.551 | -0.005 | 0.002 | -2.181 | 0.633 |
| 2014-15 | 2015 | Los Angeles Clippers | 2 | -1.545 | -0.014 | -0.008 | -1.984 | -1.628 |
| 2017-18 | 2018 | Cleveland Cavaliers | 2 | -1.518 | -0.010 | -0.014 | -2.143 | 0.706 |
| 2017-18 | 2018 | Los Angeles Lakers | 2 | -1.496 | 0.004 | -0.001 | -0.249 | 3.485 |

Descriptive associations only; selection, role changes and coaching confound effects. No causal star-stacking efficiency penalty is imposed.

## Defensive Acceptance Tolerances

Defensive provenance and availability (source availability is not imputed availability):

| feature | provenance | context | training_transfer_correlation | availability | transfer_stability |
| --- | --- | --- | --- | --- | --- |
| defensive_box_plus_minus | Cached Basketball-Reference advanced DBPM; source-estimated box/context impact | team-dependent estimate | 0.221 | {"1980": 1.0, "1990": 1.0, "2000": 0.9997759354694152, "2010": 0.9995870328308899} | null |
| defensive_win_shares | Cached advanced DWS; cumulative, strongly team/minutes dependent; excluded | team-dependent estimate | 0.508 | {"1980": 1.0, "1990": 1.0, "2000": 1.0, "2010": 1.0} | null |
| steal_pct | Cached advanced STL%; source-computed player box rate | player attribute/rate with role and opponent context | 0.300 | {"1980": 1.0, "1990": 1.0, "2000": 0.9997759354694152, "2010": 0.9995870328308899} | null |
| block_pct | Cached advanced BLK%; source-computed player box rate | player attribute/rate with role and opponent context | 0.491 | {"1980": 1.0, "1990": 1.0, "2000": 0.9997759354694152, "2010": 0.9995870328308899} | null |
| defensive_rebound_pct | Cached advanced DRB%; source-computed player opportunity rate | player attribute/rate with role and opponent context | 0.646 | {"1980": 1.0, "1990": 1.0, "2000": 0.9997759354694152, "2010": 0.9995870328308899} | null |
| steals_per_75 | Canonical totals/advanced; per75 uses prior minutes and prior team pace | player attribute/rate with role and opponent context | 0.299 | {"1980": 1.0, "1990": 1.0, "2000": 0.9997759354694152, "2010": 0.9995870328308899} | null |
| blocks_per_75 | Canonical totals/advanced; per75 uses prior minutes and prior team pace | player attribute/rate with role and opponent context | 0.484 | {"1980": 1.0, "1990": 1.0, "2000": 0.9997759354694152, "2010": 0.9995870328308899} | null |
| defensive_rebounds_per_75 | Canonical totals/advanced; per75 uses prior minutes and prior team pace | player attribute/rate with role and opponent context | 0.647 | {"1980": 1.0, "1990": 1.0, "2000": 0.9997759354694152, "2010": 0.9995870328308899} | null |
| age | Canonical source age/position | player attribute/rate with role and opponent context | 0.995 | {"1980": 1.0, "1990": 1.0, "2000": 1.0, "2010": 1.0} | null |
| position | Canonical source age/position | player attribute/rate with role and opponent context | null | {"1980": 1.0, "1990": 1.0, "2000": 1.0, "2010": 1.0} | null |
| rotation_hhi | Proposed named-player minutes | roster structure; no realized target inputs | null | {} | Not an individual metric; evaluated through team validation, not player correlation |
| prior_defensive_environment | Prior team DRtg minus prior league anchor, weighted by prior player team minutes | intentionally context-heavy; lagged only | 0.062 | {} | null |

DBPM/DWS incorporate team/box context and are not causal individual defense. DWS is not used. Stable-rates candidate retains only rate features whose training-transfer correlation>=0.35. The compact prior-environment candidate deliberately includes only lagged team environment; none uses target DRtg/pace. Prior age/position and projected concentration are available without target performance. All models predict residual DRtg relative to prior league scoring.

Limits fixed before candidate evaluation: `{"overall_mae_max": 2.2896266864236936, "anchor_validation_mae": 2.410133354130204, "elite_poor_mae_max": 5.0, "worst_bucket_abs_bias_max": 3.0, "abs_error_p90_max": 5.5, "subera_mae_over_anchor_max": 0.15, "box_heavy_abs_bias_max": 2.0, "rationale": "Require >=5% validation MAE improvement on league anchor, <=5 extreme-bucket MAE, <=3 bucket bias, <=5.5 p90 error, <=0.15 subera regression, <=2 box-heavy bias. Practical bounds declared before candidate evaluation; no final-test values used."}`

Buckets use actual DRtg minus same-season league mean for diagnostics: elite<-3, above-average [-3,-1), average[-1,1], below-average(1,3], poor>3. Positive prediction bias means underrating defense; negative bias means overrating it. The training90th-percentile prior DBPM defines box-heavy teams. Quality buckets are never predictors.

Candidate `league_anchor` buckets:

| bucket | n | mae | bias |
| --- | --- | --- | --- |
| elite | 20 | 4.205 | 4.205 |
| above_average | 34 | 2.192 | 2.163 |
| average | 39 | 1.224 | 0.012 |
| below_average | 38 | 1.906 | -1.845 |
| poor | 19 | 4.356 | -4.356 |

Suberas: `{"2015-2016": {"mae": 2.555791446274582, "anchor_mae": 2.555791446274582}, "2017-2019": {"mae": 2.3130279593672847, "anchor_mae": 2.3130279593672847}}`; box-heavy bias: `{"n": 12, "bias": 3.8319775534773797, "threshold": 0.6466391403988502}`

Candidate `previous_formula` buckets:

| bucket | n | mae | bias |
| --- | --- | --- | --- |
| elite | 20 | 2.451 | 1.499 |
| above_average | 34 | 1.575 | 0.680 |
| average | 39 | 2.079 | -0.930 |
| below_average | 38 | 2.470 | -2.250 |
| poor | 19 | 3.523 | -3.523 |

Suberas: `{"2015-2016": {"mae": 2.24766120245675, "anchor_mae": 2.555791446274582}, "2017-2019": {"mae": 2.3284095501432795, "anchor_mae": 2.3130279593672847}}`; box-heavy bias: `{"n": 12, "bias": -0.9762004428507621, "threshold": 0.6466391403988502}`

Candidate `stable_rates` buckets:

| bucket | n | mae | bias |
| --- | --- | --- | --- |
| elite | 20 | 2.711 | 2.270 |
| above_average | 34 | 1.533 | 0.975 |
| average | 39 | 2.037 | -1.088 |
| below_average | 38 | 2.626 | -2.626 |
| poor | 19 | 4.984 | -4.984 |

Suberas: `{"2015-2016": {"mae": 2.375402749272022, "anchor_mae": 2.555791446274582}, "2017-2019": {"mae": 2.6413736459908757, "anchor_mae": 2.3130279593672847}}`; box-heavy bias: `{"n": 12, "bias": 1.4453220230927062, "threshold": 0.6466391403988502}`

Candidate `prior_environment` buckets:

| bucket | n | mae | bias |
| --- | --- | --- | --- |
| elite | 20 | 2.538 | 1.804 |
| above_average | 34 | 1.618 | 0.994 |
| average | 39 | 2.087 | -0.705 |
| below_average | 38 | 2.104 | -1.898 |
| poor | 19 | 3.242 | -3.242 |

Suberas: `{"2015-2016": {"mae": 2.2264664048761555, "anchor_mae": 2.555791446274582}, "2017-2019": {"mae": 2.1680748732208737, "anchor_mae": 2.3130279593672847}}`; box-heavy bias: `{"n": 12, "bias": -0.6284721639249575, "threshold": 0.6466391403988502}`

| candidate | accepted | checks |
| --- | --- | --- |
| league_anchor | False | {"overall": false, "extremes": true, "bucket_bias": false, "p90": true, "suberas": true, "box_heavy": false} |
| previous_formula | False | {"overall": false, "extremes": true, "bucket_bias": false, "p90": true, "suberas": true, "box_heavy": true} |
| stable_rates | False | {"overall": false, "extremes": true, "bucket_bias": false, "p90": true, "suberas": false, "box_heavy": true} |
| prior_environment | False | {"overall": true, "extremes": true, "bucket_bias": false, "p90": true, "suberas": true, "box_heavy": true} |

Validation model comparisons (smaller equation preferred within0.05 MAE):

| target | candidate | mae | rmse | r2 |
| --- | --- | --- | --- | --- |
| ortg | league_anchor | 2.775 | 3.506 | 0.098 |
| ortg | formula | 2.171 | 2.795 | 0.427 |
| ortg | ridge_compact | 2.173 | 2.790 | 0.429 |
| drtg | league_anchor | 2.410 | 2.954 | 0.133 |
| drtg | previous_formula | 2.296 | 2.757 | 0.245 |
| drtg | stable_rates | 2.535 | 3.074 | 0.061 |
| drtg | prior_environment | 2.191 | 2.653 | 0.301 |
| pace | league_anchor | 1.981 | 2.541 | 0.203 |
| pace | compact | 1.503 | 1.990 | 0.511 |
| pace | previous_ridge | 1.484 | 1.896 | 0.556 |

Selected models: `{"ortg": "formula", "drtg": "prior_environment", "pace": "compact"}`

Exact selected equation coefficients, intercepts and training imputation medians are exported under `model_equations` in the validation JSON. The frozen artifact also retains the minutes regression and all preprocessing fitted on training rows only.

## Synthetic Extrapolation Detection

Each important deployable feature is inside its training1st-99th percentile interval, near the edge inside min/max, or outside observed min/max. Distance is excess beyond min/max divided by training IQR (minimum scale max(1% range,1e-6)). Report RMS distance, largest excess and outside fraction. High uncertainty if any excess>1 IQR or >15% features outside; elevated if any outside. This is a domain-coverage indicator, never a quality score or calibrated confidence. Marginal ranges cannot establish joint lineup support; correlated features may duplicate evidence.

| feature | value | status | exceed_iqr_units | min | max | p01 | p99 | scale |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| w_relative_ts | 0.081 | outside | 1.745 | -0.047 | 0.045 | -0.037 | 0.033 | 0.021 |
| w_assist_pct | 26.820 | outside | 2.680 | 8.578 | 20.800 | 10.895 | 19.210 | 2.246 |
| w_turnover_pct | 11.360 | near_edge | 0.000 | 10.647 | 18.518 | 11.424 | 17.339 | 1.519 |
| w_usage_pct | 30.980 | outside | 7.666 | 16.726 | 23.135 | 17.913 | 22.198 | 1.023 |
| w_three_point_attempt_rate | 0.162 | inside | 0.000 | 0.010 | 0.393 | 0.014 | 0.301 | 0.136 |
| w_free_throw_rate | 0.392 | inside | 0.000 | 0.207 | 0.549 | 0.240 | 0.415 | 0.050 |
| w_offensive_rebounds_per_75 | 2.214 | inside | 0.000 | 1.182 | 2.821 | 1.424 | 2.629 | 0.352 |
| w_defensive_rebounds_per_75 | 7.098 | outside | 1.842 | 3.466 | 6.004 | 3.705 | 5.565 | 0.594 |
| w_steals_per_75 | 1.589 | inside | 0.000 | 0.905 | 1.959 | 0.980 | 1.663 | 0.188 |
| w_blocks_per_75 | 1.610 | outside | 0.976 | 0.336 | 1.414 | 0.413 | 1.212 | 0.201 |
| w_defensive_box_plus_minus | 2.340 | outside | 1.389 | -1.233 | 1.440 | -1.055 | 1.085 | 0.648 |
| w_steal_pct | 2.120 | inside | 0.000 | 1.204 | 2.623 | 1.301 | 2.215 | 0.255 |
| w_block_pct | 2.900 | outside | 1.144 | 0.554 | 2.430 | 0.768 | 2.172 | 0.411 |
| w_defensive_rebound_pct | 20.160 | outside | 2.004 | 10.809 | 17.647 | 11.606 | 16.382 | 1.254 |
| usage_weighted_relative_ts | 0.082 | outside | 1.500 | -0.046 | 0.049 | -0.036 | 0.035 | 0.021 |
| usage_weighted_3par | 0.167 | inside | 0.000 | 0.010 | 0.369 | 0.014 | 0.289 | 0.136 |
| usage_weighted_ftr | 0.390 | inside | 0.000 | 0.207 | 0.446 | 0.239 | 0.409 | 0.050 |
| usage_compression | 10.980 | outside | 13.378 | 0.000 | 3.135 | 0.000 | 2.198 | 0.586 |
| usage_concentration | 0.201 | outside | 3.395 | 0.061 | 0.146 | 0.076 | 0.131 | 0.016 |
| rotation_hhi | 0.200 | outside | 7.172 | 0.057 | 0.115 | 0.070 | 0.110 | 0.012 |
| prior_pace_delta | -0.157 | inside | 0.000 | -7.666 | 8.051 | -3.647 | 4.802 | 2.054 |
| known_prior_minutes_share | 1.000 | inside | 0.000 | 0.694 | 1.000 | 0.811 | 1.000 | 0.047 |
| returner_minutes_share | 0.000 | inside | 0.000 | 0.000 | 0.205 | 0.000 | 0.130 | 0.024 |
| prior_age_years | 1.000 | inside | 0.000 | 0.728 | 1.256 | 0.823 | 1.149 | 0.050 |
| w_age | 27.000 | inside | 0.000 | 23.721 | 32.520 | 24.876 | 31.385 | 2.020 |
| center_minutes_share | 0.200 | inside | 0.000 | 0.000 | 0.494 | 0.023 | 0.334 | 0.087 |
| prior_defensive_environment | -4.228 | inside | 0.000 | -7.081 | 5.253 | -5.789 | 3.697 | 2.759 |

Custom ironman: Five named players, 48 minutes each; ironman hypothetical only; not a game simulation

| player | season | minutes | usage_pct | possession_share |
| --- | --- | --- | --- | --- |
| Stephen Curry | 2015-16 | 48.000 | 20.973 | 0.210 |
| Michael Jordan | 1990-91 | 48.000 | 21.153 | 0.212 |
| LeBron James | 2012-13 | 48.000 | 19.531 | 0.195 |
| Tim Duncan | 2002-03 | 48.000 | 18.210 | 0.182 |
| Shaquille O'Neal | 1999-00 | 48.000 | 20.132 | 0.201 |

Target era: 2016-17 preseason using prior league anchors, no fatigue/bench model

Projections: `{"ortg": 122.62733440950142, "drtg": 96.3266179193926, "pace": 92.99036643740251}`

Extrapolation: `{"risk": "high", "rms_outside_domain_distance": 3.468201900994293, "maximum_exceed_iqr_units": 13.377689000134255, "fraction_outside": 0.4444444444444444, "interpretation": "Feature coverage/uncertainty only, not quality or probability; correlated features and unobserved lineup interactions limit interpretation."}`

High: box/context estimates plus a cross-era five-star roster outside ordinary-team support; validation residual bounds are not calibrated ironman coverage.

### Final Frozen Evaluation

Exactly one final evaluation completed after protocol freeze. No retuning followed. Prior-test baseline MAE is comparison-only, never an acceptance threshold.

| target | previous_mae | mae | rmse | r2 |
| --- | --- | --- | --- | --- |
| ortg | 2.450 | 2.362 | 2.964 | 0.394 |
| drtg | 2.392 | 2.316 | 2.779 | 0.322 |
| pace | 1.688 | 1.668 | 2.052 | -0.062 |

2019-20 Lakers (held until final evaluation):

| season | team_name | actual_ortg | actual_drtg | actual_pace | predicted_ortg | predicted_drtg | predicted_pace |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2019-20 | Los Angeles Lakers | 112.000 | 106.300 | 100.900 | 110.710 | 109.551 | 102.540 |

Verdict fixed before final evaluation: **B. NEEDS ONE MORE TARGETED MODEL FIX**.

Readiness checks: `{"defensive_tolerances_pass": false, "rotation_not_worse": true, "usage_not_worse": true, "offense_within_10pct_prior_validation": true, "pace_within_10pct_prior_validation": true}`. In addition to defensive tolerances, require rotation/usage not to regress and offense/pace validation MAE within10% of the prior validation baseline. These practical allowances are fixed before final testing.

Remaining limitations: roster-conditional season aggregates, unknown entrant skill/availability, defensive context and extreme cross-era extrapolation. A permits game-log noise calibration only with these constraints; it does not validate ironman predictions. If defensive acceptance fails, the next step is targeted calibration of extreme defensive buckets and prior context, using validation only and a new genuinely unseen holdout before further model testing. No historical game logs acquired; no UI/application changes.

## Defensive Continuity Model

Reproduce with `python scripts/simulation/defensive_continuity.py`. This task uses only TRAIN 1980-81 through 2013-14 and VALIDATION 2014-15 through 2018-19. No final-test model evaluation is exposed by this script. Existing final artifacts are hashed as bytes only and remain unchanged. The official frozen benchmark remains ORtg MAE2.362/RMSE2.964/R2 .394; DRtg2.316/2.779/.322; pace1.668/2.052/-.062. It describes the previous model, NOT an evaluation of this new defense.

Artifacts: `data/simulation/defensive_continuity_v1_summary.json`, development features and predictions CSVs; new fitted defensive artifact under ignored `artifacts/simulation/`. Offense/pace/usage/rotation artifacts are not changed or refitted.

C=sum(projected minutes of players on the same franchise last season)/240. Franchise IDs preserve continuity through relocation/name changes. Also audit returning top-five minute share and the number of returning rotation players (projected minutes>=8). Source membership is known retrospectively, but target minutes/starts/performance never determine predictors. Selected projected rotations are reproduced exactly from frozen rules; current minutes define roster membership only.

Retained context=C*(previous same-franchise DRtg - previous league scoring anchor). Transferred players carry ZERO direct previous-team context. A returning player's multi-team total is not used to import unrelated team contexts. Individual evidence, age/position mix and proposed rotation remain separate. This is a fitted context adjustment, not copying previous-team DRtg. DWS stays excluded.

Candidates: A frozen current additive model; B continuity OLS; C continuity Ridge(alpha30); D one bounded hinge correction to C. The hinge is fitted on forward predictions entirely within TRAIN (1995-99/2000-04/2005-09/2010-14, each trained on earlier years); knot=training prediction75th percentile, Ridge30 correction clipped to[-2,2] DRtg. D is eligible only if all gates pass and validation MAE improves>=.03 or poor bias improves>=.25 with MAE regression<=.02. B is preferred within.05 MAE of another passing candidate. A is a diagnostic baseline, not eligible because it does not enforce synthetic-context semantics. All coefficients/imputation fit TRAIN only.

Selected: **B_continuity_additive**. Exact raw equation, intercept and imputation medians are in `selected_equation` in the summary JSON. If all candidates fail, the simpler additive equation is also preferred within.05 MAE; this is a research diagnostic, NOT an accepted replacement for the existing engine. Acceptance status is stored in the new artifact.

Tail correction parameters (training-only): `{"knot": 0.42257059351502946, "intercept": 0.23768399457272268, "coefficients": [-0.02422405171386368, -0.20832006826960361], "forward_training_rows": 588, "correction_bounds": [-2.0, 2.0]}`

## Poor-Defense Bias Analysis

Individual evidence representation: **per75**. Compare percentage rates with prior per75 rates using the same continuity Ridge30 equation and select lower validation MAE. Per75 substitutes steals, blocks and defensive rebounds per75 for STL%/BLK%/DRB%; DBPM, age, position and rotation are preserved. No target rates are used. Per75 depends on estimated prior team pace and is not context-free talent.

| representation | mae | rmse | r2 | poor_bias |
| --- | --- | --- | --- | --- |
| percentages | 2.198 | 2.665 | 0.295 | -3.304 |
| per75 | 2.125 | 2.592 | 0.332 | -3.092 |

The observed baseline failure is compressed conditional tail predictions: residual SD 1.799 versus actual 2.963. Poor defenses average 4.356 DRtg above the prior league anchor, but predictions average only 1.114. This creates the -3.242 signed bias. Continuity masking removes an invalid synthetic-context assumption but is not, by itself, a demonstrated cure for this bias. OLS comparison distinguishes this compression from Ridge shrinkage alone. Ablations, missingness and position/rate substitutions below show which explanations have empirical support; they do not isolate unobserved coaching or scheme effects.

Residual spread and conditional means diagnose compression; ablations identify model dependence, not causal player/coaching effects. Tail buckets condition on outcomes, so some mean regression is expected.

Training-fitted feature ablations (diagnostics only, not extra selection candidates):

| ablation | overall_mae | poor_bias | poor_mae |
| --- | --- | --- | --- |
| without_DBPM | 2.189 | -3.352 | 3.352 |
| without_context | 2.198 | -3.258 | 3.258 |
| without_rotation | 2.263 | -3.571 | 3.571 |
| without_box_rates | 2.111 | -3.119 | 3.119 |
| with_position | 2.195 | -3.241 | 3.241 |
| per75_instead_of_percentages | 2.128 | -3.053 | 3.053 |

Centered contributions of the frozen current equation by original bucket:

| bucket | n | w_defensive_box_plus_minus | w_steal_pct | w_block_pct | w_defensive_rebound_pct | prior_defensive_environment | w_age | rotation_hhi |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| elite | 20 | -0.876 | 0.000 | -0.251 | -0.102 | -0.544 | -0.103 | 0.231 |
| above_average | 34 | -0.247 | -0.012 | -0.168 | -0.093 | -0.166 | 0.066 | 0.207 |
| average | 39 | 0.009 | -0.009 | -0.144 | -0.084 | -0.037 | 0.064 | 0.240 |
| below_average | 38 | 0.278 | -0.012 | -0.126 | -0.059 | 0.149 | 0.143 | 0.330 |
| poor | 19 | 0.886 | -0.014 | -0.014 | -0.073 | 0.614 | 0.104 | 0.367 |

Missingness, continuity and old inherited context:

| bucket | defensive_missing_share | returning_minutes_share | prior_defensive_environment |
| --- | --- | --- | --- |
| elite | 0.030 | 0.651 | -2.280 |
| above_average | 0.032 | 0.666 | -1.162 |
| average | 0.039 | 0.607 | -0.781 |
| below_average | 0.044 | 0.576 | -0.230 |
| poor | 0.041 | 0.603 | 1.143 |

Baseline error by continuity:

| continuity | n | mae | bias |
| --- | --- | --- | --- |
| low | 3 | 2.830 | 1.983 |
| moderate | 88 | 2.284 | -0.759 |
| high | 59 | 2.021 | -0.516 |

Box-score measures do not observe schemes, matchup assignments, rim deterrence, defensive communication or coaching. Strong rate persistence does not make these complete defense. The earlier qualified-transfer correlations were DBPM .600, STL% .740, BLK% .874, DRB% .876, DWS .459 and defensive environment .049. Missing source values are documented separately from imputed model values; no new defensive statistics are fabricated.

## Defensive Tail Validation

Retain the exact original thresholds (including unrounded overall bound), and preserve the original era-relative difference buckets for acceptance comparability. ALSO report season-relative DRtg quintile rankings as requested. A requires all gates in BOTH views: the previous19-team poor bucket cannot be hidden by redefining it. Positive error underrates defense; negative error overrates it.

Unchanged criteria: `{"overall_mae_max": 2.2896266864236936, "anchor_validation_mae": 2.410133354130204, "elite_poor_mae_max": 5.0, "worst_bucket_abs_bias_max": 3.0, "abs_error_p90_max": 5.5, "subera_mae_over_anchor_max": 0.15, "box_heavy_abs_bias_max": 2.0, "rationale": "Require >=5% validation MAE improvement on league anchor, <=5 extreme-bucket MAE, <=3 bucket bias, <=5.5 p90 error, <=0.15 subera regression, <=2 box-heavy bias. Practical bounds declared before candidate evaluation; no final-test values used."}`

| candidate | view | mae | rmse | r2 | p90 | actual_sd | predicted_sd | accepted |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A_current | legacy | 2.191 | 2.653 | 0.301 | 4.302 | 2.963 | 1.799 | False |
| A_current | ranking | 2.191 | 2.653 | 0.301 | 4.302 | 2.963 | 1.799 | True |
| B_continuity_additive | legacy | 2.125 | 2.595 | 0.331 | 4.142 | 2.963 | 1.836 | False |
| B_continuity_additive | ranking | 2.125 | 2.595 | 0.331 | 4.142 | 2.963 | 1.836 | True |
| C_continuity_ridge | legacy | 2.125 | 2.592 | 0.332 | 4.137 | 2.963 | 1.808 | False |
| C_continuity_ridge | ranking | 2.125 | 2.592 | 0.332 | 4.137 | 2.963 | 1.808 | True |
| D_bounded_tail | legacy | 2.088 | 2.554 | 0.352 | 4.062 | 2.963 | 1.685 | False |
| D_bounded_tail | ranking | 2.088 | 2.554 | 0.352 | 4.062 | 2.963 | 1.685 | True |

Bucket details:

| candidate | view | bucket | n | mae | bias | median_bias | p90_abs_error | actual_residual | predicted_residual |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A_current | legacy | elite | 20 | 2.538 | 1.804 | 1.732 | 4.397 | -4.205 | -2.401 |
| A_current | legacy | above_average | 34 | 1.618 | 0.994 | 1.126 | 3.036 | -2.163 | -1.168 |
| A_current | legacy | average | 39 | 2.087 | -0.705 | -1.103 | 3.846 | -0.012 | -0.717 |
| A_current | legacy | below_average | 38 | 2.104 | -1.898 | -1.697 | 3.927 | 1.845 | -0.053 |
| A_current | legacy | poor | 19 | 3.242 | -3.242 | -3.355 | 4.985 | 4.356 | 1.114 |
| A_current | ranking | elite | 30 | 2.255 | 1.550 | 1.697 | 4.017 | -3.693 | -2.143 |
| A_current | ranking | above_average | 29 | 1.762 | 0.677 | 0.974 | 3.332 | -1.711 | -1.034 |
| A_current | ranking | average | 30 | 1.892 | -0.634 | -0.878 | 3.824 | -0.017 | -0.650 |
| A_current | ranking | below_average | 30 | 2.206 | -1.716 | -1.507 | 4.130 | 1.610 | -0.106 |
| A_current | ranking | poor | 31 | 2.808 | -2.805 | -2.878 | 4.972 | 3.465 | 0.660 |
| B_continuity_additive | legacy | elite | 20 | 2.573 | 1.935 | 1.889 | 4.387 | -4.205 | -2.270 |
| B_continuity_additive | legacy | above_average | 34 | 1.683 | 1.065 | 1.193 | 3.109 | -2.163 | -1.098 |
| B_continuity_additive | legacy | average | 39 | 1.993 | -0.523 | -0.841 | 3.513 | -0.012 | -0.535 |
| B_continuity_additive | legacy | below_average | 38 | 1.957 | -1.726 | -1.426 | 4.033 | 1.845 | 0.119 |
| B_continuity_additive | legacy | poor | 19 | 3.056 | -3.056 | -2.995 | 4.807 | 4.356 | 1.300 |
| B_continuity_additive | ranking | elite | 30 | 2.315 | 1.665 | 1.875 | 4.143 | -3.693 | -2.028 |
| B_continuity_additive | ranking | above_average | 29 | 1.822 | 0.752 | 1.163 | 3.486 | -1.711 | -0.959 |
| B_continuity_additive | ranking | average | 30 | 1.792 | -0.471 | -0.757 | 3.402 | -0.017 | -0.488 |
| B_continuity_additive | ranking | below_average | 30 | 2.060 | -1.527 | -1.227 | 3.938 | 1.610 | 0.083 |
| B_continuity_additive | ranking | poor | 31 | 2.612 | -2.612 | -2.829 | 4.776 | 3.465 | 0.852 |
| C_continuity_ridge | legacy | elite | 20 | 2.573 | 1.959 | 1.935 | 4.415 | -4.205 | -2.246 |
| C_continuity_ridge | legacy | above_average | 34 | 1.676 | 1.071 | 1.180 | 3.128 | -2.163 | -1.091 |
| C_continuity_ridge | legacy | average | 39 | 1.982 | -0.530 | -0.863 | 3.481 | -0.012 | -0.542 |
| C_continuity_ridge | legacy | below_average | 38 | 1.953 | -1.732 | -1.423 | 4.041 | 1.845 | 0.113 |
| C_continuity_ridge | legacy | poor | 19 | 3.092 | -3.092 | -3.019 | 4.830 | 4.356 | 1.264 |
| C_continuity_ridge | ranking | elite | 30 | 2.313 | 1.684 | 1.882 | 4.141 | -3.693 | -2.009 |
| C_continuity_ridge | ranking | above_average | 29 | 1.813 | 0.756 | 1.111 | 3.461 | -1.711 | -0.955 |
| C_continuity_ridge | ranking | average | 30 | 1.780 | -0.476 | -0.785 | 3.415 | -0.017 | -0.493 |
| C_continuity_ridge | ranking | below_average | 30 | 2.053 | -1.536 | -1.247 | 3.954 | 1.610 | 0.074 |
| C_continuity_ridge | ranking | poor | 31 | 2.637 | -2.637 | -2.827 | 4.800 | 3.465 | 0.828 |
| D_bounded_tail | legacy | elite | 20 | 2.673 | 2.227 | 2.234 | 4.723 | -4.205 | -1.978 |
| D_bounded_tail | legacy | above_average | 34 | 1.749 | 1.315 | 1.384 | 3.388 | -2.163 | -0.848 |
| D_bounded_tail | legacy | average | 39 | 1.867 | -0.332 | -0.594 | 3.315 | -0.012 | -0.344 |
| D_bounded_tail | legacy | below_average | 38 | 1.818 | -1.568 | -1.238 | 3.754 | 1.845 | 0.277 |
| D_bounded_tail | legacy | poor | 19 | 3.075 | -3.075 | -3.145 | 4.597 | 4.356 | 1.281 |
| D_bounded_tail | ranking | elite | 30 | 2.410 | 1.954 | 2.143 | 4.396 | -3.693 | -1.738 |
| D_bounded_tail | ranking | above_average | 29 | 1.836 | 0.980 | 1.348 | 3.399 | -1.711 | -0.731 |
| D_bounded_tail | ranking | average | 30 | 1.672 | -0.280 | -0.568 | 3.133 | -0.017 | -0.297 |
| D_bounded_tail | ranking | below_average | 30 | 1.933 | -1.370 | -1.171 | 3.672 | 1.610 | 0.240 |
| D_bounded_tail | ranking | poor | 31 | 2.566 | -2.558 | -2.752 | 4.559 | 3.465 | 0.907 |

Acceptance checks:

| candidate | view | overall | extremes | bucket_bias | p90 | suberas | box_heavy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A_current | legacy | True | True | False | True | True | True |
| A_current | ranking | True | True | True | True | True | True |
| B_continuity_additive | legacy | True | True | False | True | True | True |
| B_continuity_additive | ranking | True | True | True | True | True | True |
| C_continuity_ridge | legacy | True | True | False | True | True | True |
| C_continuity_ridge | ranking | True | True | True | True | True | True |
| D_bounded_tail | legacy | True | True | False | True | True | True |
| D_bounded_tail | ranking | True | True | True | True | True | True |

Sub-era and box-heavy diagnostics:

| candidate | view | suberas | box_heavy |
| --- | --- | --- | --- |
| A_current | legacy | {"2015-2016": {"mae": 2.2264664048761538, "anchor_mae": 2.5557914462745823}, "2017-2019": {"mae": 2.1680748732208794, "anchor_mae": 2.313027959367288}} | {"n": 12, "bias": -0.6284721639249634} |
| A_current | ranking | {"2015-2016": {"mae": 2.2264664048761538, "anchor_mae": 2.5557914462745823}, "2017-2019": {"mae": 2.1680748732208794, "anchor_mae": 2.313027959367288}} | {"n": 12, "bias": -0.6284721639249634} |
| B_continuity_additive | legacy | {"2015-2016": {"mae": 2.2063403945926656, "anchor_mae": 2.5557914462745823}, "2017-2019": {"mae": 2.0715089388295818, "anchor_mae": 2.313027959367288}} | {"n": 12, "bias": -0.6477360433606769} |
| B_continuity_additive | ranking | {"2015-2016": {"mae": 2.2063403945926656, "anchor_mae": 2.5557914462745823}, "2017-2019": {"mae": 2.0715089388295818, "anchor_mae": 2.313027959367288}} | {"n": 12, "bias": -0.6477360433606769} |
| C_continuity_ridge | legacy | {"2015-2016": {"mae": 2.2028067498258244, "anchor_mae": 2.5557914462745823}, "2017-2019": {"mae": 2.072517263521187, "anchor_mae": 2.313027959367288}} | {"n": 12, "bias": -0.5883835071507436} |
| C_continuity_ridge | ranking | {"2015-2016": {"mae": 2.2028067498258244, "anchor_mae": 2.5557914462745823}, "2017-2019": {"mae": 2.072517263521187, "anchor_mae": 2.313027959367288}} | {"n": 12, "bias": -0.5883835071507436} |
| D_bounded_tail | legacy | {"2015-2016": {"mae": 2.223427384270464, "anchor_mae": 2.5557914462745823}, "2017-2019": {"mae": 1.9982847101405243, "anchor_mae": 2.313027959367288}} | {"n": 12, "bias": -0.24362045765141715} |
| D_bounded_tail | ranking | {"2015-2016": {"mae": 2.223427384270464, "anchor_mae": 2.5557914462745823}, "2017-2019": {"mae": 1.9982847101405243, "anchor_mae": 2.313027959367288}} | {"n": 12, "bias": -0.24362045765141715} |

High/low continuity and named development sanity checks (major star arrivals do not necessarily mean low overall continuity):

| season | team_name | returning_minutes_share | returning_top5_share | returning_rotation_players | transfer_or_unknown_minutes_share | retained_defensive_context | actual_drtg | predicted_drtg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2007-08 | Boston Celtics | 0.402 | 0.347 | 5 | 0.598 | -0.209 | 98.900 | 105.664 |
| 2010-11 | Miami Heat | 0.375 | 0.386 | 5 | 0.625 | -1.578 | 103.500 | 106.024 |
| 2016-17 | Golden State Warriors | 0.549 | 0.620 | 5 | 0.451 | -1.869 | 104.000 | 103.341 |
| 2018-19 | Boston Celtics | 0.954 | 1.000 | 11 | 0.046 | -5.109 | 107.800 | 105.445 |
| 2014-15 | San Antonio Spurs | 0.954 | 1.000 | 12 | 0.046 | -4.880 | 102.000 | 102.625 |
| 2015-16 | Chicago Bulls | 0.948 | 1.000 | 10 | 0.052 | -2.094 | 106.500 | 103.567 |
| 2015-16 | Oklahoma City Thunder | 0.918 | 1.000 | 11 | 0.082 | -0.925 | 105.600 | 105.702 |
| 2018-19 | Miami Heat | 0.909 | 1.000 | 11 | 0.091 | -2.686 | 107.600 | 106.790 |
| 2017-18 | Boston Celtics | 0.314 | 0.373 | 4 | 0.686 | -0.356 | 103.900 | 109.859 |
| 2014-15 | Dallas Mavericks | 0.322 | 0.432 | 3 | 0.678 | 0.381 | 106.400 | 108.477 |
| 2017-18 | Los Angeles Clippers | 0.323 | 0.409 | 3 | 0.677 | -0.430 | 110.200 | 109.325 |
| 2016-17 | Brooklyn Nets | 0.353 | 0.469 | 5 | 0.647 | 1.307 | 110.700 | 108.778 |
| 2017-18 | Indiana Pacers | 0.355 | 0.414 | 5 | 0.645 | -0.261 | 108.100 | 110.219 |

2019-20 Lakers are in the frozen test period: no new inference or individual-error analysis was performed. Previously recorded context remains in the earlier experiment only.

## Synthetic-Roster Defensive Semantics

A completely synthetic roster has continuity=0 and retained context=0, even if selected players come from historically excellent defenses or several share a franchise name. Predictions use individual transferable evidence, proposed minutes, age/position mix and target-era league anchor. No previous NBA team's defensive environment is inherited. Historical context dispersion may raise uncertainty but does not increase predicted defensive quality.

Five-player ironman only, 48 minutes each, 240 total; DRtg is a diagnostic estimate, not precise truth or a game simulation

| player | season | minutes | defensive_fields_available |
| --- | --- | --- | --- |
| Stephen Curry | 2015-16 | 48.000 | 4 |
| Michael Jordan | 1990-91 | 48.000 | 4 |
| LeBron James | 2012-13 | 48.000 | 4 |
| Tim Duncan | 2002-03 | 48.000 | 4 |
| Shaquille O'Neal | 1999-00 | 48.000 | 4 |

Custom DRtg diagnostic: `99.37391626755516`; continuity0, retained context0. Do not interpret this number as precise truth. Individual evidence: `{"w_defensive_box_plus_minus": 2.3400000000000003, "w_steals_per_75": 1.589494678276071, "w_blocks_per_75": 1.6103301901598672, "w_defensive_rebounds_per_75": 7.098033251001157, "w_age": 27.0, "center_minutes_share": 0.2, "rotation_hhi": 0.20000000000000004}`

Unchanged custom offense/pace: `{"ortg": 122.62733440950142, "pace": 92.99036643740251}`

## Defensive Uncertainty

This is a coverage/uncertainty indicator, not quality or a calibrated interval. HIGH if continuity<.35, missing weighted defensive fields>.20, context SD>training p90 AND source diversity>training p75, or feature extrapolation is high (maximum beyond training range>1 IQR or >15% fields outside). Otherwise MODERATE if continuity<.65, missing>.05, context SD>training p75 or any feature outside training range; otherwise LOW. Source diversity=1-sum(source-season/franchise minute shares^2). Low continuity therefore always raises uncertainty; weak/missing evidence, distinct contexts and extrapolation are explicitly visible. Marginal feature ranges cannot establish joint causal support.

Training-derived context thresholds: `{"context_sd_p75": 2.4378158346707726, "context_sd_p90": 3.0488183241995404, "diversity_p75": 0.6450120906488895, "high_continuity_below": 0.35, "moderate_continuity_below": 0.65, "high_missing_above": 0.2, "moderate_missing_above": 0.05}`

Custom uncertainty: `{"category": "HIGH", "high_flags": {"low_continuity": true, "weak_availability": false, "diverse_contexts": false, "extrapolation": true}, "continuity": 0.0, "missing_share": 0.0, "source_context_sd": 1.2931461134334779, "source_diversity": 0.7999999999999999, "interpretation": "Uncalibrated uncertainty/coverage indicator, not defensive quality or a probability interval"}`

| feature | value | status | exceed_iqr_units | min | max | p01 | p99 | scale |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| w_defensive_box_plus_minus | 2.340 | outside | 1.389 | -1.233 | 1.440 | -1.055 | 1.085 | 0.648 |
| w_steals_per_75 | 1.589 | inside | 0.000 | 0.905 | 1.959 | 0.980 | 1.663 | 0.188 |
| w_blocks_per_75 | 1.610 | outside | 0.976 | 0.336 | 1.414 | 0.413 | 1.212 | 0.201 |
| w_defensive_rebounds_per_75 | 7.098 | outside | 1.842 | 3.466 | 6.004 | 3.705 | 5.565 | 0.594 |
| w_age | 27.000 | inside | 0.000 | 23.721 | 32.520 | 24.876 | 31.385 | 2.020 |
| center_minutes_share | 0.200 | inside | 0.000 | 0.000 | 0.494 | 0.023 | 0.334 | 0.087 |
| rotation_hhi | 0.200 | outside | 7.172 | 0.057 | 0.115 | 0.070 | 0.110 | 0.012 |
| retained_defensive_context | 0.000 | inside | 0.000 | -7.698 | 6.259 | -5.775 | 3.822 | 2.832 |
| returning_minutes_share | 0.000 | near_edge | 0.000 | 0.000 | 0.985 | 0.234 | 0.933 | 0.204 |
| defensive_missing_share | 0.000 | inside | 0.000 | 0.000 | 0.306 | 0.000 | 0.189 | 0.047 |
| source_context_sd | 1.293 | inside | 0.000 | 0.030 | 5.369 | 0.240 | 4.389 | 1.189 |
| source_diversity | 0.800 | inside | 0.000 | 0.029 | 0.940 | 0.142 | 0.858 | 0.246 |

Regression checks: `{"max_validation_minutes_difference": 3.552713678800501e-15, "rotation_configuration": "balanced_prior", "usage_configuration": {"name": "role_shrink10", "alpha": 1.0, "shrink": 0.1}, "ortg": {"unchanged": true, "validation": {"mae": 2.1712773887652514, "rmse": 2.79517361544265, "r2": 0.4266667403949188}}, "pace": {"unchanged": true, "validation": {"mae": 1.5026906354267182, "rmse": 1.9902300027393711, "r2": 0.510654843167533}}}`. Protected baseline artifacts and official test records unchanged: `True`. New final-test evaluations:0. Previous experiments remain above.

**B. DEFENSIVE MODEL STILL OUTSIDE ACCEPTANCE TOLERANCE**.

If accepted, next task is team game-log acquisition and noise/home-court calibration with frozen team-strength methodology; current official test does not validate this new defense. Otherwise retain the failed tolerances explicitly and do not start game-log calibration. No game logs, Simulator UI, frontend or .NET application changes were made.

## Rolling-Origin Defensive Diagnostics

Commands: `python scripts/simulation/calibrate_defensive_tails.py --prepare`, tests, then `--validate` exactly once. `--report` only renders stored results. No frozen 2019-20 through 2025-26 evaluation is exposed. Official test artifacts and all original model artifacts are protected by hashes.

Frozen base formulation: continuity-aware additive OLS with DBPM, steals/blocks/defensive rebounds per75, age, position mix, rotation concentration and retained context. No new feature families. Train starts1980-81. Warmup through1984-85 is excluded from calibration; expanding five-season evaluation blocks cover1985-86 through2013-14 (last block four seasons). Every base prediction uses coefficients and imputation trained strictly before its target season. The base Ridge30 and no-continuity variants are diagnostics only, not a new validation search.

| fold | variant | train_start | train_end | evaluate_start | evaluate_end | train_n | n | mae | rmse | r2 | actual_residual_sd | predicted_residual_sd | actual_on_predicted_slope | actual_on_predicted_intercept | residual_spearman | elite_bias | poor_bias | p90_abs_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | frozen_formulation_OLS | 1981 | 1985 | 1986 | 1990 | 115 | 120 | 2.014 | 2.453 | 0.088 | 2.620 | 1.888 | 0.652 | -0.566 | 0.469 | 2.885 | -2.478 | 4.033 |
| 1 | same_features_Ridge30 | 1981 | 1985 | 1986 | 1990 | 115 | 120 | 1.901 | 2.303 | 0.197 | 2.620 | 1.514 | 0.879 | -0.483 | 0.503 | 3.132 | -2.588 | 3.712 |
| 1 | without_continuity | 1981 | 1985 | 1986 | 1990 | 115 | 120 | 1.939 | 2.355 | 0.160 | 2.620 | 1.863 | 0.721 | -0.542 | 0.492 | 2.969 | -2.320 | 3.938 |
| 2 | frozen_formulation_OLS | 1981 | 1990 | 1991 | 1995 | 235 | 135 | 1.916 | 2.457 | 0.294 | 3.124 | 1.976 | 0.976 | -0.196 | 0.655 | 2.526 | -1.989 | 4.080 |
| 2 | same_features_Ridge30 | 1981 | 1990 | 1991 | 1995 | 235 | 135 | 1.921 | 2.455 | 0.295 | 3.124 | 1.839 | 1.048 | -0.093 | 0.652 | 2.609 | -2.183 | 4.139 |
| 2 | without_continuity | 1981 | 1990 | 1991 | 1995 | 235 | 135 | 1.958 | 2.484 | 0.278 | 3.124 | 2.025 | 0.939 | -0.262 | 0.643 | 2.586 | -1.885 | 4.123 |
| 3 | frozen_formulation_OLS | 1981 | 1995 | 1996 | 2000 | 370 | 145 | 2.711 | 3.373 | 0.258 | 3.769 | 2.075 | 0.955 | -1.107 | 0.534 | 3.875 | -2.699 | 5.853 |
| 3 | same_features_Ridge30 | 1981 | 1995 | 1996 | 2000 | 370 | 145 | 2.708 | 3.360 | 0.264 | 3.769 | 1.996 | 0.990 | -1.039 | 0.534 | 3.893 | -2.802 | 5.767 |
| 3 | without_continuity | 1981 | 1995 | 1996 | 2000 | 370 | 145 | 2.684 | 3.370 | 0.260 | 3.769 | 2.116 | 0.953 | -1.157 | 0.549 | 3.886 | -2.530 | 5.847 |
| 4 | frozen_formulation_OLS | 1981 | 2000 | 2001 | 2005 | 515 | 146 | 2.442 | 3.059 | 0.151 | 3.530 | 1.886 | 1.019 | 0.826 | 0.524 | 1.973 | -3.525 | 5.043 |
| 4 | same_features_Ridge30 | 1981 | 2000 | 2001 | 2005 | 515 | 146 | 2.443 | 3.061 | 0.150 | 3.530 | 1.836 | 1.047 | 0.862 | 0.525 | 2.024 | -3.566 | 5.061 |
| 4 | without_continuity | 1981 | 2000 | 2001 | 2005 | 515 | 146 | 2.441 | 3.072 | 0.143 | 3.530 | 1.941 | 0.977 | 0.768 | 0.521 | 2.003 | -3.423 | 5.071 |
| 5 | frozen_formulation_OLS | 1981 | 2005 | 2006 | 2010 | 661 | 150 | 1.977 | 2.542 | 0.406 | 3.275 | 1.993 | 1.032 | -0.080 | 0.632 | 2.707 | -2.997 | 4.348 |
| 5 | same_features_Ridge30 | 1981 | 2005 | 2006 | 2010 | 661 | 150 | 1.977 | 2.541 | 0.407 | 3.275 | 1.950 | 1.056 | -0.047 | 0.635 | 2.735 | -3.051 | 4.326 |
| 5 | without_continuity | 1981 | 2005 | 2006 | 2010 | 661 | 150 | 2.036 | 2.558 | 0.399 | 3.275 | 1.972 | 1.032 | -0.054 | 0.621 | 2.719 | -3.093 | 4.222 |
| 6 | frozen_formulation_OLS | 1981 | 2010 | 2011 | 2014 | 811 | 120 | 2.339 | 2.927 | 0.218 | 3.519 | 1.954 | 1.036 | -0.571 | 0.554 | 4.152 | -2.139 | 4.757 |
| 6 | same_features_Ridge30 | 1981 | 2010 | 2011 | 2014 | 811 | 120 | 2.337 | 2.922 | 0.221 | 3.519 | 1.919 | 1.058 | -0.554 | 0.559 | 4.176 | -2.166 | 4.746 |
| 6 | without_continuity | 1981 | 2010 | 2011 | 2014 | 811 | 120 | 2.344 | 2.943 | 0.209 | 3.519 | 2.011 | 0.991 | -0.572 | 0.546 | 4.114 | -2.106 | 4.900 |

Feature availability by fold:

| fold | variant | availability | n | mae | rmse | r2 | actual_residual_sd | predicted_residual_sd | actual_on_predicted_slope | actual_on_predicted_intercept | residual_spearman | elite_bias | poor_bias | p90_abs_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | frozen_formulation_OLS | <=5% missing | 64 | 1.829 | 2.226 | 0.188 | 2.530 | 1.830 | 0.751 | -0.640 | 0.561 | 2.643 | -2.207 | 3.115 |
| 1 | frozen_formulation_OLS | >5% missing | 56 | 2.226 | 2.690 | -0.028 | 2.714 | 1.937 | 0.534 | -0.426 | 0.328 | 3.556 | -2.816 | 4.336 |
| 1 | same_features_Ridge30 | <=5% missing | 64 | 1.729 | 2.103 | 0.275 | 2.530 | 1.431 | 1.032 | -0.510 | 0.579 | 3.110 | -2.437 | 3.448 |
| 1 | same_features_Ridge30 | >5% missing | 56 | 2.097 | 2.511 | 0.104 | 2.714 | 1.587 | 0.719 | -0.380 | 0.361 | 3.416 | -3.206 | 4.047 |
| 1 | without_continuity | <=5% missing | 64 | 1.727 | 2.134 | 0.254 | 2.530 | 1.819 | 0.811 | -0.594 | 0.575 | 2.631 | -2.042 | 3.321 |
| 1 | without_continuity | >5% missing | 56 | 2.181 | 2.585 | 0.050 | 2.714 | 1.893 | 0.612 | -0.431 | 0.366 | 4.038 | -2.334 | 4.097 |
| 2 | frozen_formulation_OLS | <=5% missing | 84 | 1.908 | 2.507 | 0.190 | 3.004 | 1.882 | 0.880 | -0.311 | 0.543 | 2.755 | -1.741 | 4.035 |
| 2 | frozen_formulation_OLS | >5% missing | 51 | 1.931 | 2.372 | 0.298 | 3.120 | 1.765 | 1.147 | -0.250 | 0.647 | 3.069 | -3.386 | 4.086 |
| 2 | same_features_Ridge30 | <=5% missing | 84 | 1.912 | 2.503 | 0.193 | 3.004 | 1.739 | 0.947 | -0.212 | 0.539 | 2.871 | -1.898 | 3.964 |
| 2 | same_features_Ridge30 | >5% missing | 51 | 1.935 | 2.375 | 0.296 | 3.120 | 1.657 | 1.226 | -0.139 | 0.646 | 3.052 | -3.583 | 4.367 |
| 2 | without_continuity | <=5% missing | 84 | 1.947 | 2.527 | 0.177 | 3.004 | 1.956 | 0.842 | -0.395 | 0.545 | 2.780 | -1.626 | 4.127 |
| 2 | without_continuity | >5% missing | 51 | 1.977 | 2.412 | 0.273 | 3.120 | 1.781 | 1.108 | -0.292 | 0.636 | 3.263 | -3.041 | 3.837 |
| 3 | frozen_formulation_OLS | <=5% missing | 107 | 2.711 | 3.383 | 0.249 | 3.762 | 2.127 | 0.960 | -1.276 | 0.553 | 3.629 | -2.719 | 5.850 |
| 3 | frozen_formulation_OLS | >5% missing | 38 | 2.711 | 3.347 | 0.217 | 3.644 | 1.830 | 0.824 | -0.626 | 0.445 | 3.895 | -2.140 | 5.443 |
| 3 | same_features_Ridge30 | <=5% missing | 107 | 2.706 | 3.372 | 0.254 | 3.762 | 2.046 | 0.994 | -1.207 | 0.553 | 3.659 | -2.799 | 5.743 |
| 3 | same_features_Ridge30 | >5% missing | 38 | 2.715 | 3.328 | 0.225 | 3.644 | 1.761 | 0.858 | -0.571 | 0.439 | 3.882 | -2.258 | 5.298 |
| 3 | without_continuity | <=5% missing | 107 | 2.687 | 3.384 | 0.249 | 3.762 | 2.190 | 0.962 | -1.375 | 0.574 | 3.613 | -2.504 | 5.827 |
| 3 | without_continuity | >5% missing | 38 | 2.676 | 3.331 | 0.224 | 3.644 | 1.857 | 0.814 | -0.547 | 0.419 | 3.869 | -2.112 | 5.531 |
| 4 | frozen_formulation_OLS | <=5% missing | 106 | 2.493 | 3.154 | 0.100 | 3.576 | 1.887 | 0.953 | 0.621 | 0.489 | 2.081 | -3.602 | 5.282 |
| 4 | frozen_formulation_OLS | >5% missing | 40 | 2.305 | 2.791 | 0.190 | 3.254 | 1.739 | 1.138 | 1.180 | 0.548 | 1.970 | -3.993 | 4.295 |
| 4 | same_features_Ridge30 | <=5% missing | 106 | 2.494 | 3.153 | 0.100 | 3.576 | 1.838 | 0.977 | 0.655 | 0.488 | 2.140 | -3.637 | 5.236 |
| 4 | same_features_Ridge30 | >5% missing | 40 | 2.308 | 2.801 | 0.184 | 3.254 | 1.688 | 1.175 | 1.221 | 0.553 | 1.999 | -4.048 | 4.342 |
| 4 | without_continuity | <=5% missing | 106 | 2.466 | 3.149 | 0.103 | 3.576 | 1.949 | 0.923 | 0.551 | 0.491 | 2.129 | -3.458 | 5.430 |
| 4 | without_continuity | >5% missing | 40 | 2.374 | 2.859 | 0.151 | 3.254 | 1.811 | 1.047 | 1.185 | 0.542 | 1.903 | -3.936 | 4.356 |
| 5 | frozen_formulation_OLS | <=5% missing | 119 | 1.988 | 2.493 | 0.459 | 3.336 | 2.000 | 1.116 | -0.177 | 0.673 | 2.911 | -2.691 | 4.349 |
| 5 | frozen_formulation_OLS | >5% missing | 31 | 1.937 | 2.723 | -0.352 | 2.436 | 1.632 | 0.234 | 0.964 | 0.165 | 4.136 | -3.901 | 3.773 |
| 5 | same_features_Ridge30 | <=5% missing | 119 | 1.990 | 2.496 | 0.458 | 3.336 | 1.955 | 1.142 | -0.140 | 0.674 | 2.937 | -2.750 | 4.328 |
| 5 | same_features_Ridge30 | >5% missing | 31 | 1.928 | 2.708 | -0.337 | 2.436 | 1.601 | 0.248 | 0.966 | 0.183 | 4.081 | -3.936 | 3.705 |
| 5 | without_continuity | <=5% missing | 119 | 2.076 | 2.538 | 0.439 | 3.336 | 1.996 | 1.089 | -0.181 | 0.654 | 2.930 | -2.886 | 4.223 |
| 5 | without_continuity | >5% missing | 31 | 1.881 | 2.635 | -0.266 | 2.436 | 1.554 | 0.324 | 0.937 | 0.245 | 4.017 | -3.757 | 3.540 |
| 6 | frozen_formulation_OLS | <=5% missing | 96 | 2.334 | 2.961 | 0.229 | 3.526 | 2.002 | 1.066 | -0.931 | 0.580 | 4.322 | -1.834 | 4.874 |
| 6 | frozen_formulation_OLS | >5% missing | 24 | 2.356 | 2.784 | -1.028 | 2.203 | 1.465 | -0.026 | 1.405 | -0.061 | 0.118 | null | 4.106 |
| 6 | same_features_Ridge30 | <=5% missing | 96 | 2.335 | 2.960 | 0.230 | 3.526 | 1.967 | 1.086 | -0.913 | 0.583 | 4.345 | -1.860 | 4.906 |
| 6 | same_features_Ridge30 | >5% missing | 24 | 2.345 | 2.768 | -1.004 | 2.203 | 1.430 | -0.021 | 1.403 | -0.036 | 0.159 | null | 4.047 |
| 6 | without_continuity | <=5% missing | 96 | 2.342 | 2.975 | 0.222 | 3.526 | 2.052 | 1.023 | -0.938 | 0.576 | 4.325 | -1.834 | 5.278 |
| 6 | without_continuity | >5% missing | 24 | 2.350 | 2.811 | -1.067 | 2.203 | 1.571 | 0.018 | 1.387 | -0.019 | 0.063 | null | 4.060 |

Training diagnostic interpretation: predicted residual SD is below actual SD in all six OLS folds, not only one era. Ridge30 narrows it further, but unregularized OLS also has the gap: regularization is not the sole explanation. Removing continuity does not resolve the gap, and does not consistently improve MAE. Higher missing-feature share is associated with weaker residual ranking in some later folds, but not uniformly higher MAE; these observational strata do not establish a causal missingness effect. The largest OLS MAE and p90 occur in 1995-96 through1999-00. This is persistent unexplained outcome variance with era-dependent offsets, not evidence for a universal tail expansion.

## OOF Tail Calibration

816 genuine training OOF rows; 115 warmup-only rows. Source CSV: `data/simulation/defensive_oof_calibration_v1_training_oof.csv`, including each row's fold and latest base training year. No in-sample fitted predictions calibrate any coefficient.

Let x=base_D Rtg-prior_league_anchor and y=actual_D Rtg-prior_league_anchor. Symmetric gamma=sum(x*y)/sum(x^2), through zero. Asymmetric gamma_better and gamma_worse fit the same equation separately on x<0 and x>=0; actual defensive buckets are never inference inputs. An intercept+slope regression is reported diagnostically, not silently added as another family.

Calibration coefficients:

| candidate | kind | fitting_rows | fitting_max_year | gamma | gamma_better | gamma_worse | n_better | n_worse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A_identity | A_identity | 816 | 2014 | 1.000 | null | null | null | null |
| B_symmetric | B_symmetric | 816 | 2014 | 0.964 | null | null | null | null |
| C_asymmetric | C_asymmetric | 816 | 2014 | null | 1.059 | 0.689 | 472.000 | 344.000 |

Intercept diagnostic: `{"intercept": -0.30429846682220835, "gamma": 0.9230348972004516, "folds": [{"fold": 1, "intercept": -0.565675318174581, "gamma": 0.6516378428422153}, {"fold": 2, "intercept": -0.19556113026188265, "gamma": 0.9757136279416174}, {"fold": 3, "intercept": -1.1069649470059213, "gamma": 0.9554088994995203}, {"fold": 4, "intercept": 0.8255590413498892, "gamma": 1.0190724617904805}, {"fold": 5, "intercept": -0.07983376580442672, "gamma": 1.0319756923717311}, {"fold": 6, "intercept": -0.5706493813918933, "gamma": 1.0359299950960066}], "production_candidate": false, "reason": "Diagnostic only; no extra intercept family selected on validation"}`

Optional piecewise justification: `{"supported": false, "folds": [{"fold": 1, "center_gamma": 0.5323775830574413, "tail_gamma": 0.7106474827397047, "difference": 0.17826989968226337, "tail_n": 30}, {"fold": 2, "center_gamma": 1.2294800737940774, "tail_gamma": 0.9231110981643608, "difference": -0.30636897562971666, "tail_n": 34}, {"fold": 3, "center_gamma": 1.248951951519106, "tail_gamma": 1.0493100968866618, "difference": -0.19964185463244433, "tail_n": 37}, {"fold": 4, "center_gamma": 0.627401684705194, "tail_gamma": 0.879932393059067, "difference": 0.25253070835387303, "tail_n": 37}, {"fold": 5, "center_gamma": 1.2929616133517594, "tail_gamma": 0.9523317686785533, "difference": -0.3406298446732061, "tail_n": 38}, {"fold": 6, "center_gamma": 1.1560028665823354, "tail_gamma": 1.0801092339941616, "difference": -0.07589363258817383, "tail_n": 30}], "rule": "Evaluate one hinge only if median tail-center gamma gap>=0.20, same direction in>=4/6 folds, >=20 tail rows per fold"}`. If supported, use one continuous signed hinge at training-OOF75th percentile |x|, with both coefficients fit on training OOF rows. No isotonic/quantile mapping or outcome-bucket inference.

## Defensive Residual Spread

Pooled uncalibrated OOF diagnostics: `{"n": 816, "mae": 2.2392400275208844, "rmse": 2.8332741211038592, "r2": 0.36824432148076935, "actual_residual_sd": 3.3617851188208783, "predicted_residual_sd": 1.9845531531850586, "actual_on_predicted_slope": 0.9230348972004506, "actual_on_predicted_intercept": -0.3042984668222081, "residual_spearman": 0.5524329567953647, "elite_bias": 3.029447176003838, "poor_bias": -2.7058553133990597, "p90_abs_error": 4.678393933787078}`

A narrow predicted spread is not sufficient evidence that mean predictions should be expanded: conditional means can have lower variance than realized outcomes. Estimate gamma from OOF cross-products, NOT actual_SD/predicted_SD; do not force gamma>1 or match outcome variance.

The pooled intercept is modest, but block offsets change sign and range from about-1.107 to+0.826 DRtg. A constant intercept is therefore not a supported era-stable remedy. Symmetric scaling is numerically stable but improves poor-defense bias in zero of five honest prequential blocks. The worse-defense asymmetric multiplier is unstable. Training does not support the optional piecewise expansion; no additional nonlinear family is tried.

## Calibration Stability

Per-fold multipliers below are descriptive fits to each held block's genuine OOF pairs. Honest prequential calibration performance instead fits a calibrator only to EARLIER OOF blocks and scores the next block; the first OOF block is meta-calibration warmup. Final production parameters pool all training OOF rows only.

| candidate | kind | fitting_rows | fitting_max_year | gamma | gamma_better | gamma_worse | n_better | n_worse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A_identity | A_identity | 120 | 1990 | 1.000 | null | null | null | null |
| A_identity | A_identity | 135 | 1995 | 1.000 | null | null | null | null |
| A_identity | A_identity | 145 | 2000 | 1.000 | null | null | null | null |
| A_identity | A_identity | 146 | 2005 | 1.000 | null | null | null | null |
| A_identity | A_identity | 150 | 2010 | 1.000 | null | null | null | null |
| A_identity | A_identity | 120 | 2014 | 1.000 | null | null | null | null |
| B_symmetric | B_symmetric | 120 | 1990 | 0.667 | null | null | null | null |
| B_symmetric | B_symmetric | 135 | 1995 | 1.000 | null | null | null | null |
| B_symmetric | B_symmetric | 145 | 2000 | 1.099 | null | null | null | null |
| B_symmetric | B_symmetric | 146 | 2005 | 0.823 | null | null | null | null |
| B_symmetric | B_symmetric | 150 | 2010 | 1.041 | null | null | null | null |
| B_symmetric | B_symmetric | 120 | 2014 | 1.103 | null | null | null | null |
| C_asymmetric | C_asymmetric | 120 | 1990 | null | 0.965 | 0.288 | 61.000 | 59.000 |
| C_asymmetric | C_asymmetric | 135 | 1995 | null | 1.045 | 0.880 | 81.000 | 54.000 |
| C_asymmetric | C_asymmetric | 145 | 2000 | null | 1.297 | 0.512 | 81.000 | 64.000 |
| C_asymmetric | C_asymmetric | 146 | 2005 | null | 0.775 | 1.341 | 100.000 | 46.000 |
| C_asymmetric | C_asymmetric | 150 | 2010 | null | 1.062 | 0.990 | 83.000 | 67.000 |
| C_asymmetric | C_asymmetric | 120 | 2014 | null | 1.332 | 0.556 | 66.000 | 54.000 |

| candidate | parameter | mean | sd | min | max | coefficient_of_variation |
| --- | --- | --- | --- | --- | --- | --- |
| A_identity | gamma | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 |
| B_symmetric | gamma | 0.956 | 0.175 | 0.667 | 1.103 | 0.183 |
| C_asymmetric | gamma_better | 1.079 | 0.209 | 0.775 | 1.332 | 0.194 |
| C_asymmetric | gamma_worse | 0.761 | 0.382 | 0.288 | 1.341 | 0.502 |

Stable eligibility: multipliers in[.25,2.5], fold coefficient of variation<=.35; in at least4/5 prequential blocks MAE must not regress>.15 or elite bias>.5 versus identity, and mean MAE regression must be<=.05. A nonidentity correction must not worsen poor bias>.25 in>=4/5 blocks and must improve it>.05 in>=2 blocks, rejecting a one-era tail fix. No coefficients are clipped to pass stability. All candidates/parameters/eligibility are fingerprinted before validation.

| candidate | fold | n | mae | rmse | r2 | actual_residual_sd | predicted_residual_sd | actual_on_predicted_slope | actual_on_predicted_intercept | residual_spearman | elite_bias | poor_bias | p90_abs_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A_identity | 2 | 135 | 1.916 | 2.457 | 0.294 | 3.124 | 1.976 | 0.976 | -0.196 | 0.655 | 2.526 | -1.989 | 4.080 |
| A_identity | 3 | 145 | 2.711 | 3.373 | 0.258 | 3.769 | 2.075 | 0.955 | -1.107 | 0.534 | 3.875 | -2.699 | 5.853 |
| A_identity | 4 | 146 | 2.442 | 3.059 | 0.151 | 3.530 | 1.886 | 1.019 | 0.826 | 0.524 | 1.973 | -3.525 | 5.043 |
| A_identity | 5 | 150 | 1.977 | 2.542 | 0.406 | 3.275 | 1.993 | 1.032 | -0.080 | 0.632 | 2.707 | -2.997 | 4.348 |
| A_identity | 6 | 120 | 2.339 | 2.927 | 0.218 | 3.519 | 1.954 | 1.036 | -0.571 | 0.554 | 4.152 | -2.139 | 4.757 |
| B_symmetric | 2 | 135 | 1.982 | 2.549 | 0.240 | 3.124 | 1.319 | 1.462 | -0.196 | 0.655 | 3.398 | -2.444 | 4.029 |
| B_symmetric | 3 | 145 | 2.747 | 3.407 | 0.244 | 3.769 | 1.777 | 1.116 | -1.107 | 0.534 | 4.217 | -2.817 | 5.859 |
| B_symmetric | 4 | 146 | 2.436 | 3.047 | 0.157 | 3.530 | 1.801 | 1.067 | 0.826 | 0.524 | 2.117 | -3.540 | 4.994 |
| B_symmetric | 5 | 150 | 1.992 | 2.554 | 0.401 | 3.275 | 1.824 | 1.128 | -0.080 | 0.632 | 2.942 | -3.100 | 4.297 |
| B_symmetric | 6 | 120 | 2.347 | 2.937 | 0.213 | 3.519 | 1.839 | 1.100 | -0.571 | 0.554 | 4.288 | -2.211 | 4.742 |
| C_asymmetric | 2 | 135 | 1.939 | 2.536 | 0.248 | 3.124 | 1.468 | 1.269 | 0.404 | 0.655 | 2.619 | -3.014 | 4.422 |
| C_asymmetric | 3 | 145 | 2.676 | 3.328 | 0.278 | 3.769 | 1.791 | 1.081 | -0.747 | 0.534 | 3.798 | -3.254 | 5.535 |
| C_asymmetric | 4 | 146 | 2.491 | 3.152 | 0.098 | 3.530 | 1.931 | 1.002 | 1.133 | 0.524 | 1.527 | -3.856 | 5.443 |
| C_asymmetric | 5 | 150 | 1.989 | 2.574 | 0.391 | 3.275 | 1.732 | 1.178 | 0.251 | 0.632 | 2.687 | -3.548 | 4.338 |
| C_asymmetric | 6 | 120 | 2.317 | 2.886 | 0.240 | 3.519 | 1.768 | 1.163 | -0.296 | 0.554 | 4.102 | -2.517 | 4.756 |

## Final Defensive Modeling Decision

Existing validation evaluated exactly once after parameter/protocol freeze; no retuning. All original hard gates and BOTH original/ranked bucket views are retained. Selection order: passing gates, training stability, simpler family, then validation error. No frozen test run.

| candidate | mae | rmse | r2 | stable | accepted |
| --- | --- | --- | --- | --- | --- |
| A_identity | 2.125 | 2.595 | 0.331 | True | False |
| B_symmetric | 2.117 | 2.584 | 0.336 | False | False |
| C_asymmetric | 2.178 | 2.635 | 0.310 | False | False |

Selected calibration: `{"kind": "A_identity", "fitting_rows": 816, "fitting_max_year": 2014, "gamma": 1.0}`

Validation buckets:

| candidate | view | bucket | n | mae | bias | median_bias | p90_abs_error | actual_residual | predicted_residual |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A_identity | legacy | elite | 20 | 2.573 | 1.935 | 1.889 | 4.387 | -4.205 | -2.270 |
| A_identity | legacy | above_average | 34 | 1.683 | 1.065 | 1.193 | 3.109 | -2.163 | -1.098 |
| A_identity | legacy | average | 39 | 1.993 | -0.523 | -0.841 | 3.513 | -0.012 | -0.535 |
| A_identity | legacy | below_average | 38 | 1.957 | -1.726 | -1.426 | 4.033 | 1.845 | 0.119 |
| A_identity | legacy | poor | 19 | 3.056 | -3.056 | -2.995 | 4.807 | 4.356 | 1.300 |
| A_identity | ranking | elite | 30 | 2.315 | 1.665 | 1.875 | 4.143 | -3.693 | -2.028 |
| A_identity | ranking | above_average | 29 | 1.822 | 0.752 | 1.163 | 3.486 | -1.711 | -0.959 |
| A_identity | ranking | average | 30 | 1.792 | -0.471 | -0.757 | 3.402 | -0.017 | -0.488 |
| A_identity | ranking | below_average | 30 | 2.060 | -1.527 | -1.227 | 3.938 | 1.610 | 0.083 |
| A_identity | ranking | poor | 31 | 2.612 | -2.612 | -2.829 | 4.776 | 3.465 | 0.852 |
| B_symmetric | legacy | elite | 20 | 2.581 | 2.017 | 1.981 | 4.493 | -4.205 | -2.188 |
| B_symmetric | legacy | above_average | 34 | 1.670 | 1.104 | 1.182 | 3.152 | -2.163 | -1.058 |
| B_symmetric | legacy | average | 39 | 1.956 | -0.503 | -0.785 | 3.432 | -0.012 | -0.515 |
| B_symmetric | legacy | below_average | 38 | 1.945 | -1.731 | -1.419 | 3.960 | 1.845 | 0.115 |
| B_symmetric | legacy | poor | 19 | 3.103 | -3.103 | -3.042 | 4.808 | 4.356 | 1.253 |
| B_symmetric | ranking | elite | 30 | 2.319 | 1.738 | 1.910 | 4.168 | -3.693 | -1.954 |
| B_symmetric | ranking | above_average | 29 | 1.795 | 0.787 | 1.117 | 3.411 | -1.711 | -0.924 |
| B_symmetric | ranking | average | 30 | 1.758 | -0.453 | -0.748 | 3.335 | -0.017 | -0.470 |
| B_symmetric | ranking | below_average | 30 | 2.041 | -1.530 | -1.267 | 3.872 | 1.610 | 0.080 |
| B_symmetric | ranking | poor | 31 | 2.643 | -2.643 | -2.823 | 4.772 | 3.465 | 0.821 |
| C_asymmetric | legacy | elite | 20 | 2.503 | 1.742 | 1.739 | 4.286 | -4.205 | -2.463 |
| C_asymmetric | legacy | above_average | 34 | 1.640 | 0.931 | 1.074 | 3.049 | -2.163 | -1.231 |
| C_asymmetric | legacy | average | 39 | 1.966 | -0.712 | -0.944 | 3.641 | -0.012 | -0.724 |
| C_asymmetric | legacy | below_average | 38 | 2.061 | -1.939 | -1.623 | 4.126 | 1.845 | -0.093 |
| C_asymmetric | legacy | poor | 19 | 3.464 | -3.464 | -3.557 | 4.849 | 4.356 | 0.892 |
| C_asymmetric | ranking | elite | 30 | 2.271 | 1.505 | 1.739 | 4.026 | -3.693 | -2.187 |
| C_asymmetric | ranking | above_average | 29 | 1.759 | 0.581 | 0.968 | 3.159 | -1.711 | -1.130 |
| C_asymmetric | ranking | average | 30 | 1.755 | -0.653 | -0.928 | 3.511 | -0.017 | -0.670 |
| C_asymmetric | ranking | below_average | 30 | 2.134 | -1.745 | -1.567 | 4.036 | 1.610 | -0.135 |
| C_asymmetric | ranking | poor | 31 | 2.930 | -2.930 | -3.011 | 4.784 | 3.465 | 0.535 |

Unchanged limits: `{"overall_mae_max": 2.2896266864236936, "anchor_validation_mae": 2.410133354130204, "elite_poor_mae_max": 5.0, "worst_bucket_abs_bias_max": 3.0, "abs_error_p90_max": 5.5, "subera_mae_over_anchor_max": 0.15, "box_heavy_abs_bias_max": 2.0, "rationale": "Require >=5% validation MAE improvement on league anchor, <=5 extreme-bucket MAE, <=3 bucket bias, <=5.5 p90 error, <=0.15 subera regression, <=2 box-heavy bias. Practical bounds declared before candidate evaluation; no final-test values used."}`

| candidate | view | overall | extremes | bucket_bias | p90 | suberas | box_heavy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A_identity | legacy | True | True | False | True | True | True |
| A_identity | ranking | True | True | True | True | True | True |
| B_symmetric | legacy | True | True | False | True | True | True |
| B_symmetric | ranking | True | True | True | True | True | True |
| C_asymmetric | legacy | True | True | False | True | True | True |
| C_asymmetric | ranking | True | True | True | True | True | True |

One final product decision, not another optimization loop: **B. DEFENSE REMAINS IMPERFECT BUT ACCEPTABLE FOR UNCERTAINTY-AWARE V1**. Product rule B is an explicit uncertainty-aware V1 exception under the user's stop rule, NOT a changed hard gate: at most.25 DRtg beyond the3.0 original bucket-bias limit, all other hard gates and ranked gates pass, validation residual Spearman>=.40, positive ranking in>=5/6 training folds, stable selected formulation. Every actual hard-gate failure remains labeled FAIL.

Ironman diagnostic: `{"base_drtg": 99.37391626755516, "base_prediction_reproduced": true, "raw_calibration_diagnostic": 99.37391626755516, "calibrated_drtg": 99.37391626755516, "league_anchor": 107.20165285182661, "nominal_multiplier": 1.0, "effective_multiplier": 1.0, "continuity": 0.0, "retained_defensive_context": 0.0, "unsupported": true, "guard_applied": true, "uncertainty": "HIGH", "base_uncertainty": {"category": "HIGH", "high_flags": {"low_continuity": true, "weak_availability": false, "diverse_contexts": false, "extrapolation": true}, "continuity": 0.0, "missing_share": 0.0, "source_context_sd": 1.2931461134334779, "source_diversity": 0.7999999999999999, "coverage": {"risk": "high", "rms_outside_domain_distance": 2.193093485818854, "maximum_exceed_iqr_units": 7.172085558643423, "fraction_outside": 0.3333333333333333, "features": [{"feature": "w_defensive_box_plus_minus", "value": 2.3400000000000003, "status": "outside", "exceed_iqr_units": 1.3893326670507007, "min": -1.2332222052316886, "max": 1.44005793111239, "p01": -1.05510781021616, "p99": 1.084550464598829, "scale": 0.647751319918233}, {"feature": "w_steals_per_75", "value": 1.589494678276071, "status": "inside", "exceed_iqr_units": 0.0, "min": 0.9046779604265548, "max": 1.9594315968031453, "p01": 0.9796842931501458, "p99": 1.6627458908974788, "scale": 0.18760923277181285}, {"feature": "w_blocks_per_75", "value": 1.6103301901598672, "status": "outside", "exceed_iqr_units": 0.9757638486393071, "min": 0.335969874784325, "max": 1.41412634205397, "p01": 0.412775951925196, "p99": 1.211829814346151, "scale": 0.20107718520162587}, {"feature": "w_defensive_rebounds_per_75", "value": 7.098033251001157, "status": "outside", "exceed_iqr_units": 1.8424811646109094, "min": 3.4656440880722736, "max": 6.00425436563547, "p01": 3.7046289253469293, "p99": 5.564907890268214, "scale": 0.5936445410537852}, {"feature": "w_age", "value": 27.0, "status": "inside", "exceed_iqr_units": 0.0, "min": 23.720811448181948, "max": 32.51990553949584, "p01": 24.87629859968812, "p99": 31.385420195955653, "scale": 2.02036850856544}, {"feature": "center_minutes_share", "value": 0.2, "status": "inside", "exceed_iqr_units": 0.0, "min": 0.0, "max": 0.4937682868701394, "p01": 0.02313958403493424, "p99": 0.33406277891866454, "scale": 0.08715977386722085}, {"feature": "rotation_hhi", "value": 0.20000000000000004, "status": "outside", "exceed_iqr_units": 7.172085558643423, "min": 0.0567274021943789, "max": 0.1147278926097213, "p01": 0.07020214196832235, "p99": 0.11017444812998578, "scale": 0.01188944369012905}, {"feature": "retained_defensive_context", "value": 0.0, "status": "inside", "exceed_iqr_units": 0.0, "min": -7.698405857361826, "max": 6.258635952464652, "p01": -5.774970086962996, "p99": 3.822396125100745, "scale": 2.8319957131836584}, {"feature": "returning_minutes_share", "value": 0.0, "status": "near_edge", "exceed_iqr_units": 0.0, "min": 0.0, "max": 0.9851474117884347, "p01": 0.23389610172788974, "p99": 0.9330851017955102, "scale": 0.2042365061058603}, {"feature": "defensive_missing_share", "value": 0.0, "status": "inside", "exceed_iqr_units": 0.0, "min": 0.0, "max": 0.3062645473184251, "p01": 0.0, "p99": 0.18883963968576814, "scale": 0.04688145464856368}, {"feature": "source_context_sd", "value": 1.2931461134334779, "status": "inside", "exceed_iqr_units": 0.0, "min": 0.030195935777832916, "max": 5.369359484800911, "p01": 0.24022694405205197, "p99": 4.389397975592513, "scale": 1.188524139599855}, {"feature": "source_diversity", "value": 0.7999999999999999, "status": "inside", "exceed_iqr_units": 0.0, "min": 0.029263977669965913, "max": 0.9401178721074527, "p01": 0.1418121956270078, "p99": 0.8575875318111651, "scale": 0.24586238089399925}], "interpretation": "Feature coverage/uncertainty only, not quality or probability; correlated features and unobserved lineup interactions limit interpretation."}, "interpretation": "Uncalibrated uncertainty/coverage indicator, not defensive quality or a probability interval"}, "uncertainty_warning": "Unsupported ironman feature vector: base value preserved; spread calibration is not validated here", "offense_pace_unchanged": {"ortg": 122.62733440950142, "pace": 92.99036643740251}, "semantics": "Curry2015-16/Jordan1990-91/LeBron2012-13/Duncan2002-03/Shaq1999-00; 48 minutes each; diagnostic, not precise truth"}`

For unsupported synthetic rosters, preserve the base projection rather than amplify an extrapolated residual. Record raw unguarded calibration only as a diagnostic, set effective multiplier1, add an explicit HIGH uncertainty/coverage warning. Supported rosters use the learned function normally; no universal min/max clamp. Continuity remains0 and no actual team outcome/bucket is needed.

Integrity: `{"protected_hashes_match": true, "offense_pace_usage_rotation_unchanged": true, "official_test_untouched": true, "new_final_test_evaluations": 0, "frozen_offense_pace_values": {"ortg": 122.62733440950142, "pace": 92.99036643740251}}`. Official frozen test remains the previous model's benchmark, not evidence for a newly fitted calibration. All prior results preserved.

Next product step: for A, game-log calibration; for B, an explicitly approved uncertainty-aware V1 scope before game-level calibration; for C, restrict defensive claims rather than continue the same validation search. No game logs, UI or application changes in this task.
