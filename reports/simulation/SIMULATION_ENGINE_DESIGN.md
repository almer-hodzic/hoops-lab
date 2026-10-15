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
