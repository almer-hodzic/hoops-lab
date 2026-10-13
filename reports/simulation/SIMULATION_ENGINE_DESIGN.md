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
