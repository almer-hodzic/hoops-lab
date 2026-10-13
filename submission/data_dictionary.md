# Data Dictionary

Primary dataset: `data/processed/shots_enriched.parquet`.

The project predicts the pre-shot probability that an NBA field-goal attempt will be made. The target is `shot_made`; all production features must be known before the shot result.

## 1. Identification / Context

| Field | Type | Description | Example | Used by production model |
| --- | --- | --- | --- | --- |
| `season` | string | NBA season label. | `2025-26` | Yes |
| `season_type` | string | Regular season or playoffs. | `regular_season` | Used for splits/evaluation |
| `dataset_split` | string | Chronological modeling split. | `test` | No, split control only |
| `game_id` | string | NBA game identifier. | `22500001` | No |
| `event_id` | string/int | Shot event identifier within game. | `18` | No |
| `game_date` | date/string | Date of the game. | `2025-10-21` | Split/order only |
| `player_id`, `player_name` | id/string | Shooter identity. | `201939`, Stephen Curry | ID used only in comparison/raw-ID model; profile features preferred |
| `team_id`, `team_name` | id/string | Shooter team. | Golden State Warriors | No |
| `opponent_team_id` | id | Opponent team identifier. | `1610612747` | No |
| `home_team_id`, `away_team_id`, `is_home` | id/bool | Home/away context. | `is_home=True` | `is_home` yes |

## 2. Shot Geometry

| Field | Type | Description | Example | Used |
| --- | --- | --- | --- | --- |
| `shot_distance` | integer | Canonical NBA shot distance in feet. | `26` | Yes |
| `loc_x`, `loc_y` | integer | Shot coordinates in NBA court units. | `80`, `250` | Yes |
| `shot_value` | integer | Point value of the attempt. | `2` or `3` | Yes |

## 3. Game Context

| Field | Type | Description | Example | Used |
| --- | --- | --- | --- | --- |
| `period` | integer | Quarter or overtime period. | `4` | Yes |
| `clock` | string | Original clock display. | `04:00` | No; converted field used |
| `seconds_remaining_period` | integer | Seconds left in the current period. | `240` | Yes |
| `is_overtime` | bool | Whether period is overtime. | `False` | Yes |
| `home_score_before`, `away_score_before` | integer | Reconstructed pre-shot score. | `98`, `100` | No direct model input |
| `shooter_score_before`, `opponent_score_before` | integer | Pre-shot score from shooter perspective. | `98`, `100` | No direct model input |
| `score_diff_before` | integer | Shooter team score minus opponent score before shot. | `-2` | Yes |
| `abs_score_diff_before` | integer | Absolute pre-shot score differential. | `2` | Yes |
| `game_state_before` | string | Leading, trailing, or tied. | `trailing` | Yes |
| `is_clutch` | bool | Q4/OT, <=5:00, abs score diff <=5. | `True` | Evaluation/analysis, not final model input |
| `clutch_time_bucket` | string | Clutch time grouping. | `0-1 min` | Analysis only |

## 4. Zone / Category Fields

| Field | Type | Description | Example | Used |
| --- | --- | --- | --- | --- |
| `shot_type` | string | 2PT or 3PT field goal. | `3PT Field Goal` | Yes |
| `action_type` | string | NBA action label. | `Jump Shot` | Yes |
| `shot_zone_basic` | string | NBA zone. | `Above the Break 3` | Yes |
| `shot_zone_area` | string | Court side/area. | `Center(C)` | Yes |
| `shot_zone_range` | string | Distance range label. | `24+ ft.` | Yes |

## 5. Target Variable

| Field | Type | Description | Used |
| --- | --- | --- | --- |
| `shot_made` | bool | Target: whether the shot was made. | Target only |
| `points_scored` | integer | Outcome points. | Not used as pre-shot feature |

## 6. Leakage-Safe Historical Shooter Features

These are engineered during modeling from prior shots only, before the current attempt:

| Feature family | Description | Used by final production model |
| --- | --- | --- |
| `player_prior_fga`, `player_prior_fg_pct`, shrunk FG% | Shooter career-to-date attempts and field-goal rate before this shot. | Yes |
| `player_prior_2pa`, `player_prior_2p_pct_shrunk` | Prior 2PT profile. | Yes |
| `player_prior_3pa`, `player_prior_3p_pct_shrunk` | Prior 3PT profile. | Yes |
| `player_zone_prior_attempts`, `player_zone_prior_fg_pct_shrunk` | Prior performance in the same shot zone. | Yes |
| `player_distance_prior_attempts`, `player_distance_prior_fg_pct_shrunk` | Prior performance in the same distance bucket. | Yes |
| `player_season_prior_*` | Season-to-date shooter profile. | Yes |
| `player_previous_season_*` | Previous-season shooter profile. | Yes |
| `player_cold_start` | Indicates limited/no prior NBA shot history. | Yes |

## 7. Derived Features

| Field/family | Description | Used |
| --- | --- | --- |
| `abs_score_diff_before` | Absolute value of pre-shot score differential. | Yes |
| `game_state_before` | Derived from score differential. | Yes |
| `is_overtime` | Derived from period. | Yes |
| `is_clutch` | Official project clutch definition. | Analysis/evaluation |
| `shot_xpps` | `p_make * shot_value`, produced by Decision Policy. | Decision layer |
| `decision_margin` | `shot_xpps - replacement_xpps`. | Decision layer |

## 8. Fields Excluded From Modeling And Why

| Field/family | Reason |
| --- | --- |
| `shot_made`, `points_scored` | Post-outcome leakage; target/result only. |
| `home_score_before`, `away_score_before` | Raw score components are summarized by pre-shot differential/context. |
| `game_id`, `event_id`, `game_date` | Identifiers/order controls, not stable basketball predictors. |
| Team names/IDs | Avoid overfitting and roster/context instability in the production model. |
| Raw player ID alone | Tested as comparison; historical profile is more interpretable and cold-start safer. |
| Defensive tracking fields | 2014/15 research extension only; not part of production model. |
