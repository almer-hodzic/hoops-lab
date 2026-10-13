# Data Pipeline Plan

1. Keep all downloaded archives in `data/raw/` and treat them as immutable.
2. Add missing regular-season Shot Detail and PBP V3 archives using the same folder convention.
3. Build a canonical interim table by reading archives, normalizing column names, and retaining source file lineage.
4. Match shots to PBP with `GAME_ID + GAME_EVENT_ID/actionNumber`, then verify period, clock, player, team, and result.
5. Derive pre-shot home/away scores from ordered PBP events, not directly from blank miss rows.
6. Compute official clutch flags from pre-shot score, period, and remaining clock.
7. Only after this audit is accepted, generate processed modeling datasets under `data/processed/`.

Normalized schema proposal: `season`, `season_type`, `game_id`, `event_id`, `period`, `clock_seconds_remaining`, `game_date`, `player_id`, `player_name`, `team_id`, `team_tricode`, `is_home`, `shot_made`, `shot_value`, `action_type`, `shot_zone_basic`, `shot_zone_area`, `shot_zone_range`, `shot_distance`, `loc_x`, `loc_y`, `home_score_before`, `away_score_before`, `shooter_score_diff_before`, `is_clutch`.
