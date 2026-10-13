# 2014/15 Shot Logs Schema Mapping

- `GAME_ID` -> `game_id`
- `MATCHUP` -> `matchup/opponent context`
- `LOCATION` -> `location`
- `FINAL_MARGIN` -> `score_margin`
- `SHOT_NUMBER` -> `shot_number`
- `PERIOD` -> `period`
- `GAME_CLOCK` -> `game_clock`
- `SHOT_CLOCK` -> `shot_clock`
- `DRIBBLES` -> `dribbles`
- `TOUCH_TIME` -> `touch_time`
- `SHOT_DIST` -> `shot_distance`
- `PTS_TYPE` -> `shot_value and shot_type`
- `SHOT_RESULT` -> `shot_result`
- `FGM` -> `made`
- `PTS` -> `points`
- `player_name` -> `shooter_name`
- `player_id` -> `shooter_id`
- `CLOSEST_DEFENDER` -> `closest_defender_name`
- `CLOSEST_DEFENDER_PLAYER_ID` -> `closest_defender_id`
- `CLOSE_DEF_DIST` -> `closest_defender_distance`

Unknown or unused source columns are retained in raw data, not silently interpreted.