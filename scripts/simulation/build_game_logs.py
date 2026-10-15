"""Canonical regular-season games and strictly prior-date team strength inputs."""
from __future__ import annotations

import json
import sys
from itertools import groupby
from pathlib import Path

import numpy as np
import pandas as pd

from acquire_game_logs import MANIFEST, RAW, ROOT, acquire, fingerprint

sys.path.insert(0, str(ROOT / "scripts/historical"))
from build_full_historical_dataset import team_identity_from_name

OUTPUT = ROOT / "data/simulation/game_logs_processed.csv"
QUALITY = ROOT / "data/simulation/game_log_quality.json"
TEAM_PRIOR_GAMES = 20
LEAGUE_PRIOR_GAMES = 500


def season_end(game_id):
    start = (int(game_id) // 100000) % 100
    return (1901 if start >= 79 else 2001) + start


def identity(city, name, year):
    city = "Los Angeles" if city == "LA" else city
    if name == "Hornets" and year in (2006, 2007):
        city = "New Orleans/Oklahoma City"
    full = city + " " + name
    if full == "Seattle Supersonics":
        full = "Seattle SuperSonics"
    return team_identity_from_name(full, year)


def team_table():
    base = pd.read_csv(RAW / "TeamStatistics.csv", low_memory=False)
    ext = pd.read_csv(RAW / "TeamStatisticsExtended.csv", low_memory=False,
                      usecols=["gameId", "teamId", "possessions", "pace"])
    if base.duplicated(["gameId", "teamId"]).any() or ext.duplicated(["gameId", "teamId"]).any():
        raise ValueError("Duplicate raw team box score")
    base = base.merge(ext, on=["gameId", "teamId"], how="left", validate="one_to_one")
    return base


def canonical_games(games, boxes):
    games = games.copy()
    games["game_id"] = games.gameId.astype(str).str.zfill(10)
    games["season_end_year"] = games.game_id.map(season_end)
    games = games[games.game_id.str.startswith("002") & games.season_end_year.between(1980, 2025)].copy()
    if games.gameId.duplicated().any():
        raise ValueError("Duplicate game ID")
    for side in ("home", "away"):
        table = boxes.copy()
        table = table.rename(columns={col: side + "_" + col for col in table if col != "gameId"})
        games = games.merge(table, left_on=["gameId", side + "teamId"],
                            right_on=["gameId", side + "_teamId"], how="left", validate="one_to_one")
    records, rejected = [], []
    for row in games.to_dict("records"):
        gid, year = row["game_id"], row["season_end_year"]
        date = row["gameDateTimeEst"][:10]
        record = dict(game_id=gid, season=f"{year-1}-{str(year)[-2:]}", season_end_year=year, date=date,
                      season_type="regular_season", source_version=515,
                      neutral_site=bool("2020-07-30" <= date <= "2020-08-14"),
                      neutral_provenance="2020 Orlando restart dates; other neutral venues not fully identified")
        issue = None
        for side in ("home", "away"):
            official_id = row[side + "teamId"]
            box_id = row.get(side + "_teamId")
            score = row[side + "Score"]
            if pd.isna(box_id) or official_id != box_id or score != row.get(side + "_teamScore"):
                issue = "missing/mismatched team box score"
                break
            mapped = identity(row[side + "teamCity"], row[side + "teamName"], year)
            record[side + "_team_id"] = mapped["team_id"]
            record[side + "_franchise_id"] = mapped["franchise_id"]
            record[side + "_source_team_id"] = int(official_id)
            record[side + "_points"] = int(score)
            minutes = row[side + "_numMinutes"] / 5.
            quarters = [row[side + f"_q{q}Points"] for q in range(1, 5)]
            complete_quarters = all(pd.notna(v) for v in quarters)
            regulation = sum(quarters) if complete_quarters else score if minutes == 48 else np.nan
            record[side + "_regulation_points"] = regulation
            record[side + "_minutes"] = minutes
            formula_fields = [row[side + "_" + f] for f in
                              ("fieldGoalsAttempted", "freeThrowsAttempted", "reboundsOffensive", "turnovers")]
            valid_box = (all(pd.notna(v) for v in formula_fields) and
                         formula_fields[0] >= row[side + "_fieldGoalsMade"] > 0)
            estimate = (formula_fields[0] + .44 * formula_fields[1] - formula_fields[2] + formula_fields[3]
                        if valid_box else np.nan)
            reported = row[side + "_possessions"]
            record[side + "_possessions"] = reported if pd.notna(reported) and reported > 0 else estimate
            record[side + "_possession_method"] = ("source_reported" if pd.notna(reported) and reported > 0 else
                                                    "box_estimate_044" if valid_box else "unavailable_zero_coded_box")
        if issue:
            rejected.append(dict(game_id=gid, issue=issue))
            continue
        minutes = record["home_minutes"]
        minutes_valid = (pd.notna(minutes) and 48 <= minutes <= 78 and
                         abs((minutes-48)/5-round((minutes-48)/5)) < .02 and
                         abs(minutes-record["away_minutes"]) < .02)
        record["overtime_periods"] = int(round((minutes-48)/5)) if minutes_valid else np.nan
        record["overtime"] = bool(minutes > 48) if minutes_valid else None
        record["possessions"] = (record["home_possessions"] + record["away_possessions"]) / 2.
        record["pace"] = record["possessions"] * 48 / minutes if minutes_valid else np.nan
        record["margin"] = record["home_points"] - record["away_points"]
        if record["margin"] == 0 or min(record["home_points"], record["away_points"]) <= 0:
            rejected.append(dict(game_id=gid, issue="tied/zero final source score; not accepted as completed NBA game"))
            continue
        record["home_win"] = int(record["margin"] > 0)
        record["regulation_margin"] = record["home_regulation_points"] - record["away_regulation_points"]
        record["regulation_total"] = record["home_regulation_points"] + record["away_regulation_points"]
        regulation_valid = (0 < record["home_regulation_points"] <= record["home_points"] and
                            0 < record["away_regulation_points"] <= record["away_points"] and
                            (not minutes_valid or (record["regulation_margin"] == 0 if minutes > 48 else
                                                  record["regulation_margin"] == record["margin"])))
        record["eligible"] = bool(minutes_valid and regulation_valid and 60 <= record["pace"] <= 140 and record["margin"] != 0)
        record["split"] = "train" if year <= 2014 else "validation" if year <= 2019 else "test"
        for side in ("home", "away"):
            record[side + "_game_ortg"] = record[side + "_points"] / record["possessions"] * 100
        records.append(record)
    return pd.DataFrame(records).sort_values(["date", "game_id"]).reset_index(drop=True), rejected


def add_pregame_strength(games, team_prior=TEAM_PRIOR_GAMES, league_prior=LEAGUE_PRIOR_GAMES):
    """Update all games on a date together; no current-day result enters any feature."""
    games = games.copy()
    observations = []
    for side, other in [("home", "away"), ("away", "home")]:
        valid = games[games.eligible]
        observations.append(pd.DataFrame(dict(year=valid.season_end_year, franchise=valid[side + "_franchise_id"],
            points=valid[side + "_regulation_points"], allowed=valid[other + "_regulation_points"],
            possessions=valid.pace, games=1, date=valid.date)))
    historical = pd.concat(observations)
    prior_teams = historical.groupby(["year", "franchise"])[["points", "allowed", "possessions", "games"]].sum().to_dict("index")
    prior_leagues = historical.groupby("year")[["points", "allowed", "possessions", "games"]].sum().to_dict("index")
    prior_dates = historical.groupby(["year", "franchise"]).date.max().to_dict()
    league_dates = historical.groupby("year").date.max().to_dict()
    team_running, league_running, output = {}, {}, {}
    last_date = {}
    for date, day_iterator in groupby(games.sort_values(["date", "game_id"]).to_dict("records"), key=lambda r:r["date"]):
        day = list(day_iterator)
        for row in day:
            year = row["season_end_year"]
            league_before = prior_leagues.get(year - 1)
            if league_before is None:
                continue
            prior_pace = league_before["possessions"] / league_before["games"]
            prior_rating = 100 * league_before["points"] / league_before["possessions"]
            league = league_running.get(year, np.zeros(4))
            anchor = 100 * (league_prior * prior_pace * prior_rating / 100 + league[0]) / (league_prior * prior_pace + league[2])
            features = dict(league_ortg=anchor, league_pace=(league_prior * prior_pace + league[2]) / (league_prior + league[3]),
                            league_prior_season_end_year=year-1)
            for side in ("home", "away"):
                franchise = row[side + "_franchise_id"]
                prior = prior_teams.get((year-1, franchise), league_before)
                pace = prior["possessions"] / prior["games"]
                offense, defense = 100 * prior["points"] / prior["possessions"], 100 * prior["allowed"] / prior["possessions"]
                running = team_running.get((year, franchise), np.zeros(4))
                denom = team_prior * pace + running[2]
                features[side + "_ortg"] = 100 * (team_prior * pace * offense / 100 + running[0]) / denom
                features[side + "_drtg"] = 100 * (team_prior * pace * defense / 100 + running[1]) / denom
                features[side + "_pace"] = (team_prior * pace + running[2]) / (team_prior + running[3])
                features[side + "_prior_games"] = int(running[3])
                features[side + "_last_information_date"] = last_date.get(
                    (year, franchise), prior_dates.get((year-1, franchise), league_dates[year-1]))
            output[row["game_id"]] = features
        for row in day:
            if not row["eligible"]:
                continue
            year = row["season_end_year"]
            for side, other in [("home", "away"), ("away", "home")]:
                franchise = row[side + "_franchise_id"]
                vector = np.array([row[side + "_regulation_points"], row[other + "_regulation_points"], row["pace"], 1.])
                key = (year, franchise)
                team_running[key] = team_running.get(key, np.zeros(4)) + vector
                league_running[year] = league_running.get(year, np.zeros(4)) + vector
                last_date[key] = date
    features = pd.DataFrame.from_dict(output, orient="index").rename_axis("game_id").reset_index()
    return games.merge(features, on="game_id", how="left", validate="one_to_one")


def build():
    acquire(offline=True)
    games, rejected = canonical_games(pd.read_csv(RAW / "Games.csv", low_memory=False), team_table())
    games = add_pregame_strength(games)
    if games.game_id.duplicated().any():
        raise ValueError("Duplicate canonical game")
    for side in ("home", "away"):
        valid = games[games[side + "_last_information_date"].notna()]
        if not (valid[side + "_last_information_date"] < valid.date).all():
            raise ValueError("Non-pregame team information")
    teams = pd.read_csv(ROOT / "data/historical/processed/team_seasons.csv")
    expected = teams[teams.season_end_year <= 2025].copy()
    counts = pd.concat([games[["season_end_year", side + "_team_id"]].rename(columns={side + "_team_id": "team_id"})
                        for side in ("home", "away")]).groupby(["season_end_year", "team_id"]).size()
    expected["actual_games_in_source"] = [int(counts.get((r.season_end_year, r.team_id), 0)) for r in expected.itertuples()]
    mismatch = expected[expected.actual_games_in_source != expected.wins + expected.losses]
    games.to_csv(OUTPUT, index=False, float_format="%.8f")
    summary = dict(source_manifest_sha256=fingerprint(MANIFEST), canonical_sha256=fingerprint(OUTPUT),
                   games=len(games), start=games.season.iloc[0], end=games.season.iloc[-1],
                   excluded_after_2025="2025-26 snapshot not assumed complete; no outcome-based exclusion",
                   quarantined_games=rejected, season_counts=games.groupby("season").size().to_dict(),
                   eligible_by_split=games[games.eligible & games.home_ortg.notna() & ~games.neutral_site].groupby("split").size().to_dict(),
                   missingness=games.isna().sum().to_dict(), neutral_games=int(games.neutral_site.sum()),
                   possession_methods=games.home_possession_method.value_counts().to_dict(),
                   eligible_by_season=games.groupby("season").eligible.sum().astype(int).to_dict(),
                   team_season_count_mismatches=mismatch[["season", "team_name", "wins", "losses", "actual_games_in_source"]].to_dict("records"),
                   formula="possessions: mean of source team possessions, else FGA+0.44*FTA-ORB+TO; pace normalized by48/minutes",
                   strength="Prior-date cumulative regulation points/estimated regulation possessions;20-game prior-season franchise prior,500-team-game league prior; both fixed before modeling",
                   limitations=["Box possession formula is approximate, especially legacy eras", "Only Orlando restart neutral sites explicitly marked",
                                "Regulation possessions in OT games approximated from full-game pace", "No injury/rest/travel or timestamped roster inputs"])
    QUALITY.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k:summary[k] for k in ["games", "start", "end", "eligible_by_split", "possession_methods", "team_season_count_mismatches"]}, indent=2))


if __name__ == "__main__":
    build()
