#!/usr/bin/env python3
"""Validate the historical player-season feasibility prototype outputs."""

from __future__ import annotations

import json
import math
from pathlib import Path
import unicodedata

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "historical" / "processed"
METADATA_DIR = ROOT / "data" / "historical" / "metadata"


EXPECTED_PLAYERS = {
    ("Kareem Abdul-Jabbar", "1976-77"): "abdulka01",
    ("Larry Bird", "1985-86"): "birdla01",
    ("Michael Jordan", "1990-91"): "jordami01",
    ("Shaquille O'Neal", "1999-00"): "onealsh01",
    ("LeBron James", "2012-13"): "jamesle01",
    ("Stephen Curry", "2015-16"): "curryst01",
    ("Nikola Jokic", "2022-23"): "jokicni01",
}

EXPECTED_TEAMS = {
    ("Boston Celtics", "1985-86"): ("team:boston-celtics", "franchise:boston-celtics"),
    ("Chicago Bulls", "1995-96"): ("team:chicago-bulls", "franchise:chicago-bulls"),
    ("Los Angeles Lakers", "2000-01"): ("team:los-angeles-lakers", "franchise:lakers"),
    ("Miami Heat", "2012-13"): ("team:miami-heat", "franchise:miami-heat"),
    ("Golden State Warriors", "2016-17"): ("team:golden-state-warriors", "franchise:warriors"),
}

REQUIRED_PROVENANCE = [
    "player_efficiency_rating",
    "win_shares",
    "box_plus_minus",
    "usage_pct",
    "assist_pct",
    "turnover_pct",
    "total_rebound_pct",
    "true_shooting_pct",
    "team_pace",
    "league_pace",
    "league_ts_pct",
    "estimated_possessions_minutes_team_pace",
    "points_per_75",
]


def ascii_fold(value: object) -> str:
    return "".join(
        char for char in unicodedata.normalize("NFKD", str(value)) if not unicodedata.combining(char)
    )


def fail(message: str) -> None:
    raise SystemExit(message)


def close(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(float(a) - float(b)) <= tol


def main() -> None:
    players_path = PROCESSED_DIR / "player_season_prototype.csv"
    comparison_path = PROCESSED_DIR / "player_season_comparison.csv"
    teams_path = PROCESSED_DIR / "team_season_prototype.csv"
    manifest_path = METADATA_DIR / "historical_player_data_manifest.json"
    team_map_path = METADATA_DIR / "team_identity_map.json"
    for path in [players_path, comparison_path, teams_path, manifest_path, team_map_path]:
        if not path.exists():
            fail(f"Missing expected output: {path}")

    players = pd.read_csv(players_path)
    comparison = pd.read_csv(comparison_path)
    teams = pd.read_csv(teams_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    team_map = json.loads(team_map_path.read_text(encoding="utf-8"))

    if manifest.get("schema_version") != "historical_player_season_prototype_v2":
        fail("Unexpected schema version.")
    if manifest.get("validation_issues"):
        fail(f"Manifest validation issues: {manifest['validation_issues']}")
    for field in REQUIRED_PROVENANCE:
        if field not in manifest.get("field_provenance", {}):
            fail(f"Missing provenance metadata for {field}.")
    if not manifest.get("source_fetches"):
        fail("Manifest missing source fetch metadata.")

    actual_players = set(zip(players["player_name"].map(ascii_fold), players["season"]))
    expected_players = {(ascii_fold(name), season) for (name, season) in EXPECTED_PLAYERS}
    missing_players = expected_players - actual_players
    if missing_players:
        fail(f"Missing representative players: {sorted(missing_players)}")

    if players.duplicated(["player_id", "season"]).any():
        fail("Duplicate PlayerSeason rows found.")
    if "league_relative_scoring_per_75" in players.columns or "league_relative_scoring_per_75" in comparison.columns:
        fail("Invalid sample-relative league scoring field is still present.")

    for (name, season), source_id in EXPECTED_PLAYERS.items():
        row = players[(players["player_name"].map(ascii_fold) == ascii_fold(name)) & (players["season"] == season)]
        if row.empty:
            fail(f"Missing row for {name} {season}.")
        item = row.iloc[0]
        if item["source_player_id"] != source_id or item["player_id"] != f"br:{source_id}":
            fail(f"Unexpected source id for {name} {season}: {item['player_id']}")

    jokic = players[players["player_name"] == "Nikola Jokic"].iloc[0]
    if "Jokić" not in str(jokic["source_player_name"]):
        fail("Unicode source name regression for Nikola Jokic.")

    kareem = players[players["player_name"] == "Kareem Abdul-Jabbar"].iloc[0]
    if not pd.isna(kareem["three_point_attempt_rate"]):
        fail("Kareem 1976-77 3PAr should be null because 3PT was not tracked.")
    if not pd.isna(kareem["turnovers_per_75"]):
        fail("Kareem 1976-77 turnovers_per_75 should be null because TOV is unavailable.")

    for _, row in players.iterrows():
        if not 0 < row["true_shooting_pct"] < 1:
            fail(f"Invalid player TS% for {row['player_name']} {row['season']}.")
        if not 0 < row["league_ts_pct"] < 1:
            fail(f"Invalid league TS% for {row['season']}.")
        expected_poss = (row["minutes"] / 48.0) * row["team_pace"]
        if not close(row["estimated_possessions_minutes_team_pace"], expected_poss):
            fail(f"Possession estimate mismatch for {row['player_name']} {row['season']}.")
        expected_points_per_75 = row["points"] / row["estimated_possessions_minutes_team_pace"] * 75
        if not close(row["points_per_75"], expected_points_per_75):
            fail(f"points_per_75 mismatch for {row['player_name']} {row['season']}.")
        if row["source_status"] != "loaded":
            fail(f"Unexpected source status for {row['player_name']} {row['season']}.")

    actual_teams = set(zip(teams["team_name"], teams["season"]))
    missing_teams = set(EXPECTED_TEAMS) - actual_teams
    if missing_teams:
        fail(f"Missing representative teams: {sorted(missing_teams)}")
    if teams.duplicated(["team_id", "season"]).any():
        fail("Duplicate TeamSeason rows found.")
    for (name, season), (team_id, franchise_id) in EXPECTED_TEAMS.items():
        row = teams[(teams["team_name"] == name) & (teams["season"] == season)].iloc[0]
        if row["team_id"] != team_id or row["franchise_id"] != franchise_id:
            fail(f"Unexpected team/franchise id for {name} {season}.")
    for abbr in ["SEA", "OKC", "NJN", "BRK", "VAN", "MEM", "CHH", "NOH", "NOP"]:
        if abbr not in team_map:
            fail(f"Historical team identity map missing {abbr}.")

    league_context = manifest["league_context"]
    for season_end, context in league_context.items():
        pts = context["league_points"]
        fga = context["league_fga"]
        fta = context["league_fta"]
        expected_ts = pts / (2 * (fga + 0.44 * fta))
        if not close(context["league_ts_pct"], expected_ts):
            fail(f"League TS formula mismatch for {season_end}.")
        if context["league_context_provenance"] != "team_totals_and_team_misc_no_player_rows":
            fail(f"League context provenance is not protected for {season_end}.")

    print(
        json.dumps(
            {
                "ok": True,
                "player_rows": int(len(players)),
                "team_rows": int(len(teams)),
                "schema_version": manifest["schema_version"],
                "outputs": manifest.get("outputs", []),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
