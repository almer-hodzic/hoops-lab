#!/usr/bin/env python3
"""Validate the historical player-season feasibility prototype outputs."""

from __future__ import annotations

import json
from pathlib import Path
import unicodedata

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "historical" / "processed"
METADATA_DIR = ROOT / "data" / "historical" / "metadata"


EXPECTED_PLAYERS = {
    ("Kareem Abdul-Jabbar", "1976-77"),
    ("Larry Bird", "1985-86"),
    ("Michael Jordan", "1990-91"),
    ("Shaquille O'Neal", "1999-00"),
    ("LeBron James", "2012-13"),
    ("Stephen Curry", "2015-16"),
    ("Nikola Jokic", "2022-23"),
}

EXPECTED_TEAMS = {
    ("Boston Celtics", "1985-86"),
    ("Chicago Bulls", "1995-96"),
    ("Los Angeles Lakers", "2000-01"),
    ("Miami Heat", "2012-13"),
    ("Golden State Warriors", "2016-17"),
}


def ascii_fold(value: object) -> str:
    return "".join(
        char for char in unicodedata.normalize("NFKD", str(value)) if not unicodedata.combining(char)
    )


def fail(message: str) -> None:
    raise SystemExit(message)


def main() -> None:
    players_path = PROCESSED_DIR / "player_season_prototype.csv"
    teams_path = PROCESSED_DIR / "team_season_prototype.csv"
    manifest_path = METADATA_DIR / "historical_player_data_manifest.json"
    for path in [players_path, teams_path, manifest_path]:
        if not path.exists():
            fail(f"Missing expected output: {path}")

    players = pd.read_csv(players_path)
    teams = pd.read_csv(teams_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    actual_players = set(zip(players["player_name"].map(ascii_fold), players["season"]))
    missing_players = EXPECTED_PLAYERS - actual_players
    if missing_players:
        fail(f"Missing representative players: {sorted(missing_players)}")

    actual_teams = set(zip(teams["team_name"], teams["season"]))
    missing_teams = EXPECTED_TEAMS - actual_teams
    if missing_teams:
        fail(f"Missing representative teams: {sorted(missing_teams)}")

    if players.duplicated(["player_name", "season"]).any():
        fail("Duplicate player-season rows found.")
    if teams.duplicated(["team_name", "season"]).any():
        fail("Duplicate team-season rows found.")

    loaded_players = players[players["source_status"] == "loaded"]
    for col in ["games", "minutes", "points", "points_per_75", "true_shooting_pct"]:
        if loaded_players[col].isna().any():
            fail(f"Loaded players have null required field: {col}")
    if not loaded_players["true_shooting_pct"].between(0, 1).all():
        fail("TS% outside 0-1 range.")
    if (loaded_players["points_per_75"] <= 0).any():
        fail("Non-positive points_per_75 found.")
    if manifest.get("validation_issues"):
        fail(f"Manifest validation issues: {manifest['validation_issues']}")

    print(
        json.dumps(
            {
                "ok": True,
                "player_rows": int(len(players)),
                "team_rows": int(len(teams)),
                "outputs": manifest.get("outputs", []),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
