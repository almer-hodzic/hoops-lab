#!/usr/bin/env python3
"""Validate the full HoopsLab historical NBA season datasets."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "historical" / "processed"
METADATA = ROOT / "data" / "historical" / "metadata"
START_YEAR = 1980
END_YEAR = 2026
EXPECTED_YEARS = set(range(START_YEAR, END_YEAR + 1))

STAR_CHECKS = [
    ("Kareem Abdul-Jabbar", 1980),
    ("Larry Bird", 1986),
    ("Michael Jordan", 1991),
    ("Shaquille O'Neal", 2000),
    ("Tim Duncan", 2003),
    ("LeBron James", 2013),
    ("Stephen Curry", 2016),
    ("Nikola Jokić", 2023),
]


def fail(errors: list[str], message: str) -> None:
    errors.append(message)


def close(a: float | None, b: float | None, tolerance: float = 1e-6) -> bool:
    if a is None or b is None or pd.isna(a) or pd.isna(b):
        return False
    return abs(float(a) - float(b)) <= tolerance


def require_columns(df: pd.DataFrame, name: str, columns: list[str], errors: list[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        fail(errors, f"{name} missing columns: {missing}")


def validate() -> int:
    errors: list[str] = []
    players = pd.read_csv(PROCESSED / "player_seasons.csv")
    splits = pd.read_csv(PROCESSED / "player_team_splits.csv")
    teams = pd.read_csv(PROCESSED / "team_seasons.csv")
    leagues = pd.read_csv(PROCESSED / "league_seasons.csv")
    manifest = json.loads((METADATA / "full_historical_ingestion_manifest.json").read_text(encoding="utf-8"))

    require_columns(
        players,
        "player_seasons",
        [
            "player_id",
            "player_name",
            "season",
            "season_end_year",
            "team",
            "team_id",
            "franchise_id",
            "minutes",
            "points",
            "estimated_possessions_minutes_team_pace",
            "points_per_75",
            "three_point_attempt_rate",
            "free_throw_rate",
            "relative_ts",
            "comparison_sample",
        ],
        errors,
    )
    require_columns(teams, "team_seasons", ["team_id", "franchise_id", "season_end_year", "wins", "losses", "pace"], errors)
    require_columns(leagues, "league_seasons", ["season_end_year", "league_ts_pct", "league_pace"], errors)

    for name, df in [("player_seasons", players), ("team_seasons", teams), ("league_seasons", leagues)]:
        years = set(df["season_end_year"].astype(int))
        if years != EXPECTED_YEARS:
            fail(errors, f"{name} season coverage mismatch: missing={sorted(EXPECTED_YEARS - years)}, extra={sorted(years - EXPECTED_YEARS)}")

    dup_players = players.duplicated(["player_id", "season_end_year"]).sum()
    if dup_players:
        fail(errors, f"player_seasons has {dup_players} duplicate player_id/season rows")
    dup_teams = teams.duplicated(["team_id", "season_end_year"]).sum()
    if dup_teams:
        fail(errors, f"team_seasons has {dup_teams} duplicate team_id/season rows")
    if len(leagues) != len(EXPECTED_YEARS):
        fail(errors, f"league_seasons row count {len(leagues)} != {len(EXPECTED_YEARS)}")

    summary_leaks = players["player_name"].astype(str).eq("League Average").sum() + teams["team_name"].astype(str).eq("League Average").sum()
    if summary_leaks:
        fail(errors, f"source summary rows leaked into outputs: {summary_leaks}")

    pct_cols = [
        "true_shooting_pct",
        "league_ts_pct",
        "three_point_attempt_rate",
        "free_throw_rate",
        "assist_pct",
        "turnover_pct",
        "total_rebound_pct",
        "usage_pct",
    ]
    for column in pct_cols:
        if column in players:
            values = players[column].dropna().astype(float)
            upper = 10 if column == "free_throw_rate" else 100 if column in {"assist_pct", "turnover_pct", "total_rebound_pct", "usage_pct"} else 3
            bad = values[(values < 0) | (values > upper)]
            if not bad.empty:
                fail(errors, f"{column} has {len(bad)} implausible percentage/rate values")

    nonnegative_cols = [
        "games",
        "minutes",
        "points",
        "field_goal_attempts",
        "free_throw_attempts",
        "total_rebounds",
        "assists",
        "steals",
        "blocks",
        "turnovers",
    ]
    for column in nonnegative_cols:
        bad = players[column].dropna().astype(float) < 0
        if bad.any():
            fail(errors, f"{column} has {int(bad.sum())} negative values")

    sample = players[players["estimated_possessions_minutes_team_pace"].notna()].head(1000)
    for _, row in sample.iterrows():
        poss = row["estimated_possessions_minutes_team_pace"]
        if poss and poss > 0:
            expected_points = row["points"] / poss * 75 if not pd.isna(row["points"]) else math.nan
            if not pd.isna(row["points_per_75"]) and not close(row["points_per_75"], expected_points, 1e-9):
                fail(errors, "points_per_75 formula mismatch in sampled rows")
                break
            expected_ftr = row["free_throw_attempts"] / row["field_goal_attempts"] if row["field_goal_attempts"] else math.nan
            if not pd.isna(expected_ftr) and not pd.isna(row["free_throw_rate"]) and not close(row["free_throw_rate"], expected_ftr, 1e-9):
                fail(errors, "free_throw_rate formula mismatch in sampled rows")
                break

    for _, row in leagues.iterrows():
        expected_ts = row["league_points"] / (2 * (row["league_fga"] + 0.44 * row["league_fta"]))
        if not close(row["league_ts_pct"], expected_ts, 1e-9):
            fail(errors, f"league_ts_pct formula mismatch for {row['season']}")

    percentile_cols = [column for column in players.columns if column.endswith("_season_percentile")]
    for column in percentile_cols:
        values = players[column].dropna().astype(float)
        if not values.empty and ((values < 0) | (values > 100)).any():
            fail(errors, f"{column} has values outside 0-100")

    if players["player_name"].astype(str).str.contains("Ä|�", regex=True).any():
        fail(errors, "player names contain mojibake markers")

    for player_name, year in STAR_CHECKS:
        subset = players[(players["season_end_year"] == year) & (players["player_name"] == player_name)]
        if subset.empty:
            fail(errors, f"missing star sanity row: {player_name} {year}")

    identity_checks = [
        ("Seattle SuperSonics", "franchise:thunder"),
        ("Oklahoma City Thunder", "franchise:thunder"),
        ("New Jersey Nets", "franchise:nets"),
        ("Brooklyn Nets", "franchise:nets"),
        ("Vancouver Grizzlies", "franchise:grizzlies"),
        ("Memphis Grizzlies", "franchise:grizzlies"),
        ("Charlotte Bobcats", "franchise:charlotte-hornets-current"),
        ("New Orleans Pelicans", "franchise:new-orleans-pelicans"),
    ]
    for team_name, franchise_id in identity_checks:
        subset = teams[teams["team_name"] == team_name]
        if subset.empty:
            fail(errors, f"missing team identity row for {team_name}")
        elif not subset["franchise_id"].eq(franchise_id).all():
            fail(errors, f"unexpected franchise_id for {team_name}")

    split_keys = set(zip(splits["player_id"], splits["season_end_year"]))
    primary_keys = set(zip(players["player_id"], players["season_end_year"]))
    missing_primary = split_keys - primary_keys
    if missing_primary:
        fail(errors, f"{len(missing_primary)} player split keys have no primary player season")

    manifest_counts = manifest.get("row_counts", {})
    expected_counts = {
        "player_seasons": len(players),
        "player_team_splits": len(splits),
        "team_seasons": len(teams),
        "league_seasons": len(leagues),
        "unique_players": players["player_id"].nunique(),
    }
    for key, expected in expected_counts.items():
        if int(manifest_counts.get(key, -1)) != int(expected):
            fail(errors, f"manifest row_counts.{key} mismatch: {manifest_counts.get(key)} != {expected}")

    summary = {
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "row_counts": expected_counts,
        "season_range": {
            "start": players["season"].iloc[players["season_end_year"].astype(int).argmin()],
            "end": players["season"].iloc[players["season_end_year"].astype(int).argmax()],
        },
        "team_season_rows": len(teams),
        "league_season_rows": len(leagues),
        "player_split_rows": len(splits),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(validate())
