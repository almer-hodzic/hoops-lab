"""Validation checks for defensive context datasets."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed" / "defense"
ANALYSIS = ROOT / "data" / "analysis" / "defense"


def fail(message: str) -> None:
    raise AssertionError(message)


def check_matchups() -> None:
    path = PROCESSED / "matchups_canonical.parquet"
    if not path.exists():
        fail("matchups_canonical.parquet missing")
    df = pd.read_parquet(path)
    key = ["game_id", "season", "season_type", "offensive_player_id", "defensive_player_id"]
    dupes = int(df.duplicated(key).sum())
    if dupes:
        fail(f"duplicate matchup canonical keys: {dupes}")
    for made, attempted in [
        ("matchup_fgm", "matchup_fga"),
        ("matchup_3pm", "matchup_3pa"),
        ("matchup_ftm", "matchup_fta"),
        ("help_fgm", "help_fga"),
    ]:
        bad = int((df[made] > df[attempted]).fillna(False).sum())
        if bad:
            fail(f"{made} > {attempted}: {bad}")
    pct_cols = [c for c in df.columns if c.endswith("_pct")]
    for col in pct_cols:
        bad = int(((df[col] < 0) | (df[col] > 1)).fillna(False).sum())
        if bad:
            fail(f"percentage out of range in {col}: {bad}")
    for col in [
        "percentage_defender_total_time",
        "percentage_offensive_total_time",
        "percentage_total_time_both_on",
    ]:
        bad = int(((df[col] < 0) | (df[col] > 2)).fillna(False).sum())
        if bad:
            fail(f"time-share percentage outside documented source range in {col}: {bad}")
    if not df["season"].astype(str).str.match(r"^20\d{2}-\d{2}$").all():
        fail("season parsing invalid in matchups")
    if not set(df["season_type"].dropna().unique()).issubset({"regular_season", "playoffs"}):
        fail("season_type invalid in matchups")


def check_shot_logs() -> None:
    path = PROCESSED / "shot_logs_2014_15.parquet"
    if not path.exists():
        fail("shot_logs_2014_15.parquet missing")
    df = pd.read_parquet(path)
    for col in ["closest_defender_distance", "shot_distance", "touch_time", "dribbles"]:
        bad = int((df[col] < 0).fillna(False).sum())
        if bad and col != "touch_time":
            fail(f"negative {col}: {bad}")
    shot_clock_bad = int(((df["shot_clock"] < 0) | (df["shot_clock"] > 24)).fillna(False).sum())
    if shot_clock_bad:
        fail(f"shot_clock out of bounds: {shot_clock_bad}")
    if df["closest_defender_id"].dropna().astype("int64").astype(str).str.contains(r"\.").any():
        fail("closest defender IDs lost integer precision")
    exp = ANALYSIS / "shot_quality_tracking_experiment.parquet"
    if not exp.exists():
        fail("shot_quality_tracking_experiment.parquet missing")


def check_pt_defend() -> None:
    path = PROCESSED / "pt_defend_season_category.parquet"
    if not path.exists():
        return
    df = pd.read_parquet(path)
    for col in ["defended_fg_pct", "normal_fg_pct", "frequency"]:
        if col in df.columns:
            bad = int(((df[col] < 0) | (df[col] > 1)).fillna(False).sum())
            if bad:
                fail(f"{col} out of range: {bad}")
    if "defended_fgm" in df and "defended_fga" in df:
        bad = int((df["defended_fgm"] > df["defended_fga"]).fillna(False).sum())
        if bad:
            fail(f"defended_fgm > defended_fga: {bad}")


def check_priors() -> None:
    path = PROCESSED / "defender_game_priors.parquet"
    if not path.exists():
        fail("defender_game_priors.parquet missing")
    df = pd.read_parquet(path)
    if (df["defender_prior_matchup_fga"] < 0).any():
        fail("negative prior FGA")
    if not ((df["defender_prior_matchup_fg_pct_shrunk"] >= 0) & (df["defender_prior_matchup_fg_pct_shrunk"] <= 1)).all():
        fail("shrunk FG pct out of range")
    first = df.sort_values(["defensive_player_id", "game_id"]).groupby("defensive_player_id").head(1)
    if (first["defender_prior_matchup_fga"] != 0).any():
        fail("first defender prior is not zero-attempt cold start")


def main() -> int:
    checks = [check_matchups, check_shot_logs, check_pt_defend, check_priors]
    for check in checks:
        check()
    print("defensive data checks passed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1)
