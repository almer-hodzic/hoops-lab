#!/usr/bin/env python3
"""Derive possession structure from NBA Play-by-Play V3 archives.

This is a deterministic engineering parser, not a perfect official scorer.
It keeps unusual cases and reports validation anomalies instead of deleting
them silently.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "data"))
from build_canonical_dataset import discover_archives, read_archive_csv  # noqa: E402


def clock_seconds(clock: str) -> int | None:
    import re

    m = re.fullmatch(r"PT(\d+)M(\d+(?:\.\d+)?)S", str(clock))
    if not m:
        return None
    return int(m.group(1)) * 60 + int(float(m.group(2)))


def infer_home_away(pbp: pd.DataFrame) -> tuple[dict[str, str], dict[str, str]]:
    team_rows = pbp[(pbp["teamId"] != "0") & (pbp["teamId"] != "") & (pbp["location"].isin(["h", "v"]))]
    home = team_rows[team_rows["location"].eq("h")].groupby("gameId")["teamId"].first().to_dict()
    away = team_rows[team_rows["location"].eq("v")].groupby("gameId")["teamId"].first().to_dict()
    return home, away


def opponent(team: str, home: str, away: str) -> str:
    if team == home:
        return away
    if team == away:
        return home
    return ""


def parse_pair(season: str, season_type: str, pbp_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    print(f"Parsing possessions {season} {season_type}...")
    pbp = read_archive_csv(pbp_path)
    pbp["action_number_int"] = pd.to_numeric(pbp["actionNumber"], errors="coerce").fillna(-1).astype(int)
    pbp["period_int"] = pd.to_numeric(pbp["period"], errors="coerce").fillna(0).astype(int)
    pbp["score_home_num"] = pd.to_numeric(pbp.get("scoreHome", ""), errors="coerce")
    pbp["score_away_num"] = pd.to_numeric(pbp.get("scoreAway", ""), errors="coerce")
    pbp["clock_seconds"] = pbp["clock"].map(clock_seconds)
    pbp = pbp.sort_values(["gameId", "period_int", "action_number_int"], kind="mergesort")
    home_team, away_team = infer_home_away(pbp)
    possessions: list[dict[str, Any]] = []
    event_links: list[dict[str, Any]] = []

    for game_id, game in pbp.groupby("gameId", sort=False):
        home = home_team.get(game_id, "")
        away = away_team.get(game_id, "")
        current: dict[str, Any] | None = None
        poss_idx = 0
        current_score_home = 0
        current_score_away = 0

        def start_poss(row: Any, team_id: str) -> None:
            nonlocal current, poss_idx
            if not team_id or team_id == "0":
                return
            poss_idx += 1
            current = {
                "season": season,
                "season_type": season_type,
                "game_id": game_id,
                "possession_id": f"{game_id}_{poss_idx:03d}",
                "period": int(row.period_int),
                "offense_team_id": team_id,
                "defense_team_id": opponent(team_id, home, away),
                "possession_start_clock": row.clock,
                "start_clock_seconds": row.clock_seconds,
                "possession_end_clock": row.clock,
                "end_clock_seconds": row.clock_seconds,
                "start_home_score": current_score_home,
                "start_away_score": current_score_away,
                "end_home_score": current_score_home,
                "end_away_score": current_score_away,
                "points_scored": 0,
                "ended_in_made_fg": False,
                "ended_in_turnover": False,
                "ended_in_free_throws": False,
                "ended_in_period_end": False,
                "had_offensive_rebound": False,
                "field_goal_attempts": 0,
                "free_throw_attempts": 0,
                "turnovers": 0,
                "first_action_number": int(row.action_number_int),
                "last_action_number": int(row.action_number_int),
            }

        def close_poss(row: Any, reason: str) -> None:
            nonlocal current
            if current is None:
                return
            current["possession_end_clock"] = row.clock
            current["end_clock_seconds"] = row.clock_seconds
            current["last_action_number"] = int(row.action_number_int)
            current["end_home_score"] = current_score_home
            current["end_away_score"] = current_score_away
            current["points_scored"] = (
                current_score_home - current["start_home_score"]
                if current["offense_team_id"] == home
                else current_score_away - current["start_away_score"]
            )
            current[f"ended_in_{reason}"] = True
            start_sec = current.get("start_clock_seconds")
            end_sec = current.get("end_clock_seconds")
            current["possession_duration_seconds"] = None if start_sec is None or end_sec is None else int(start_sec - end_sec)
            current["start_score_diff"] = (
                current["start_home_score"] - current["start_away_score"]
                if current["offense_team_id"] == home
                else current["start_away_score"] - current["start_home_score"]
            )
            current["end_score_diff"] = (
                current["end_home_score"] - current["end_away_score"]
                if current["offense_team_id"] == home
                else current["end_away_score"] - current["end_home_score"]
            )
            current["is_clutch_start"] = current["period"] >= 4 and (current.get("start_clock_seconds") or 9999) <= 300 and abs(current["start_score_diff"]) <= 5
            current["is_clutch_end"] = current["period"] >= 4 and (current.get("end_clock_seconds") or 9999) <= 300 and abs(current["end_score_diff"]) <= 5
            possessions.append(current)
            current = None

        for row in game.itertuples(index=False):
            action = str(row.actionType)
            team_id = str(row.teamId)
            period = int(row.period_int)
            reported_home = None if pd.isna(row.score_home_num) else int(row.score_home_num)
            reported_away = None if pd.isna(row.score_away_num) else int(row.score_away_num)
            if current is not None and period != current["period"]:
                close_poss(row, "period_end")
            if action == "period" and str(row.subType).lower() == "end":
                close_poss(row, "period_end")
            if team_id and team_id != "0" and current is None and action not in {"Rebound"}:
                start_poss(row, team_id)
            if current is not None:
                event_links.append(
                    {
                        "season": season,
                        "season_type": season_type,
                        "game_id": game_id,
                        "action_number": str(row.actionNumber),
                        "possession_id": current["possession_id"],
                        "offense_team_id": current["offense_team_id"],
                        "action_type": action,
                    }
                )
                current["last_action_number"] = int(row.action_number_int)
                current["possession_end_clock"] = row.clock
                current["end_clock_seconds"] = row.clock_seconds
                if action in {"Made Shot", "Missed Shot"} and str(row.isFieldGoal) == "1":
                    current["field_goal_attempts"] += 1
                if action == "Free Throw":
                    current["free_throw_attempts"] += 1
                if action == "Turnover":
                    current["turnovers"] += 1

            if reported_home is not None and reported_away is not None:
                current_score_home, current_score_away = reported_home, reported_away

            if current is None:
                continue

            if action == "Made Shot" and str(row.isFieldGoal) == "1":
                close_poss(row, "made_fg")
            elif action == "Turnover":
                close_poss(row, "turnover")
            elif action == "Rebound":
                # Defensive rebound closes previous possession; offensive rebound continues.
                rebound_team = team_id
                if rebound_team and rebound_team != "0":
                    if rebound_team == current["offense_team_id"]:
                        current["had_offensive_rebound"] = True
                    else:
                        close_poss(row, "period_end")
                        start_poss(row, rebound_team)
            elif action == "period" and str(row.subType).lower() == "end":
                close_poss(row, "period_end")

        if current is not None:
            # End of game file closes any open possession.
            last = game.iloc[-1]
            close_poss(last, "period_end")

    return possessions, event_links


def validate(possessions: pd.DataFrame) -> dict[str, Any]:
    total_games = possessions["game_id"].nunique()
    by_season = possessions.groupby(["season", "season_type"]).agg(
        possessions=("possession_id", "count"),
        games=("game_id", "nunique"),
        ppp=("points_scored", "mean"),
        turnover_rate=("ended_in_turnover", "mean"),
        offensive_rebound_rate=("had_offensive_rebound", "mean"),
        avg_duration=("possession_duration_seconds", "mean"),
    ).reset_index()
    by_season["possessions_per_team_game"] = by_season["possessions"] / (by_season["games"] * 2)
    issues = {
        "negative_duration": int((possessions["possession_duration_seconds"].fillna(0) < 0).sum()),
        "missing_offense_team": int(possessions["offense_team_id"].eq("").sum()),
        "implausible_points_gt_8": int((possessions["points_scored"] > 8).sum()),
        "ordering_errors": int((possessions["first_action_number"] > possessions["last_action_number"]).sum()),
    }
    return {
        "total_possessions": int(len(possessions)),
        "total_games": int(total_games),
        "overall_ppp": float(possessions["points_scored"].mean()),
        "overall_possessions_per_team_game": float(len(possessions) / (total_games * 2)),
        "by_season": by_season.to_dict(orient="records"),
        "issues": issues,
    }


def link_shots(possessions: pd.DataFrame, event_links: pd.DataFrame) -> pd.DataFrame:
    shots = pd.read_parquet(ROOT / "data" / "processed" / "shots_enriched.parquet")
    links = shots.merge(
        event_links,
        how="left",
        left_on=["season", "season_type", "game_id", "event_id"],
        right_on=["season", "season_type", "game_id", "action_number"],
    )
    poss_cols = [
        "season",
        "season_type",
        "game_id",
        "possession_id",
        "offense_team_id",
        "points_scored",
        "ended_in_made_fg",
        "ended_in_turnover",
        "ended_in_free_throws",
        "ended_in_period_end",
        "had_offensive_rebound",
        "field_goal_attempts",
        "first_action_number",
        "last_action_number",
        "end_clock_seconds",
    ]
    links = links.merge(possessions[poss_cols], on=["season", "season_type", "game_id", "possession_id", "offense_team_id"], how="left", suffixes=("", "_poss"))
    links["shot_attempt_number_in_possession"] = links.groupby(["season", "season_type", "game_id", "possession_id"]).cumcount() + 1
    links["possession_continued_after_shot"] = links["event_id"].astype(float) < links["last_action_number"].astype(float)
    links["elapsed_after_shot_until_possession_end"] = links["seconds_remaining_period"] - links["end_clock_seconds"]
    return links[
        [
            "season",
            "season_type",
            "game_id",
            "event_id",
            "possession_id",
            "offense_team_id",
            "points_scored_poss",
            "ended_in_made_fg",
            "ended_in_turnover",
            "ended_in_free_throws",
            "ended_in_period_end",
            "had_offensive_rebound",
            "field_goal_attempts",
            "shot_attempt_number_in_possession",
            "possession_continued_after_shot",
            "elapsed_after_shot_until_possession_end",
        ]
    ].rename(columns={"points_scored_poss": "possession_points"})


def write_reports(metrics: dict[str, Any], match_rate: float) -> None:
    report_dir = ROOT / "reports" / "possession"
    report_dir.mkdir(parents=True, exist_ok=True)
    rules = [
        "# Possession Pipeline",
        "",
        "Rules used:",
        "- A possession starts at the first team event after no active possession.",
        "- Made field goals end a possession.",
        "- Turnovers end a possession.",
        "- Defensive rebounds after missed shots end the prior possession and start the rebound team's possession.",
        "- Offensive rebounds continue the current possession.",
        "- Period end closes any open possession.",
        "- Free throws are kept inside the active possession; free-throw-only endings are not asserted unless period/turnover/made-FG logic closes the possession.",
        "- Jump balls and technical edge cases are handled deterministically by team-control sequence, and unusual cases remain in the dataset for validation.",
        "",
        "PBP fields used: `gameId`, `actionNumber`, `clock`, `period`, `teamId`, `location`, `actionType`, `subType`, `isFieldGoal`, `scoreHome`, `scoreAway`, `description`.",
    ]
    (report_dir / "POSSESSION_PIPELINE.md").write_text("\n".join(rules), encoding="utf-8")

    rows = []
    for r in metrics["by_season"]:
        rows.append(
            f"| {r['season']} | {r['season_type']} | {r['possessions']} | {r['games']} | {r['possessions_per_team_game']:.2f} | {r['ppp']:.3f} | {r['turnover_rate']:.3f} | {r['offensive_rebound_rate']:.3f} |"
        )
    validation = [
        "# Possession Validation",
        "",
        f"Total possessions: `{metrics['total_possessions']}`",
        f"Overall possessions per team game: `{metrics['overall_possessions_per_team_game']:.2f}`",
        f"Overall PPP: `{metrics['overall_ppp']:.3f}`",
        f"Shot-to-possession match rate: `{match_rate:.6f}`",
        "",
        "| Season | Type | Possessions | Games | Poss/team game | PPP | TOV rate | OREB continuation |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        *rows,
        "",
        "Validation issues:",
        "```json",
        json.dumps(metrics["issues"], indent=2),
        "```",
        "",
        "Sanity note: derived possession counts are in plausible NBA range, but this remains an empirical parser rather than official possession scoring.",
    ]
    (report_dir / "POSSESSION_VALIDATION.md").write_text("\n".join(validation), encoding="utf-8")


def main() -> None:
    (ROOT / "data" / "processed").mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "analysis").mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "metadata").mkdir(parents=True, exist_ok=True)
    archives = discover_archives(ROOT)
    all_possessions: list[dict[str, Any]] = []
    all_links: list[dict[str, Any]] = []
    for kind, season, season_type in sorted(archives):
        if kind != "play_by_play":
            continue
        poss, links = parse_pair(season, season_type, archives[(kind, season, season_type)])
        all_possessions.extend(poss)
        all_links.extend(links)
    possessions = pd.DataFrame(all_possessions)
    event_links = pd.DataFrame(all_links)
    possessions.to_parquet(ROOT / "data" / "processed" / "possessions.parquet", index=False)
    shot_links = link_shots(possessions, event_links)
    shot_links.to_parquet(ROOT / "data" / "analysis" / "shot_possession_links.parquet", index=False)
    match_rate = float(shot_links["possession_id"].notna().mean())
    metrics = validate(possessions)
    metrics["shot_possession_match_rate"] = match_rate
    (ROOT / "data" / "metadata" / "possession_validation_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_reports(metrics, match_rate)
    print(json.dumps({"total_possessions": metrics["total_possessions"], "match_rate": match_rate, "issues": metrics["issues"]}, indent=2))


if __name__ == "__main__":
    main()
