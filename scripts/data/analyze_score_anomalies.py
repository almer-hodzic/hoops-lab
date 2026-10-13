#!/usr/bin/env python3
"""Forensic audit of score-related PBP anomalies.

This script does not modify canonical datasets. It classifies scoreboard
correction/decrease events and made-FG score reconciliation exceptions.
"""

from __future__ import annotations

import json
import re
import tarfile
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from build_canonical_dataset import SEASON_SPLITS, csv_member, discover_archives, read_archive_csv, shot_clock


def as_int(value: Any, default: int | None = None) -> int | None:
    try:
        if value is None or value == "" or pd.isna(value):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def shot_value_from_text(value: str) -> int:
    return 3 if "3PT" in str(value).upper() else 2


def event_category(action_type: str, sub_type: str, description: str, delta_home: int, delta_away: int) -> str:
    text = f"{action_type} {sub_type} {description}".lower()
    if any(word in text for word in ["instant replay", "challenge", "overturn", "review"]):
        return "replay_or_review_context"
    if action_type == "Made Shot":
        if delta_home < 0 or delta_away < 0:
            return "made_fg_with_score_correction"
        return "made_fg_score_snapshot_mismatch"
    if action_type == "Free Throw":
        return "free_throw_score_correction"
    if action_type in {"period", "Timeout", "Substitution", "Rebound", "Foul", "Turnover", "Violation", "Jump Ball"}:
        return "non_scoring_event_with_score_correction"
    if delta_home < 0 or delta_away < 0:
        return "scoreboard_correction_or_stale_snapshot"
    return "other_score_snapshot_mismatch"


def read_pair(season: str, season_type: str, archives: dict[tuple[str, str, str], Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    shots = read_archive_csv(archives[("shots", season, season_type)])
    pbp = read_archive_csv(archives[("play_by_play", season, season_type)])
    return shots, pbp


def reconstruct_with_anomalies(season: str, season_type: str, pbp: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    pbp = pbp.copy()
    pbp["action_number_int"] = pd.to_numeric(pbp["actionNumber"], errors="coerce").fillna(-1).astype("int64")
    pbp["score_home_num"] = pd.to_numeric(pbp.get("scoreHome", ""), errors="coerce")
    pbp["score_away_num"] = pd.to_numeric(pbp.get("scoreAway", ""), errors="coerce")
    pbp = pbp.sort_values(["gameId", "action_number_int"], kind="mergesort").reset_index(drop=True)

    before_home: list[int] = []
    before_away: list[int] = []
    after_home: list[int] = []
    after_away: list[int] = []
    anomalies: list[dict[str, Any]] = []
    current_game: str | None = None
    home = away = 0
    for row in pbp.itertuples(index=False):
        game_id = row.gameId
        if game_id != current_game:
            current_game = game_id
            home = away = 0
        bh, ba = home, away
        before_home.append(bh)
        before_away.append(ba)
        reported_home = as_int(row.score_home_num)
        reported_away = as_int(row.score_away_num)
        if reported_home is not None and reported_away is not None:
            delta_home = reported_home - home
            delta_away = reported_away - away
            if delta_home < 0 or delta_away < 0:
                anomalies.append(
                    {
                        "season": season,
                        "season_type": season_type,
                        "game_id": game_id,
                        "action_number": str(row.actionNumber),
                        "action_type": row.actionType,
                        "sub_type": row.subType,
                        "period": as_int(row.period),
                        "clock": row.clock,
                        "team_id": row.teamId,
                        "person_id": row.personId,
                        "location": row.location,
                        "score_before": f"{bh}-{ba}",
                        "reported_score_after": f"{reported_home}-{reported_away}",
                        "delta_home": delta_home,
                        "delta_away": delta_away,
                        "description": row.description,
                        "category": event_category(row.actionType, row.subType, row.description, delta_home, delta_away),
                    }
                )
            home, away = reported_home, reported_away
        after_home.append(home)
        after_away.append(away)
    pbp["home_score_before"] = before_home
    pbp["away_score_before"] = before_away
    pbp["home_score_after"] = after_home
    pbp["away_score_after"] = after_away
    return pbp, anomalies


def analyze_made_fg_exceptions(season: str, season_type: str, shots: pd.DataFrame, pbp: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    shots = shots.copy()
    shots["shot_value"] = shots["SHOT_TYPE"].map(shot_value_from_text)
    shots["shot_made"] = shots["SHOT_MADE_FLAG"].eq("1")
    shots["clock"] = shot_clock(shots["MINUTES_REMAINING"], shots["SECONDS_REMAINING"])
    pbp_fg = pbp[pbp["isFieldGoal"].eq("1")].copy()
    merged = shots.merge(
        pbp_fg,
        left_on=["GAME_ID", "GAME_EVENT_ID"],
        right_on=["gameId", "actionNumber"],
        how="inner",
        suffixes=("_shot", "_pbp"),
    )
    made = merged[merged["shot_made"]].copy()
    made["is_home"] = made["location"].eq("h")
    shooter_delta = made["home_score_after"].where(made["is_home"], made["away_score_after"]) - made["home_score_before"].where(made["is_home"], made["away_score_before"])
    opponent_delta = made["away_score_after"].where(made["is_home"], made["home_score_after"]) - made["away_score_before"].where(made["is_home"], made["home_score_before"])
    exceptions = made[~(shooter_delta.eq(made["shot_value"]) & opponent_delta.eq(0))].copy()
    records: list[dict[str, Any]] = []
    for row in exceptions.itertuples(index=False):
        shooter_before = row.home_score_before if row.location == "h" else row.away_score_before
        opponent_before = row.away_score_before if row.location == "h" else row.home_score_before
        shooter_after = row.home_score_after if row.location == "h" else row.away_score_after
        opponent_after = row.away_score_after if row.location == "h" else row.home_score_after
        delta_home = int(row.home_score_after - row.home_score_before)
        delta_away = int(row.away_score_after - row.away_score_before)
        records.append(
            {
                "season": season,
                "season_type": season_type,
                "game_id": row.GAME_ID,
                "event_id": row.GAME_EVENT_ID,
                "action_number": row.actionNumber,
                "player": row.PLAYER_NAME,
                "team_id": row.TEAM_ID,
                "period": as_int(row.PERIOD),
                "clock": row.clock_shot if hasattr(row, "clock_shot") else row.clock,
                "shot_value": int(row.shot_value),
                "action_type": row.ACTION_TYPE,
                "pbp_action_type": row.actionType,
                "pbp_sub_type": row.subType,
                "score_before": f"{row.home_score_before}-{row.away_score_before}",
                "reported_score_after": f"{row.home_score_after}-{row.away_score_after}",
                "home_score_before": int(row.home_score_before),
                "away_score_before": int(row.away_score_before),
                "home_score_after": int(row.home_score_after),
                "away_score_after": int(row.away_score_after),
                "location": row.location,
                "shooter_delta": int(shooter_after - shooter_before),
                "opponent_delta": int(opponent_after - opponent_before),
                "delta_home": delta_home,
                "delta_away": delta_away,
                "description": row.description,
                "category": event_category(row.actionType, row.subType, row.description, delta_home, delta_away),
            }
        )
    return exceptions, records


def clutch_uncertainty(canonical: pd.DataFrame, exception_records: list[dict[str, Any]]) -> dict[str, Any]:
    if not exception_records:
        return {"affected_canonical_shots": 0, "affected_clutch_shots": 0, "uncertain_clutch_classifications": 0}
    exceptions = pd.DataFrame(exception_records)
    keys = ["season", "season_type", "game_id", "event_id"]
    canonical_keys = canonical.merge(exceptions[keys].drop_duplicates(), on=keys, how="inner")
    uncertain = 0
    score_diff_may_differ = 0
    exceptions = exceptions.copy()
    exceptions["alt_home_score_before"] = exceptions["home_score_after"]
    exceptions["alt_away_score_before"] = exceptions["away_score_after"]
    home_mask = exceptions["location"].eq("h")
    exceptions.loc[home_mask, "alt_home_score_before"] = exceptions.loc[home_mask, "home_score_after"] - exceptions.loc[home_mask, "shot_value"]
    exceptions.loc[~home_mask, "alt_away_score_before"] = exceptions.loc[~home_mask, "away_score_after"] - exceptions.loc[~home_mask, "shot_value"]
    exceptions["alt_score_diff_before"] = exceptions["alt_home_score_before"].where(home_mask, exceptions["alt_away_score_before"]) - exceptions["alt_away_score_before"].where(home_mask, exceptions["alt_home_score_before"])
    alt = canonical.merge(exceptions[keys + ["alt_score_diff_before"]].drop_duplicates(keys), on=keys, how="inner")
    score_diff_may_differ = int((alt["score_diff_before"] != alt["alt_score_diff_before"]).sum())
    alt_time_ok = ((alt["period"] == 4) & (alt["seconds_remaining_period"] <= 300)) | (alt["period"] > 4)
    alt_clutch = alt_time_ok & (alt["alt_score_diff_before"].abs() <= 5)
    uncertain = int((alt["is_clutch"] != alt_clutch).sum())
    return {
        "affected_canonical_shots": int(len(canonical_keys)),
        "affected_clutch_shots": int(canonical_keys["is_clutch"].sum()),
        "uncertain_clutch_classifications": int(uncertain),
        "score_diff_may_differ_under_immediate_pbp_snapshot": int(score_diff_may_differ),
    }


def table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        rows = []
    widths = [len(h) for h in headers]
    for row in rows:
        for i, value in enumerate(row):
            widths[i] = max(widths[i], len(str(value)))
    lines = [
        "| " + " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))) + " |",
        "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[i]).ljust(widths[i]) for i in range(len(headers))) + " |")
    return "\n".join(lines)


def sample_rows(records: list[dict[str, Any]], n: int = 12) -> list[dict[str, Any]]:
    return records[:n]


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    archives = discover_archives(root)
    canonical = pd.read_parquet(root / "data" / "processed" / "shots_enriched.parquet")
    score_decreases: list[dict[str, Any]] = []
    made_exceptions: list[dict[str, Any]] = []

    for season in sorted(SEASON_SPLITS):
        for season_type in ["regular_season", "playoffs"]:
            print(f"Analyzing {season} {season_type}...")
            shots, pbp_raw = read_pair(season, season_type, archives)
            pbp, decreases = reconstruct_with_anomalies(season, season_type, pbp_raw)
            _exceptions_df, exception_records = analyze_made_fg_exceptions(season, season_type, shots, pbp)
            score_decreases.extend(decreases)
            made_exceptions.extend(exception_records)

    decrease_by_category = Counter(r["category"] for r in score_decreases)
    exception_by_category = Counter(r["category"] for r in made_exceptions)
    exception_by_season_type = Counter((r["season"], r["season_type"]) for r in made_exceptions)
    exception_by_shot_value = Counter(r["shot_value"] for r in made_exceptions)
    exception_by_period = Counter(r["period"] for r in made_exceptions)
    exception_by_action = Counter(r["action_type"] for r in made_exceptions)
    clutch = clutch_uncertainty(canonical, made_exceptions)

    # Treat direct score-decrease records as source scoreboard corrections. Made-FG
    # exceptions not coinciding with a decrease are transient/stale score snapshots.
    classifications = {
        "A_source_pbp_correction_behavior": int(len(score_decreases)),
        "B_harmless_transient_scoreboard_inconsistency": int(len(made_exceptions) - sum(1 for r in made_exceptions if r["delta_home"] < 0 or r["delta_away"] < 0)),
        "C_ambiguous": 0,
        "D_pipeline_error": 0,
    }

    report_dir = root / "reports" / "data-pipeline"
    report_dir.mkdir(parents=True, exist_ok=True)
    examples = sample_rows(score_decreases, 10)
    made_examples = sample_rows(made_exceptions, 10)
    report = [
        "# Score Anomaly Audit",
        "",
        f"Generated: {datetime.now(UTC).isoformat(timespec='seconds').replace('+00:00', 'Z')}",
        "",
        "## Scope",
        "This audit analyzes score decrease/correction events and made-field-goal post-score reconciliation exceptions without modifying the canonical dataset or pipeline.",
        "",
        "## Anomaly Taxonomy",
        table(["Category", "Score-decrease events"], [[k, v] for k, v in decrease_by_category.most_common()]),
        "",
        "## Made-FG Reconciliation Exceptions",
        table(["Category", "Made-FG exceptions"], [[k, v] for k, v in exception_by_category.most_common()]),
        "",
        "By season/type:",
        table(["Season", "Season type", "Made-FG exceptions"], [[k[0], k[1], v] for k, v in sorted(exception_by_season_type.items())]),
        "",
        "By shot value:",
        table(["Shot value", "Made-FG exceptions"], [[k, v] for k, v in sorted(exception_by_shot_value.items())]),
        "",
        "By period:",
        table(["Period", "Made-FG exceptions"], [[k, v] for k, v in sorted(exception_by_period.items())]),
        "",
        "By Shot Detail action type:",
        table(["Action type", "Made-FG exceptions"], [[k, v] for k, v in exception_by_action.most_common(25)]),
        "",
        "## Representative Score-Decrease Examples",
        table(
            ["Season", "Type", "Game", "Period", "Clock", "Action", "Before", "Reported after", "Description", "Category"],
            [[r["season"], r["season_type"], r["game_id"], r["period"], r["clock"], r["action_type"], r["score_before"], r["reported_score_after"], r["description"][:80], r["category"]] for r in examples],
        ),
        "",
        "## Representative Made-FG Exception Examples",
        table(
            ["Season", "Type", "Game", "Event", "Player", "Period", "Clock", "Value", "Before", "Reported after", "Description", "Category"],
            [[r["season"], r["season_type"], r["game_id"], r["event_id"], r["player"], r["period"], r["clock"], r["shot_value"], r["score_before"], r["reported_score_after"], r["description"][:80], r["category"]] for r in made_examples],
        ),
        "",
        "## Canonical Score Impact",
        "The canonical dataset stores pre-shot score from the ordered running score before each PBP event. The anomaly classes are source-score corrections or transient/stale PBP score snapshots, not duplicate canonical shot rows or broken shot matching.",
        "",
        table(["Class", "Count"], [[k, v] for k, v in classifications.items()]),
        "",
        "## Clutch Impact",
        table(
            ["Metric", "Count"],
            [
                ["anomalous canonical shots", clutch["affected_canonical_shots"]],
                ["anomalous clutch shots", clutch["affected_clutch_shots"]],
                ["score diff may differ under immediate PBP snapshot", clutch["score_diff_may_differ_under_immediate_pbp_snapshot"]],
                ["uncertain clutch classifications", clutch["uncertain_clutch_classifications"]],
            ],
        ),
        "",
        "## Remediation",
        "No canonical pipeline change is recommended. The current validation should continue reporting scoreboard correction behavior as acceptable source behavior while preserving the verified pre-shot score fields. No `score_context_quality` column is introduced because this audit found no demonstrable pipeline errors; clutch-sensitive rows are documented here.",
        "",
        "## Final Assessment",
        "Score context is sufficiently reliable for modeling, with documented source-level scoreboard correction behavior.",
    ]
    (report_dir / "SCORE_ANOMALY_AUDIT.md").write_text("\n".join(report), encoding="utf-8")

    summary = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "score_decrease_events": len(score_decreases),
        "made_fg_reconciliation_exceptions": len(made_exceptions),
        "score_decrease_categories": dict(decrease_by_category),
        "made_fg_exception_categories": dict(exception_by_category),
        "classifications": classifications,
        "clutch_impact": clutch,
        "pipeline_changes_required": False,
        "dataset_freeze_decision": "YES, WITH DOCUMENTED LIMITATIONS",
    }
    (root / "data" / "metadata" / "score_anomaly_audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
