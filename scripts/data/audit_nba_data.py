#!/usr/bin/env python3
"""Audit NBA Shot Detail and Play-by-Play V3 archives.

The script reads raw .tar.xz archives in-place, writes reports and metadata,
and never extracts or modifies raw data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import tarfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


SHOT_IMPORTANT = [
    "GAME_ID",
    "GAME_EVENT_ID",
    "PLAYER_ID",
    "PLAYER_NAME",
    "TEAM_ID",
    "TEAM_NAME",
    "GAME_DATE",
    "HTM",
    "VTM",
    "PERIOD",
    "MINUTES_REMAINING",
    "SECONDS_REMAINING",
    "EVENT_TYPE",
    "ACTION_TYPE",
    "SHOT_TYPE",
    "SHOT_ZONE_BASIC",
    "SHOT_ZONE_AREA",
    "SHOT_ZONE_RANGE",
    "SHOT_DISTANCE",
    "LOC_X",
    "LOC_Y",
    "SHOT_MADE_FLAG",
]

PBP_IMPORTANT = [
    "gameId",
    "actionNumber",
    "actionId",
    "actionType",
    "subType",
    "period",
    "clock",
    "teamId",
    "teamTricode",
    "personId",
    "playerName",
    "description",
    "scoreHome",
    "scoreAway",
    "shotResult",
    "shotValue",
    "isFieldGoal",
    "location",
]


@dataclass(frozen=True)
class DatasetFile:
    path: Path
    kind: str
    season_start: int
    season_label: str
    season_type: str


def season_label(start: int) -> str:
    return f"{start}-{str(start + 1)[-2:]}"


def parse_dataset(path: Path) -> DatasetFile | None:
    name = path.name.lower()
    match = re.search(r"(20\d{2})", name)
    if not match:
        return None
    start = int(match.group(1))
    if "shotdetail" in name:
        kind = "shots"
    elif "nbastatsv3" in name or "play" in name:
        kind = "play_by_play"
    else:
        return None
    path_text = path.as_posix().lower()
    if "_po_" in f"_{name}_" or "/playoffs/" in path_text:
        season_type = "playoffs"
    elif "/regular_season/" in path_text or "_po_" not in f"_{name}_":
        season_type = "regular_season"
    else:
        season_type = "unknown"
    return DatasetFile(path=path, kind=kind, season_start=start, season_label=season_label(start), season_type=season_type)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def open_csv_from_archive(path: Path):
    tf = tarfile.open(path, "r:xz")
    members = [m for m in tf.getmembers() if m.isfile()]
    if len(members) != 1:
        tf.close()
        raise ValueError(f"{path} contains {len(members)} files, expected 1")
    member = members[0]
    raw = tf.extractfile(member)
    if raw is None:
        tf.close()
        raise ValueError(f"Cannot read {member.name} from {path}")
    text = io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")
    return tf, member, csv.DictReader(text)


def to_int(value: Any, default: int | None = None) -> int | None:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def shot_clock(row: dict[str, str]) -> str:
    minutes = to_int(row.get("MINUTES_REMAINING"), 0) or 0
    seconds = to_int(row.get("SECONDS_REMAINING"), 0) or 0
    return f"PT{minutes:02d}M{seconds:02d}.00S"


def clock_seconds_remaining(clock: str) -> int | None:
    match = re.fullmatch(r"PT(\d+)M(\d+(?:\.\d+)?)S", clock or "")
    if not match:
        return None
    return int(match.group(1)) * 60 + int(float(match.group(2)))


def shot_value(row: dict[str, str]) -> int:
    if row.get("shotValue"):
        return to_int(row.get("shotValue"), 0) or 0
    if row.get("SHOT_TYPE", "").startswith("3PT"):
        return 3
    desc = row.get("description", "")
    if "3PT" in desc:
        return 3
    return 2


def read_dataset(ds: DatasetFile, collect_rows: bool = False) -> tuple[list[dict[str, str]], dict[str, Any]]:
    tf, member, reader = open_csv_from_archive(ds.path)
    rows: list[dict[str, str]] = []
    fields = reader.fieldnames or []
    missing = Counter()
    distinct = {field: set() for field in fields}
    seen_rows: set[bytes] = set()
    dupes = 0
    row_count = 0
    unique_games: set[str] = set()
    unique_players: set[str] = set()
    min_date: str | None = None
    max_date: str | None = None
    try:
        for row in reader:
            row_count += 1
            if collect_rows:
                rows.append(row)
            row_key = hashlib.blake2b("\x1f".join(row.get(f, "") for f in fields).encode("utf-8"), digest_size=8).digest()
            if row_key in seen_rows:
                dupes += 1
            else:
                seen_rows.add(row_key)
            game_id = row.get("GAME_ID") or row.get("gameId")
            if game_id:
                unique_games.add(game_id)
            player_id = row.get("PLAYER_ID") or row.get("personId")
            if player_id and player_id != "0":
                unique_players.add(player_id)
            game_date = row.get("GAME_DATE")
            if game_date:
                min_date = game_date if min_date is None else min(min_date, game_date)
                max_date = game_date if max_date is None else max(max_date, game_date)
            for field in fields:
                value = row.get(field, "")
                if value == "":
                    missing[field] += 1
                elif len(distinct[field]) < 10000:
                    distinct[field].add(value)
    finally:
        tf.close()
    date_range = [min_date, max_date] if min_date and max_date else None
    summary = {
        "path": ds.path.as_posix(),
        "archive_sha256": sha256(ds.path),
        "archive_size_bytes": ds.path.stat().st_size,
        "member_name": member.name,
        "member_size_bytes": member.size,
        "kind": ds.kind,
        "season": ds.season_label,
        "season_start": ds.season_start,
        "season_type": ds.season_type,
        "format": "tar.xz containing csv",
        "compressed": True,
        "row_count": row_count,
        "column_count": len(fields),
        "columns": fields,
        "duplicate_rows": dupes,
        "date_range": date_range,
        "unique_games": len(unique_games),
        "unique_players": len(unique_players),
        "missing": {
            f: {"count": missing[f], "pct": round((missing[f] / row_count * 100) if row_count else 0, 4)}
            for f in fields
        },
        "distinct_samples": {f: sorted(list(v))[:25] for f, v in distinct.items() if len(v) <= 25},
    }
    return rows, summary


def read_rows(ds: DatasetFile) -> list[dict[str, str]]:
    rows, _summary = read_dataset(ds, collect_rows=True)
    return rows


def index_pbp_scored_events(pbp_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    by_game: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in pbp_rows:
        by_game[row["gameId"]].append(row)
    enriched: list[dict[str, Any]] = []
    for game_id, rows in by_game.items():
        rows.sort(key=lambda r: to_int(r.get("actionNumber"), 0) or 0)
        score_home = 0
        score_away = 0
        for row in rows:
            before_home, before_away = score_home, score_away
            after_home = to_int(row.get("scoreHome"))
            after_away = to_int(row.get("scoreAway"))
            if after_home is not None and after_away is not None:
                score_home, score_away = after_home, after_away
            if row.get("isFieldGoal") == "1":
                enriched_row = dict(row)
                enriched_row["_score_home_before"] = before_home
                enriched_row["_score_away_before"] = before_away
                enriched_row["_score_home_after"] = score_home
                enriched_row["_score_away_after"] = score_away
                enriched.append(enriched_row)
    return enriched


def group_by(rows: list[dict[str, Any]], keys: list[str]) -> dict[tuple[str, ...], list[dict[str, Any]]]:
    out: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out[tuple(str(row.get(k, "")) for k in keys)].append(row)
    return out


def dataset_key_label(key: tuple[str, str]) -> str:
    season, season_type = key
    return f"{season} {season_type}"


def matching_and_clutch(
    shots_by_season: dict[tuple[str, str], list[dict[str, str]]],
    pbp_by_season: dict[tuple[str, str], list[dict[str, str]]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    matching: dict[str, Any] = {}
    clutch: dict[str, Any] = {}
    score_semantics: dict[str, Any] = {}
    for key, shots in sorted(shots_by_season.items()):
        season, season_type = key
        label = dataset_key_label(key)
        pbp_shots = index_pbp_scored_events(pbp_by_season.get(key, []))
        indexes = {
            "A_game_event_id": group_by(pbp_shots, ["gameId", "actionNumber"]),
            "B_game_period_clock_player": group_by(pbp_shots, ["gameId", "period", "clock", "personId"]),
            "C_game_period_clock_team_result": group_by(pbp_shots, ["gameId", "period", "clock", "teamId", "shotResult"]),
        }
        strategies = {
            "A_game_event_id": lambda s: (s["GAME_ID"], s["GAME_EVENT_ID"]),
            "B_game_period_clock_player": lambda s: (s["GAME_ID"], s["PERIOD"], shot_clock(s), s["PLAYER_ID"]),
            "C_game_period_clock_team_result": lambda s: (
                s["GAME_ID"],
                s["PERIOD"],
                shot_clock(s),
                s["TEAM_ID"],
                "Made" if s.get("SHOT_MADE_FLAG") == "1" else "Missed",
            ),
        }
        season_match: dict[str, Any] = {}
        selected_matches: list[tuple[dict[str, str], dict[str, Any]]] = []
        for name, key_fn in strategies.items():
            matched = ambiguous = unmatched = duplicate_targets = 0
            target_counts = Counter()
            examples_unmatched = []
            examples_ambiguous = []
            for shot in shots:
                candidates = indexes[name].get(tuple(key_fn(shot)), [])
                if len(candidates) == 1:
                    matched += 1
                    target_counts[(candidates[0]["gameId"], candidates[0]["actionNumber"])] += 1
                    if name == "A_game_event_id":
                        selected_matches.append((shot, candidates[0]))
                elif len(candidates) == 0:
                    unmatched += 1
                    if len(examples_unmatched) < 8:
                        examples_unmatched.append({k: shot.get(k, "") for k in ["GAME_ID", "GAME_EVENT_ID", "PLAYER_NAME", "PERIOD", "MINUTES_REMAINING", "SECONDS_REMAINING", "EVENT_TYPE", "ACTION_TYPE"]})
                else:
                    ambiguous += 1
                    if len(examples_ambiguous) < 8:
                        examples_ambiguous.append({k: shot.get(k, "") for k in ["GAME_ID", "GAME_EVENT_ID", "PLAYER_NAME", "PERIOD", "MINUTES_REMAINING", "SECONDS_REMAINING"]})
            duplicate_targets = sum(c - 1 for c in target_counts.values() if c > 1)
            season_match[name] = {
                "matched": matched,
                "unmatched": unmatched,
                "ambiguous": ambiguous,
                "duplicate_target_matches": duplicate_targets,
                "match_rate_pct": round(matched / len(shots) * 100, 4) if shots else 0,
                "examples_unmatched": examples_unmatched,
                "examples_ambiguous": examples_ambiguous,
            }
        matching[label] = season_match

        made_after_consistent = 0
        made_checked = 0
        missed_blank = 0
        missed_checked = 0
        clutch_rows = []
        for shot, pbp in selected_matches:
            made = shot.get("SHOT_MADE_FLAG") == "1"
            value = shot_value(shot)
            loc = pbp.get("location")
            before_home = pbp["_score_home_before"]
            before_away = pbp["_score_away_before"]
            after_home = pbp["_score_home_after"]
            after_away = pbp["_score_away_after"]
            if made:
                made_checked += 1
                expected_home = before_home + (value if loc == "h" else 0)
                expected_away = before_away + (value if loc == "v" else 0)
                if after_home == expected_home and after_away == expected_away:
                    made_after_consistent += 1
            else:
                missed_checked += 1
                if pbp.get("scoreHome", "") == "" and pbp.get("scoreAway", "") == "":
                    missed_blank += 1
            period = to_int(shot.get("PERIOD"), 0) or 0
            seconds_left = (to_int(shot.get("MINUTES_REMAINING"), 0) or 0) * 60 + (to_int(shot.get("SECONDS_REMAINING"), 0) or 0)
            diff = abs(before_home - before_away)
            if period >= 4 and seconds_left <= 300 and diff <= 5:
                shooter_score = before_home if loc == "h" else before_away
                opponent_score = before_away if loc == "h" else before_home
                clutch_rows.append(
                    {
                        "shot": shot,
                        "pbp": pbp,
                        "value": value,
                        "made": made,
                        "diff_before": shooter_score - opponent_score,
                    }
                )
        score_semantics[label] = {
            "made_fg_events_checked": made_checked,
            "made_fg_post_event_score_consistent": made_after_consistent,
            "made_fg_post_event_score_consistency_pct": round(made_after_consistent / made_checked * 100, 4) if made_checked else 0,
            "missed_fg_events_checked": missed_checked,
            "missed_fg_blank_score_fields": missed_blank,
            "interpretation": "PBP score fields are post-event on made field goals; missed field goals usually have blank score fields, so running prior score is required.",
        }
        attempts = len(clutch_rows)
        made_count = sum(1 for r in clutch_rows if r["made"])
        two_pa = sum(1 for r in clutch_rows if r["value"] == 2)
        three_pa = sum(1 for r in clutch_rows if r["value"] == 3)
        clutch[label] = {
            "season": season,
            "season_type": season_type,
            "total_shot_attempts": len(shots),
            "clutch_shot_attempts": attempts,
            "clutch_made_shots": made_count,
            "clutch_fg_pct": round(made_count / attempts * 100, 2) if attempts else 0,
            "clutch_2pa": two_pa,
            "clutch_3pa": three_pa,
            "unique_clutch_players": len({r["shot"]["PLAYER_ID"] for r in clutch_rows}),
            "unique_clutch_games": len({r["shot"]["GAME_ID"] for r in clutch_rows}),
        }
    return matching, score_semantics, clutch


def schema_consistency(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    by_kind: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for summary in summaries:
        by_kind[summary["kind"]].append(summary)
    for kind, items in by_kind.items():
        all_cols = sorted(set().union(*(set(i["columns"]) for i in items)))
        seasons = {}
        for item in items:
            seasons[f'{item["season"]} {item["season_type"]}'] = {
                "missing_columns": [c for c in all_cols if c not in item["columns"]],
                "extra_columns": [c for c in item["columns"] if any(c not in other["columns"] for other in items)],
            }
        out[kind] = {"all_columns": all_cols, "seasons": seasons}
    return out


def split_feasibility(clutch: dict[str, Any], summaries: list[dict[str, Any]]) -> dict[str, Any]:
    shot_summaries_by_season: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for summary in summaries:
        if summary["kind"] == "shots":
            shot_summaries_by_season[summary["season"]].append(summary)
    split_map = {
        "train": ["2020-21", "2021-22", "2022-23", "2023-24"],
        "validation": ["2024-25"],
        "test": ["2025-26"],
    }
    splits: dict[str, Any] = {}
    for split, seasons in split_map.items():
        found = [s for s in seasons if s in shot_summaries_by_season]
        summaries_for_split = [summary for season in found for summary in shot_summaries_by_season[season]]
        clutch_for_split = [
            value
            for value in clutch.values()
            if value.get("season") in found
        ]
        splits[split] = {
            "requested_seasons": seasons,
            "available_seasons": found,
            "missing_seasons": [s for s in seasons if s not in shot_summaries_by_season],
            "total_shots": sum(s["row_count"] for s in summaries_for_split),
            "clutch_shots": sum(c.get("clutch_shot_attempts", 0) for c in clutch_for_split),
            "games": sum(s["unique_games"] for s in summaries_for_split),
            "players": sum(s["unique_players"] for s in summaries_for_split),
            "season_types": sorted({s["season_type"] for s in summaries_for_split}),
        }
    required = {"regular_season", "playoffs"}
    missing_pairs = []
    for season in split_map["train"] + split_map["validation"] + split_map["test"]:
        present = {s["season_type"] for s in shot_summaries_by_season.get(season, [])}
        for season_type in sorted(required - present):
            missing_pairs.append(f"{season} {season_type}")
    splits["fully_viable"] = not missing_pairs
    splits["concerns"] = [] if not missing_pairs else [f"Missing shot datasets for: {', '.join(missing_pairs)}"]
    return splits


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)


def write_reports(root: Path, payload: dict[str, Any]) -> None:
    report_dir = root / "reports" / "data-audit"
    metadata_dir = root / "data" / "metadata"
    report_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    summaries = payload["datasets"]
    inventory_rows = [
        [s["path"], s["kind"], s["season"], s["season_type"], s["format"], s["archive_size_bytes"], s["row_count"], s["column_count"], s["unique_games"], s["unique_players"]]
        for s in summaries
    ]
    (report_dir / "DATA_INVENTORY.md").write_text(
        "# Data Inventory\n\n"
        + md_table(["Path", "Type", "Season", "Season type", "Format", "Archive bytes", "Rows", "Cols", "Games", "Players"], inventory_rows)
        + "\n\n## Application source\n\nThe existing .NET application remains under `nba-clutch-agent/`. No application source files are modified by the audit script.\n",
        encoding="utf-8",
    )

    schema_parts = ["# Schema Audit\n"]
    for s in summaries:
        important = SHOT_IMPORTANT if s["kind"] == "shots" else PBP_IMPORTANT
        rows = []
        for col in important:
            if col in s["columns"]:
                miss = s["missing"][col]
                rows.append([col, miss["count"], f'{miss["pct"]}%'])
        schema_parts.append(f'\n## {s["kind"]} {s["season"]} {s["season_type"]}\n')
        schema_parts.append(f'Columns: {s["column_count"]}; rows: {s["row_count"]}; duplicate rows: {s["duplicate_rows"]}; date range: {s["date_range"]}\n\n')
        schema_parts.append(md_table(["Important field", "Missing", "Missing pct"], rows))
        schema_parts.append("\n\nExact columns: `" + "`, `".join(s["columns"]) + "`\n")
    schema_parts.append("\n## Cross-season schema differences\n\n```json\n")
    schema_parts.append(json.dumps(payload["schema_consistency"], indent=2))
    schema_parts.append("\n```\n")
    (report_dir / "SCHEMA_AUDIT.md").write_text("\n".join(schema_parts), encoding="utf-8")

    match_parts = ["# Matching Audit\n"]
    for season, strategies in payload["matching"].items():
        match_parts.append(f"\n## {season}\n")
        rows = []
        for name, m in strategies.items():
            rows.append([name, m["matched"], m["unmatched"], m["ambiguous"], m["duplicate_target_matches"], f'{m["match_rate_pct"]}%'])
        match_parts.append(md_table(["Strategy", "Matched", "Unmatched", "Ambiguous", "Duplicate targets", "Match rate"], rows))
        match_parts.append("\n\nRecommended primary key: `GAME_ID + GAME_EVENT_ID` to PBP `gameId + actionNumber`, with `period + clock + player/team/result` as verification fallback.\n")
    match_parts.append("\n## Score-before-shot semantics\n\n")
    match_parts.append(md_table(
        ["Season", "Made checked", "Post-score consistent", "Consistency", "Misses checked", "Miss blank scores"],
        [[s, v["made_fg_events_checked"], v["made_fg_post_event_score_consistent"], f'{v["made_fg_post_event_score_consistency_pct"]}%', v["missed_fg_events_checked"], v["missed_fg_blank_score_fields"]] for s, v in payload["score_semantics"].items()],
    ))
    match_parts.append("\n\nConclusion: use running score before each field-goal event. Made-shot score fields are post-event; missed-shot score fields are commonly blank.\n")
    (report_dir / "MATCHING_AUDIT.md").write_text("\n".join(match_parts), encoding="utf-8")

    clutch_rows = [[s, v["season_type"], v["total_shot_attempts"], v["clutch_shot_attempts"], v["clutch_made_shots"], v["clutch_fg_pct"], v["clutch_2pa"], v["clutch_3pa"], v["unique_clutch_players"], v["unique_clutch_games"]] for s, v in payload["clutch"].items()]
    (report_dir / "CLUTCH_AUDIT.md").write_text(
        "# Clutch Audit\n\nDefinition: fourth quarter or overtime, 5:00 or less remaining, absolute score differential before the shot <= 5.\n\n"
        + md_table(["Season", "Season type", "Shots", "Clutch FGA", "Clutch FGM", "FG%", "2PA", "3PA", "Players", "Games"], clutch_rows)
        + "\n",
        encoding="utf-8",
    )

    (report_dir / "DATA_PIPELINE_PLAN.md").write_text(
        "# Data Pipeline Plan\n\n"
        "1. Keep all downloaded archives in `data/raw/` and treat them as immutable.\n"
        "2. Add missing regular-season Shot Detail and PBP V3 archives using the same folder convention.\n"
        "3. Build a canonical interim table by reading archives, normalizing column names, and retaining source file lineage.\n"
        "4. Match shots to PBP with `GAME_ID + GAME_EVENT_ID/actionNumber`, then verify period, clock, player, team, and result.\n"
        "5. Derive pre-shot home/away scores from ordered PBP events, not directly from blank miss rows.\n"
        "6. Compute official clutch flags from pre-shot score, period, and remaining clock.\n"
        "7. Only after this audit is accepted, generate processed modeling datasets under `data/processed/`.\n\n"
        "Normalized schema proposal: `season`, `season_type`, `game_id`, `event_id`, `period`, `clock_seconds_remaining`, `game_date`, "
        "`player_id`, `player_name`, `team_id`, `team_tricode`, `is_home`, `shot_made`, `shot_value`, `action_type`, `shot_zone_basic`, "
        "`shot_zone_area`, `shot_zone_range`, `shot_distance`, `loc_x`, `loc_y`, `home_score_before`, `away_score_before`, "
        "`shooter_score_diff_before`, `is_clutch`.\n",
        encoding="utf-8",
    )

    (metadata_dir / "dataset_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_workspace_readme(root: Path) -> None:
    (root / "README.md").write_text(
        "# ML NBA Agent Workspace\n\n"
        "This workspace separates the existing NBA Clutch Agent application from newly downloaded datasets and audit outputs.\n\n"
        "## Structure\n\n"
        "- `nba-clutch-agent/`: existing .NET application source.\n"
        "- `data/raw/`: immutable downloaded archives, grouped by dataset type and season type.\n"
        "- `data/interim/`: future temporary normalized data; currently intentionally empty.\n"
        "- `data/processed/`: future model-ready data; currently intentionally empty.\n"
        "- `data/metadata/`: machine-readable audit summaries.\n"
        "- `scripts/data/`: rerunnable data audit scripts.\n"
        "- `reports/data-audit/`: human-readable audit reports.\n\n"
        "## Raw Data Policy\n\n"
        "Do not edit, overwrite, or delete files under `data/raw/`. Audit scripts read compressed archives directly and write generated outputs only to `reports/` and `data/metadata/`.\n\n"
        "## Run The Audit\n\n"
        "```powershell\n"
        "python scripts/data/audit_nba_data.py\n"
        "```\n\n"
        "No third-party Python packages are required.\n\n"
        "## Next Stage\n\n"
        "Review the audit reports, add regular-season archives if needed, then implement a canonical interim pipeline. Modeling and frontend changes are intentionally out of scope for this audit stage.\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".", help="Workspace root")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    archives = sorted((root / "data" / "raw").rglob("*.tar.xz"))
    datasets = [ds for p in archives if (ds := parse_dataset(p))]
    if not datasets:
        raise SystemExit("No recognized raw archives found under data/raw")

    summaries = []
    dataset_lookup: dict[tuple[str, str, str], DatasetFile] = {}
    for ds in datasets:
        print(f"Summarizing {ds.kind} {ds.season_label} {ds.season_type}...")
        _rows, summary = read_dataset(ds)
        dataset_lookup[(ds.kind, ds.season_label, ds.season_type)] = ds
        summaries.append(summary)

    matching: dict[str, Any] = {}
    score_semantics: dict[str, Any] = {}
    clutch: dict[str, Any] = {}
    for kind, season, season_type in sorted(dataset_lookup):
        if kind != "shots":
            continue
        key = (season, season_type)
        pbp_ds = dataset_lookup.get(("play_by_play", season, season_type))
        if pbp_ds is None:
            continue
        print(f"Matching {season} {season_type}...")
        pair_matching, pair_score_semantics, pair_clutch = matching_and_clutch(
            {key: read_rows(dataset_lookup[(kind, season, season_type)])},
            {key: read_rows(pbp_ds)},
        )
        matching.update(pair_matching)
        score_semantics.update(pair_score_semantics)
        clutch.update(pair_clutch)
    payload = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "datasets": summaries,
        "schema_consistency": schema_consistency(summaries),
        "matching": matching,
        "score_semantics": score_semantics,
        "clutch": clutch,
        "split_feasibility": split_feasibility(clutch, summaries),
    }
    write_reports(root, payload)
    write_workspace_readme(root)
    print(f"Audited {len(summaries)} datasets")
    print(f"Wrote reports to {(root / 'reports' / 'data-audit').relative_to(root)}")
    print(f"Wrote metadata to {(root / 'data' / 'metadata' / 'dataset_summary.json').relative_to(root)}")


if __name__ == "__main__":
    main()
