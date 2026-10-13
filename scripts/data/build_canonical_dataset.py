#!/usr/bin/env python3
"""Build canonical enriched NBA shot datasets from immutable raw archives."""

from __future__ import annotations

import json
import re
import tarfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd


PIPELINE_VERSION = "2.0.0"
SCHEMA_VERSION = "canonical_shots_v1"
SEASON_SPLITS = {
    "2020-21": "train",
    "2021-22": "train",
    "2022-23": "train",
    "2023-24": "train",
    "2024-25": "validation",
    "2025-26": "test",
}
CANONICAL_COLUMNS = [
    "season",
    "season_type",
    "dataset_split",
    "game_id",
    "event_id",
    "game_date",
    "player_id",
    "player_name",
    "team_id",
    "team_name",
    "opponent_team_id",
    "home_team_id",
    "away_team_id",
    "is_home",
    "shot_made",
    "shot_value",
    "points_scored",
    "shot_type",
    "action_type",
    "shot_distance",
    "loc_x",
    "loc_y",
    "shot_zone_basic",
    "shot_zone_area",
    "shot_zone_range",
    "period",
    "clock",
    "seconds_remaining_period",
    "is_overtime",
    "home_score_before",
    "away_score_before",
    "shooter_score_before",
    "opponent_score_before",
    "score_diff_before",
    "game_state_before",
    "abs_score_diff_before",
    "is_clutch",
    "clutch_time_bucket",
]


def season_label(start: int) -> str:
    return f"{start}-{str(start + 1)[-2:]}"


def parse_archive(path: Path) -> dict[str, str] | None:
    name = path.name.lower()
    match = re.search(r"(20\d{2})", name)
    if not match:
        return None
    if "shotdetail" in name:
        kind = "shots"
    elif "nbastatsv3" in name:
        kind = "play_by_play"
    else:
        return None
    text_path = path.as_posix().lower()
    season_type = "playoffs" if "_po_" in f"_{name}_" or "/playoffs/" in text_path else "regular_season"
    start = int(match.group(1))
    return {"kind": kind, "season": season_label(start), "season_type": season_type}


def csv_member(path: Path) -> str:
    with tarfile.open(path, "r:xz") as tf:
        members = [m.name for m in tf.getmembers() if m.isfile()]
    if len(members) != 1:
        raise ValueError(f"{path} should contain exactly one CSV file, found {len(members)}")
    return members[0]


def read_archive_csv(path: Path) -> pd.DataFrame:
    member = csv_member(path)
    with tarfile.open(path, "r:xz") as tf:
        file_obj = tf.extractfile(member)
        if file_obj is None:
            raise ValueError(f"Cannot read {member} from {path}")
        return pd.read_csv(file_obj, dtype=str, keep_default_na=False)


def discover_archives(root: Path) -> dict[tuple[str, str, str], Path]:
    archives: dict[tuple[str, str, str], Path] = {}
    for path in sorted((root / "data" / "raw").rglob("*.tar.xz")):
        parsed = parse_archive(path)
        if parsed:
            key = (parsed["kind"], parsed["season"], parsed["season_type"])
            archives[key] = path
    return archives


def shot_clock(minutes: pd.Series, seconds: pd.Series) -> pd.Series:
    mins = pd.to_numeric(minutes, errors="coerce").fillna(0).astype(int)
    secs = pd.to_numeric(seconds, errors="coerce").fillna(0).astype(int)
    return "PT" + mins.astype(str).str.zfill(2) + "M" + secs.astype(str).str.zfill(2) + ".00S"


def derive_shot_value(shot_type: pd.Series) -> pd.Series:
    return shot_type.str.contains("3PT", case=False, na=False).map({True: 3, False: 2}).astype("int8")


def normalize_game_date(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, format="%Y%m%d", errors="coerce")
    return parsed.dt.strftime("%Y-%m-%d").fillna("")


def reconstruct_pbp(pbp: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    pbp = pbp.copy()
    pbp["action_number_int"] = pd.to_numeric(pbp["actionNumber"], errors="coerce").fillna(-1).astype("int64")
    pbp["score_home_num"] = pd.to_numeric(pbp.get("scoreHome", ""), errors="coerce")
    pbp["score_away_num"] = pd.to_numeric(pbp.get("scoreAway", ""), errors="coerce")
    pbp = pbp.sort_values(["gameId", "action_number_int"], kind="mergesort").reset_index(drop=True)

    before_home: list[int] = []
    before_away: list[int] = []
    after_home: list[int] = []
    after_away: list[int] = []
    score_decrease = 0
    final_scores: dict[str, tuple[int, int]] = {}
    home = away = 0
    current_game: str | None = None
    for row in pbp.itertuples(index=False):
        game_id = row.gameId
        if game_id != current_game:
            if current_game is not None:
                final_scores[current_game] = (home, away)
            current_game = game_id
            home = away = 0
        before_home.append(home)
        before_away.append(away)
        score_home = row.score_home_num
        score_away = row.score_away_num
        if pd.notna(score_home) and pd.notna(score_away):
            new_home = int(score_home)
            new_away = int(score_away)
            if new_home < home or new_away < away:
                score_decrease += 1
            home, away = new_home, new_away
        after_home.append(home)
        after_away.append(away)
    if current_game is not None:
        final_scores[current_game] = (home, away)

    pbp["home_score_before"] = before_home
    pbp["away_score_before"] = before_away
    pbp["home_score_after"] = after_home
    pbp["away_score_after"] = after_away
    team_rows = pbp[(pbp["teamId"] != "0") & (pbp["teamId"] != "") & (pbp["location"].isin(["h", "v"]))]
    home_team = team_rows[team_rows["location"] == "h"].groupby("gameId")["teamId"].first()
    away_team = team_rows[team_rows["location"] == "v"].groupby("gameId")["teamId"].first()
    pbp["home_team_id"] = pbp["gameId"].map(home_team).fillna("")
    pbp["away_team_id"] = pbp["gameId"].map(away_team).fillna("")
    pbp["pbp_shot_value"] = pd.to_numeric(pbp.get("shotValue", ""), errors="coerce")
    metrics = {
        "score_decrease_events": int(score_decrease),
        "games_with_home_team_id": int(home_team.size),
        "games_with_away_team_id": int(away_team.size),
    }
    return pbp, metrics


def time_bucket(seconds: pd.Series) -> pd.Series:
    bins = [-1, 30, 60, 120, 180, 300]
    labels = ["0-30", "31-60", "61-120", "121-180", "181-300"]
    return pd.cut(seconds, bins=bins, labels=labels).astype("string").fillna("")


def process_pair(season: str, season_type: str, shot_path: Path, pbp_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    print(f"Loading {season} {season_type}...")
    shots = read_archive_csv(shot_path)
    pbp_raw = read_archive_csv(pbp_path)
    pbp, score_metrics = reconstruct_pbp(pbp_raw)

    shots["game_id"] = shots["GAME_ID"]
    shots["event_id"] = shots["GAME_EVENT_ID"]
    shots["period_num"] = pd.to_numeric(shots["PERIOD"], errors="coerce").fillna(0).astype("int16")
    shots["clock"] = shot_clock(shots["MINUTES_REMAINING"], shots["SECONDS_REMAINING"])
    shots["seconds_remaining_period"] = (
        pd.to_numeric(shots["MINUTES_REMAINING"], errors="coerce").fillna(0).astype("int16") * 60
        + pd.to_numeric(shots["SECONDS_REMAINING"], errors="coerce").fillna(0).astype("int16")
    )
    shots["shot_made"] = shots["SHOT_MADE_FLAG"].eq("1")
    shots["shot_value"] = derive_shot_value(shots["SHOT_TYPE"])
    shots["points_scored"] = shots["shot_made"].astype("int8") * shots["shot_value"]

    pbp_field_goals = pbp[pbp["isFieldGoal"].eq("1")].copy()
    pbp_subset = pbp_field_goals[
        [
            "gameId",
            "actionNumber",
            "period",
            "clock",
            "teamId",
            "personId",
            "location",
            "description",
            "actionType",
            "subType",
            "shotResult",
            "pbp_shot_value",
            "home_score_before",
            "away_score_before",
            "home_score_after",
            "away_score_after",
            "home_team_id",
            "away_team_id",
        ]
    ].copy()
    merged = shots.merge(
        pbp_subset,
        how="left",
        left_on=["GAME_ID", "GAME_EVENT_ID"],
        right_on=["gameId", "actionNumber"],
        suffixes=("", "_pbp"),
        indicator=True,
    )
    matched = merged["_merge"].eq("both")
    unmatched = merged[~matched].copy()
    unmatched_out = pd.DataFrame(
        {
            "season": season,
            "season_type": season_type,
            "game_id": unmatched["GAME_ID"],
            "event_id": unmatched["GAME_EVENT_ID"],
            "player": unmatched["PLAYER_NAME"],
            "period": pd.to_numeric(unmatched["PERIOD"], errors="coerce"),
            "clock": shot_clock(unmatched["MINUTES_REMAINING"], unmatched["SECONDS_REMAINING"]),
            "shot_type": unmatched["SHOT_TYPE"],
            "action_type": unmatched["ACTION_TYPE"],
            "shot_distance": pd.to_numeric(unmatched["SHOT_DISTANCE"], errors="coerce"),
            "shot_made": unmatched["SHOT_MADE_FLAG"].eq("1"),
            "reason_unmatched": "No PBP event matched GAME_ID + GAME_EVENT_ID/actionNumber",
        }
    )

    canonical = merged[matched].copy()
    canonical["season"] = season
    canonical["season_type"] = season_type
    canonical["dataset_split"] = SEASON_SPLITS[season]
    canonical["game_date"] = normalize_game_date(canonical["GAME_DATE"])
    canonical["player_id"] = pd.to_numeric(canonical["PLAYER_ID"], errors="coerce").astype("Int64")
    canonical["team_id"] = pd.to_numeric(canonical["TEAM_ID"], errors="coerce").astype("Int64")
    canonical["home_team_id_num"] = pd.to_numeric(canonical["home_team_id"], errors="coerce").astype("Int64")
    canonical["away_team_id_num"] = pd.to_numeric(canonical["away_team_id"], errors="coerce").astype("Int64")
    canonical["is_home"] = canonical["location"].eq("h")
    canonical["opponent_team_id"] = canonical["away_team_id_num"].where(canonical["is_home"], canonical["home_team_id_num"])
    canonical["shooter_score_before"] = canonical["home_score_before"].where(canonical["is_home"], canonical["away_score_before"])
    canonical["opponent_score_before"] = canonical["away_score_before"].where(canonical["is_home"], canonical["home_score_before"])
    canonical["score_diff_before"] = canonical["shooter_score_before"] - canonical["opponent_score_before"]
    canonical["abs_score_diff_before"] = canonical["score_diff_before"].abs()
    canonical["game_state_before"] = "tied"
    canonical.loc[canonical["score_diff_before"] > 0, "game_state_before"] = "leading"
    canonical.loc[canonical["score_diff_before"] < 0, "game_state_before"] = "trailing"
    canonical["is_overtime"] = canonical["period_num"] > 4
    canonical["is_clutch"] = (
        ((canonical["period_num"] == 4) & (canonical["seconds_remaining_period"] <= 300))
        | canonical["is_overtime"]
    ) & (canonical["abs_score_diff_before"] <= 5)
    canonical["clutch_time_bucket"] = ""
    canonical.loc[canonical["is_clutch"], "clutch_time_bucket"] = time_bucket(canonical.loc[canonical["is_clutch"], "seconds_remaining_period"])

    canonical["pbp_shot_value_mismatch"] = (
        canonical["pbp_shot_value"].notna()
        & canonical["pbp_shot_value"].ne(0)
        & canonical["pbp_shot_value"].ne(canonical["shot_value"])
    )
    made = canonical["shot_made"]
    shooter_delta = (
        canonical["home_score_after"].where(canonical["is_home"], canonical["away_score_after"])
        - canonical["shooter_score_before"]
    )
    opponent_delta = (
        canonical["away_score_after"].where(canonical["is_home"], canonical["home_score_after"])
        - canonical["opponent_score_before"]
    )
    made_reconcile = made & shooter_delta.eq(canonical["shot_value"]) & opponent_delta.eq(0)
    miss_reconcile = (~made) & shooter_delta.eq(0) & opponent_delta.eq(0)

    out = pd.DataFrame(
        {
            "season": canonical["season"],
            "season_type": canonical["season_type"],
            "dataset_split": canonical["dataset_split"],
            "game_id": canonical["game_id"],
            "event_id": canonical["event_id"],
            "game_date": canonical["game_date"],
            "player_id": canonical["player_id"],
            "player_name": canonical["PLAYER_NAME"],
            "team_id": canonical["team_id"],
            "team_name": canonical["TEAM_NAME"],
            "opponent_team_id": canonical["opponent_team_id"].astype("Int64"),
            "home_team_id": canonical["home_team_id_num"],
            "away_team_id": canonical["away_team_id_num"],
            "is_home": canonical["is_home"],
            "shot_made": canonical["shot_made"],
            "shot_value": canonical["shot_value"].astype("int8"),
            "points_scored": canonical["points_scored"].astype("int8"),
            "shot_type": canonical["SHOT_TYPE"],
            "action_type": canonical["ACTION_TYPE"],
            "shot_distance": pd.to_numeric(canonical["SHOT_DISTANCE"], errors="coerce").astype("Int64"),
            "loc_x": pd.to_numeric(canonical["LOC_X"], errors="coerce").astype("Int64"),
            "loc_y": pd.to_numeric(canonical["LOC_Y"], errors="coerce").astype("Int64"),
            "shot_zone_basic": canonical["SHOT_ZONE_BASIC"],
            "shot_zone_area": canonical["SHOT_ZONE_AREA"],
            "shot_zone_range": canonical["SHOT_ZONE_RANGE"],
            "period": canonical["period_num"].astype("int16"),
            "clock": canonical["clock"],
            "seconds_remaining_period": canonical["seconds_remaining_period"].astype("int16"),
            "is_overtime": canonical["is_overtime"],
            "home_score_before": canonical["home_score_before"].astype("int16"),
            "away_score_before": canonical["away_score_before"].astype("int16"),
            "shooter_score_before": canonical["shooter_score_before"].astype("int16"),
            "opponent_score_before": canonical["opponent_score_before"].astype("int16"),
            "score_diff_before": canonical["score_diff_before"].astype("int16"),
            "game_state_before": canonical["game_state_before"],
            "abs_score_diff_before": canonical["abs_score_diff_before"].astype("int16"),
            "is_clutch": canonical["is_clutch"],
            "clutch_time_bucket": canonical["clutch_time_bucket"],
        }
    )
    validation = {
        "season": season,
        "season_type": season_type,
        "shot_rows": int(len(shots)),
        "canonical_rows": int(len(out)),
        "unmatched_rows": int(len(unmatched_out)),
        "match_rate_pct": round(len(out) / len(shots) * 100, 6) if len(shots) else 0,
        "pbp_shot_value_disagreements": int(canonical["pbp_shot_value_mismatch"].sum()),
        "made_fg_score_reconcile_failures": int((made & ~made_reconcile).sum()),
        "miss_fg_score_reconcile_failures": int(((~made) & ~miss_reconcile).sum()),
        **score_metrics,
    }
    return out[CANONICAL_COLUMNS], unmatched_out, validation


def validate_dataset(shots: pd.DataFrame, unmatched: pd.DataFrame, pair_validations: list[dict[str, Any]]) -> dict[str, Any]:
    identity_cols = ["season", "season_type", "game_id", "event_id"]
    duplicate_identity = int(shots.duplicated(identity_cols).sum())
    shot_value_invalid = int((~shots["shot_value"].isin([2, 3])).sum())
    points_invalid = int(
        ((shots["shot_made"] & shots["points_scored"].ne(shots["shot_value"])) | (~shots["shot_made"] & shots["points_scored"].ne(0))).sum()
    )
    coord_extreme = int(((shots["loc_x"].abs() > 300) | (shots["loc_y"].abs() > 500)).sum())
    score_negative = int(((shots["home_score_before"] < 0) | (shots["away_score_before"] < 0)).sum())
    diff_invalid = int((shots["score_diff_before"] != (shots["shooter_score_before"] - shots["opponent_score_before"])).sum())
    map_home = shots["is_home"] & (
        (shots["shooter_score_before"] != shots["home_score_before"])
        | (shots["opponent_score_before"] != shots["away_score_before"])
    )
    map_away = (~shots["is_home"]) & (
        (shots["shooter_score_before"] != shots["away_score_before"])
        | (shots["opponent_score_before"] != shots["home_score_before"])
    )
    score_map_invalid = int((map_home | map_away).sum())
    clutch_rows = shots[shots["is_clutch"]]
    clutch_invalid = int(
        (
            ~(
                (((clutch_rows["period"] == 4) & (clutch_rows["seconds_remaining_period"] <= 300)) | (clutch_rows["period"] > 4))
                & (clutch_rows["abs_score_diff_before"] <= 5)
            )
        ).sum()
    )
    split_by_season = shots.groupby("season")["dataset_split"].nunique()
    split_invalid = int((split_by_season > 1).sum())
    critical_null_columns = [
        "season",
        "season_type",
        "dataset_split",
        "game_id",
        "event_id",
        "player_id",
        "team_id",
        "home_team_id",
        "away_team_id",
        "shot_value",
        "home_score_before",
        "away_score_before",
        "score_diff_before",
    ]
    critical_nulls = {col: int(shots[col].isna().sum() + shots[col].eq("").sum() if shots[col].dtype == "object" else shots[col].isna().sum()) for col in critical_null_columns}
    hard_failures = {
        "duplicate_identity": duplicate_identity,
        "shot_value_invalid": shot_value_invalid,
        "points_invalid": points_invalid,
        "score_negative": score_negative,
        "score_diff_invalid": diff_invalid,
        "score_mapping_invalid": score_map_invalid,
        "clutch_invalid": clutch_invalid,
        "split_invalid": split_invalid,
        "critical_nulls": critical_nulls,
    }
    passed = (
        duplicate_identity == 0
        and shot_value_invalid == 0
        and points_invalid == 0
        and score_negative == 0
        and diff_invalid == 0
        and score_map_invalid == 0
        and clutch_invalid == 0
        and split_invalid == 0
        and all(v == 0 for v in critical_nulls.values())
    )
    return {
        "passed": passed,
        "hard_failures": hard_failures,
        "coordinate_extreme_rows_report_only": coord_extreme,
        "pair_validations": pair_validations,
        "total_unmatched_rows": int(len(unmatched)),
        "match_rate_overall_pct": round(len(shots) / (len(shots) + len(unmatched)) * 100, 6),
        "pbp_shot_value_disagreements": int(sum(v["pbp_shot_value_disagreements"] for v in pair_validations)),
        "made_fg_score_reconcile_failures": int(sum(v["made_fg_score_reconcile_failures"] for v in pair_validations)),
        "miss_fg_score_reconcile_failures": int(sum(v["miss_fg_score_reconcile_failures"] for v in pair_validations)),
        "score_decrease_events": int(sum(v["score_decrease_events"] for v in pair_validations)),
    }


def feature_policy() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "identifier_columns": ["season", "season_type", "game_id", "event_id", "game_date", "player_id", "team_id"],
        "safe_pre_shot_candidate_features": [
            "season",
            "season_type",
            "period",
            "seconds_remaining_period",
            "is_overtime",
            "player_id",
            "team_id",
            "opponent_team_id",
            "is_home",
            "shot_value",
            "shot_type",
            "action_type",
            "shot_distance",
            "loc_x",
            "loc_y",
            "shot_zone_basic",
            "shot_zone_area",
            "shot_zone_range",
            "home_score_before",
            "away_score_before",
            "shooter_score_before",
            "opponent_score_before",
            "score_diff_before",
            "game_state_before",
            "abs_score_diff_before",
            "is_clutch",
            "clutch_time_bucket",
        ],
        "target_columns": ["shot_made", "points_scored"],
        "forbidden_leakage_columns": [
            "shot_made",
            "points_scored",
            "post-shot score fields",
            "pbp description",
            "shotResult",
            "EVENT_TYPE",
            "scoreHome",
            "scoreAway",
            "home_score_after",
            "away_score_after",
        ],
        "notes": "Future modeling code must explicitly separate pre-shot features from labels/outcomes.",
    }


def markdown_table(df: pd.DataFrame) -> str:
    rows = [list(df.columns)] + df.astype(str).values.tolist()
    widths = [max(len(str(row[i])) for row in rows) for i in range(len(rows[0]))]
    header = "| " + " | ".join(str(rows[0][i]).ljust(widths[i]) for i in range(len(widths))) + " |"
    sep = "| " + " | ".join("-" * widths[i] for i in range(len(widths))) + " |"
    body = ["| " + " | ".join(str(row[i]).ljust(widths[i]) for i in range(len(widths))) + " |" for row in rows[1:]]
    return "\n".join([header, sep, *body])


def summarize(shots: pd.DataFrame, validation: dict[str, Any]) -> dict[str, Any]:
    rows_per_season = shots.groupby("season").size().astype(int).to_dict()
    rows_per_season_type = shots.groupby("season_type").size().astype(int).to_dict()
    clutch = shots[shots["is_clutch"]].groupby(["season", "season_type"]).size().astype(int)
    split_counts = shots.groupby("dataset_split").size().astype(int).to_dict()
    return {
        "generation_timestamp": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "pipeline_version": PIPELINE_VERSION,
        "schema_version": SCHEMA_VERSION,
        "source_seasons": sorted(shots["season"].unique().tolist()),
        "source_season_types": sorted(shots["season_type"].unique().tolist()),
        "total_rows": int(len(shots)),
        "total_games": int(shots["game_id"].nunique()),
        "total_players": int(shots["player_id"].nunique()),
        "total_teams": int(shots["team_id"].nunique()),
        "rows_per_season": rows_per_season,
        "rows_per_season_type": rows_per_season_type,
        "clutch_rows_per_season_type": {f"{k[0]} {k[1]}": int(v) for k, v in clutch.items()},
        "dataset_split_counts": split_counts,
        "match_rates": {f'{v["season"]} {v["season_type"]}': v["match_rate_pct"] for v in validation["pair_validations"]},
        "unmatched_count": validation["total_unmatched_rows"],
        "canonical_column_schema": {col: str(dtype) for col, dtype in shots.dtypes.items()},
        "validation": validation,
    }


def write_reports(root: Path, shots: pd.DataFrame, unmatched: pd.DataFrame, validation: dict[str, Any], summary: dict[str, Any]) -> None:
    report_dir = root / "reports" / "data-pipeline"
    report_dir.mkdir(parents=True, exist_ok=True)
    clutch_counts = shots[shots["is_clutch"]].groupby(["season", "season_type"]).size().reset_index(name="rows")
    match_rows = pd.DataFrame(validation["pair_validations"])[["season", "season_type", "shot_rows", "canonical_rows", "unmatched_rows", "match_rate_pct"]]

    report = [
        "# Canonical Dataset Report",
        "",
        "## Methodology",
        "The pipeline reads immutable compressed raw archives from `data/raw/`, reconstructs running scores from full Play-by-Play V3 event streams, matches Shot Detail rows by `GAME_ID + GAME_EVENT_ID` to PBP `gameId + actionNumber`, and writes generated Parquet outputs under `data/processed/` and `data/interim/`.",
        "",
        "## Output Files",
        "- `data/processed/shots_enriched.parquet`",
        "- `data/processed/clutch_shots.parquet`",
        "- `data/interim/unmatched_shots.parquet`",
        "- `data/metadata/canonical_dataset_summary.json`",
        "- `data/metadata/feature_policy.json`",
        "",
        "## Score Reconstruction",
        "PBP events are sorted by game and action number. The current home/away score is captured before each event, then updated from available post-event score fields. This includes free throws and other scoring events, not only field goals.",
        "",
        "## Clutch Definition",
        "`is_clutch` is true for Q4 with 5:00 or less remaining or any overtime period, with absolute pre-shot score differential <= 5.",
        "",
        "## Split Methodology",
        "TRAIN: 2020-21 through 2023-24; VALIDATION: 2024-25; TEST: 2025-26. Both regular season and playoff rows are preserved with `season_type`.",
        "",
        "## Match Rates",
        markdown_table(match_rows),
        "",
        "## Clutch Counts",
        markdown_table(clutch_counts),
        "",
        "## Leakage Considerations",
        "The canonical table contains labels/outcomes such as `shot_made` and `points_scored`; future modeling must exclude them from input features. Event descriptions, post-shot scores, and PBP shot result fields are intentionally not included in the canonical feature table.",
        "",
        "## Known Limitations",
        f"{len(unmatched)} Shot Detail events are unmatched and excluded from canonical rows requiring verified pre-shot game state.",
    ]
    (report_dir / "CANONICAL_DATASET_REPORT.md").write_text("\n".join(report), encoding="utf-8")

    validation_report = [
        "# Data Validation",
        "",
        f"Passed: `{validation['passed']}`",
        "",
        "## Hard Checks",
        "```json",
        json.dumps(validation["hard_failures"], indent=2),
        "```",
        "",
        f"Coordinate extreme rows, report only: `{validation['coordinate_extreme_rows_report_only']}`",
        f"Overall match rate: `{validation['match_rate_overall_pct']}%`",
        f"Unmatched rows: `{validation['total_unmatched_rows']}`",
        f"PBP shot value disagreements: `{validation['pbp_shot_value_disagreements']}`",
        f"Made FG score reconcile failures: `{validation['made_fg_score_reconcile_failures']}`",
        f"Miss FG score reconcile failures: `{validation['miss_fg_score_reconcile_failures']}`",
        f"Score decrease events: `{validation['score_decrease_events']}`",
        "",
        "## Pair Validation",
        markdown_table(pd.DataFrame(validation["pair_validations"])),
    ]
    (report_dir / "DATA_VALIDATION.md").write_text("\n".join(validation_report), encoding="utf-8")


def update_readme(root: Path) -> None:
    readme = root / "README.md"
    text = readme.read_text(encoding="utf-8") if readme.exists() else "# ML NBA Agent Workspace\n"
    addition = (
        "\n## Canonical Dataset Pipeline\n\n"
        "Install data dependencies:\n\n"
        "```powershell\n"
        "python -m pip install -r scripts/data/requirements.txt\n"
        "```\n\n"
        "Run the audit:\n\n"
        "```powershell\n"
        "python scripts/data/audit_nba_data.py\n"
        "```\n\n"
        "Build canonical processed datasets:\n\n"
        "```powershell\n"
        "python scripts/data/build_canonical_dataset.py\n"
        "```\n\n"
        "Outputs are written to `data/processed/`, `data/interim/`, `data/metadata/`, and `reports/data-pipeline/`. Raw archives under `data/raw/` are immutable. The temporal split is train `2020-21` through `2023-24`, validation `2024-25`, and test `2025-26`.\n"
    )
    marker = "## Canonical Dataset Pipeline"
    if marker in text:
        text = text[: text.index(marker)].rstrip() + "\n" + addition
    else:
        text = text.rstrip() + "\n" + addition
    readme.write_text(text, encoding="utf-8")


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    processed_dir = root / "data" / "processed"
    interim_dir = root / "data" / "interim"
    metadata_dir = root / "data" / "metadata"
    processed_dir.mkdir(parents=True, exist_ok=True)
    interim_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    archives = discover_archives(root)
    all_shots: list[pd.DataFrame] = []
    all_unmatched: list[pd.DataFrame] = []
    pair_validations: list[dict[str, Any]] = []
    for season in sorted(SEASON_SPLITS):
        for season_type in ["regular_season", "playoffs"]:
            shot_path = archives.get(("shots", season, season_type))
            pbp_path = archives.get(("play_by_play", season, season_type))
            if shot_path is None or pbp_path is None:
                raise SystemExit(f"Missing source archives for {season} {season_type}")
            shots, unmatched, validation = process_pair(season, season_type, shot_path, pbp_path)
            all_shots.append(shots)
            all_unmatched.append(unmatched)
            pair_validations.append(validation)

    print("Combining and validating...")
    shots = pd.concat(all_shots, ignore_index=True)
    unmatched = pd.concat(all_unmatched, ignore_index=True)
    validation = validate_dataset(shots, unmatched, pair_validations)
    summary = summarize(shots, validation)

    print("Writing Parquet outputs...")
    shots.to_parquet(processed_dir / "shots_enriched.parquet", index=False, engine="pyarrow")
    shots[shots["is_clutch"]].to_parquet(processed_dir / "clutch_shots.parquet", index=False, engine="pyarrow")
    unmatched.to_parquet(interim_dir / "unmatched_shots.parquet", index=False, engine="pyarrow")
    (metadata_dir / "canonical_dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (metadata_dir / "feature_policy.json").write_text(json.dumps(feature_policy(), indent=2), encoding="utf-8")
    write_reports(root, shots, unmatched, validation, summary)
    update_readme(root)

    if not validation["passed"]:
        raise SystemExit("Canonical dataset validation failed; see reports/data-pipeline/DATA_VALIDATION.md")
    print("Canonical dataset build complete")


if __name__ == "__main__":
    main()
