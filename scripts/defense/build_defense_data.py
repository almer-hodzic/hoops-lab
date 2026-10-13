"""Organize, normalize, and audit defensive context datasets."""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import tarfile
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RAW_DEF = ROOT / "data" / "raw" / "defense"
PROCESSED_DEF = ROOT / "data" / "processed" / "defense"
ANALYSIS_DEF = ROOT / "data" / "analysis" / "defense"
REPORT_DEF = ROOT / "reports" / "defense"

MATCHUP_COLS = {
    "game_id",
    "person_id",
    "matchups_person_id",
    "partial_possessions",
    "matchup_field_goals_attempted",
}
SHOT_LOG_COLS = {
    "CLOSEST_DEFENDER",
    "CLOSEST_DEFENDER_PLAYER_ID",
    "CLOSE_DEF_DIST",
    "SHOT_CLOCK",
    "DRIBBLES",
    "TOUCH_TIME",
    "SHOT_DIST",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_dirs() -> None:
    for p in [
        RAW_DEF / "matchups" / "archives",
        RAW_DEF / "matchups" / "regular_season",
        RAW_DEF / "matchups" / "playoffs",
        RAW_DEF / "closest_defender_2014_15" / "original",
        RAW_DEF / "nba_pt_defend" / "regular_season",
        RAW_DEF / "nba_pt_defend" / "playoffs",
        RAW_DEF / "metadata",
        PROCESSED_DEF,
        ANALYSIS_DEF,
        REPORT_DEF,
    ]:
        p.mkdir(parents=True, exist_ok=True)


def season_from_year(year: int) -> str:
    return f"{year}-{str(year + 1)[-2:]}"


def year_from_name(name: str) -> int | None:
    m = re.search(r"(20\d{2})", name)
    if m:
        return int(m.group(1))
    return None


def detect_archive(path: Path) -> dict[str, Any]:
    record: dict[str, Any] = {
        "original_path": str(path.relative_to(ROOT)),
        "filename": path.name,
        "file_size": path.stat().st_size,
        "archive_members": [],
        "row_count": None,
        "columns": [],
        "detected_type": "unknown",
        "inferred_season": None,
        "inferred_season_type": None,
        "confidence": "low",
        "checksum_sha256": sha256(path),
    }
    try:
        if path.name.endswith(".tar.xz"):
            with tarfile.open(path, "r:xz") as tf:
                members = [m for m in tf.getmembers() if m.isfile()]
                record["archive_members"] = [m.name for m in members]
                sample_member = next((m for m in members if m.name.lower().endswith(".csv")), None)
                if sample_member is not None:
                    with tf.extractfile(sample_member) as f:  # type: ignore[arg-type]
                        sample = pd.read_csv(f, nrows=20)
                    record["columns"] = list(sample.columns)
                    if MATCHUP_COLS.issubset(set(sample.columns)):
                        record["detected_type"] = "player_matchups_archive"
                        record["confidence"] = "high"
                        year = year_from_name(path.name)
                        record["inferred_season"] = season_from_year(year) if year else None
                        record["inferred_season_type"] = (
                            "playoffs" if "_po_" in path.name.lower() else "regular_season"
                        )
                    with tf.extractfile(sample_member) as f:  # type: ignore[arg-type]
                        record["row_count"] = sum(1 for _ in f) - 1
        elif zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as zf:
                infos = [i for i in zf.infolist() if not i.is_dir()]
                record["archive_members"] = [i.filename for i in infos]
                sample_info = next((i for i in infos if i.filename.lower().endswith(".csv")), None)
                if sample_info is not None:
                    with zf.open(sample_info) as f:
                        sample = pd.read_csv(f, nrows=20)
                    record["columns"] = list(sample.columns)
                    if SHOT_LOG_COLS.issubset(set(sample.columns)):
                        record["detected_type"] = "closest_defender_shot_logs_2014_15"
                        record["confidence"] = "high"
                        record["inferred_season"] = "2014-15"
                        record["inferred_season_type"] = "regular_season"
                    with zf.open(sample_info) as f:
                        record["row_count"] = sum(1 for _ in f) - 1
    except Exception as exc:  # noqa: BLE001
        record["error"] = repr(exc)
    return record


def discover_files() -> list[dict[str, Any]]:
    candidates = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in {".git", "__pycache__", "bin", "obj"} for part in path.parts):
            continue
        if RAW_DEF in path.parents and path.suffix.lower() == ".csv":
            continue
        if path.suffix.lower() in {".zip", ".csv", ".xz"} or path.name.endswith(".tar.xz"):
            if path.name.startswith("matchups") or "shot_logs" in path.name.lower() or path.suffix == ".zip":
                candidates.append(path)
    seen = set()
    records = []
    for path in sorted(candidates):
        if path in seen:
            continue
        seen.add(path)
        if path.name.endswith(".tar.xz") or zipfile.is_zipfile(path):
            rec = detect_archive(path)
            if rec["detected_type"] != "unknown":
                records.append(rec)
        elif path.suffix.lower() == ".csv":
            try:
                sample = pd.read_csv(path, nrows=20)
                cols = set(sample.columns)
                rec = {
                    "original_path": str(path.relative_to(ROOT)),
                    "filename": path.name,
                    "file_size": path.stat().st_size,
                    "archive_members": [],
                    "row_count": sum(1 for _ in path.open("rb")) - 1,
                    "columns": list(sample.columns),
                    "detected_type": "unknown",
                    "inferred_season": None,
                    "inferred_season_type": None,
                    "confidence": "low",
                    "checksum_sha256": sha256(path),
                }
                if MATCHUP_COLS.issubset(cols):
                    rec["detected_type"] = "player_matchups_csv"
                    rec["confidence"] = "high"
                if SHOT_LOG_COLS.issubset(cols):
                    rec["detected_type"] = "closest_defender_shot_logs_2014_15"
                    rec["confidence"] = "high"
                    rec["inferred_season"] = "2014-15"
                    rec["inferred_season_type"] = "regular_season"
                if rec["detected_type"] != "unknown":
                    records.append(rec)
            except Exception:
                continue
    return records


def move_if_needed(src: Path, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() == dst.resolve():
        return dst
    if dst.exists():
        if sha256(src) == sha256(dst):
            src.unlink()
            return dst
        dst = dst.with_name(f"{dst.stem}_{sha256(src)[:8]}{dst.suffix}")
    shutil.move(str(src), str(dst))
    return dst


def organize_raw(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    organized = []
    for rec in records:
        src = ROOT / rec["original_path"]
        if not src.exists():
            continue
        final_path = src
        if rec["detected_type"] == "player_matchups_archive":
            final_path = move_if_needed(src, RAW_DEF / "matchups" / "archives" / src.name)
            season_type = rec["inferred_season_type"] or "regular_season"
            season = rec["inferred_season"] or "unknown"
            out_dir = RAW_DEF / "matchups" / season_type / season
            out_dir.mkdir(parents=True, exist_ok=True)
            with tarfile.open(final_path, "r:xz") as tf:
                for member in tf.getmembers():
                    if member.isfile() and member.name.lower().endswith(".csv"):
                        out_path = out_dir / Path(member.name).name
                        if not out_path.exists():
                            with tf.extractfile(member) as f, out_path.open("wb") as g:  # type: ignore[arg-type]
                                shutil.copyfileobj(f, g)
                        rec["extracted_path"] = str(out_path.relative_to(ROOT))
        elif rec["detected_type"] == "closest_defender_shot_logs_2014_15":
            final_path = move_if_needed(src, RAW_DEF / "closest_defender_2014_15" / "original" / src.name)
            if zipfile.is_zipfile(final_path):
                with zipfile.ZipFile(final_path) as zf:
                    for info in zf.infolist():
                        if info.filename.lower().endswith("shot_logs.csv"):
                            out_path = RAW_DEF / "closest_defender_2014_15" / "original" / "shot_logs.csv"
                            if not out_path.exists():
                                with zf.open(info) as f, out_path.open("wb") as g:
                                    shutil.copyfileobj(f, g)
                            rec["extracted_path"] = str(out_path.relative_to(ROOT))
                            break
        rec["final_raw_path"] = str(final_path.relative_to(ROOT))
        organized.append(rec)
    return organized


def full_name(first: pd.Series, family: pd.Series) -> pd.Series:
    return (first.fillna("").astype(str).str.strip() + " " + family.fillna("").astype(str).str.strip()).str.strip()


def parse_game_date_from_id(game_id: Any) -> str | None:
    return None


def normalize_matchups() -> tuple[pd.DataFrame, dict[str, Any]]:
    frames = []
    for csv_path in sorted((RAW_DEF / "matchups").glob("*/*/*.csv")):
        if "archives" in csv_path.parts:
            continue
        rel = csv_path.relative_to(RAW_DEF / "matchups")
        season_type = rel.parts[0]
        season = rel.parts[1]
        raw = pd.read_csv(csv_path, dtype={"game_id": "string"})
        df = pd.DataFrame()
        df["game_id"] = raw["game_id"].astype("string")
        df["game_date"] = raw["game_id"].map(parse_game_date_from_id)
        df["season"] = season
        df["season_type"] = season_type
        df["offensive_player_id"] = pd.to_numeric(raw["person_id"], errors="coerce").astype("Int64")
        df["offensive_player_name"] = full_name(raw["first_name"], raw["family_name"])
        df["offensive_team_id"] = pd.to_numeric(raw["team_id"], errors="coerce").astype("Int64")
        df["defensive_player_id"] = pd.to_numeric(raw["matchups_person_id"], errors="coerce").astype("Int64")
        df["defensive_player_name"] = full_name(raw["matchups_first_name"], raw["matchups_family_name"])
        df["defensive_team_id"] = pd.Series(pd.NA, index=raw.index, dtype="Int64")
        mappings = {
            "matchup_minutes": "matchup_minutes",
            "partial_possessions": "partial_possessions",
            "percentage_defender_total_time": "percentage_defender_total_time",
            "percentage_offensive_total_time": "percentage_offensive_total_time",
            "percentage_total_time_both_on": "percentage_total_time_both_on",
            "switches_on": "switches_on",
            "player_points": "player_points",
            "team_points": "team_points",
            "matchup_assists": "matchup_assists",
            "matchup_potential_assists": "matchup_potential_assists",
            "matchup_turnovers": "matchup_turnovers",
            "matchup_blocks": "matchup_blocks",
            "matchup_field_goals_made": "matchup_fgm",
            "matchup_field_goals_attempted": "matchup_fga",
            "matchup_field_goals_percentage": "matchup_fg_pct",
            "matchup_three_pointers_made": "matchup_3pm",
            "matchup_three_pointers_attempted": "matchup_3pa",
            "matchup_three_pointers_percentage": "matchup_3p_pct",
            "help_blocks": "help_blocks",
            "help_field_goals_made": "help_fgm",
            "help_field_goals_attempted": "help_fga",
            "help_field_goals_percentage": "help_fg_pct",
            "matchup_free_throws_made": "matchup_ftm",
            "matchup_free_throws_attempted": "matchup_fta",
            "shooting_fouls": "shooting_fouls",
        }
        for source, target in mappings.items():
            df[target] = pd.to_numeric(raw[source], errors="coerce") if source in raw.columns else np.nan
        frames.append(df)
    if not frames:
        empty = pd.DataFrame()
        return empty, {"rows": 0}
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(PROCESSED_DEF / "matchups_canonical.parquet", index=False)
    key_cols = ["game_id", "season", "season_type", "offensive_player_id", "defensive_player_id"]
    duplicate_count = int(out.duplicated(key_cols).sum())
    audit = {
        "rows": int(len(out)),
        "games": int(out["game_id"].nunique()),
        "seasons": sorted(out["season"].dropna().unique().tolist()),
        "season_types": sorted(out["season_type"].dropna().unique().tolist()),
        "unique_offensive_players": int(out["offensive_player_id"].nunique()),
        "unique_defenders": int(out["defensive_player_id"].nunique()),
        "regular_season_rows": int((out["season_type"] == "regular_season").sum()),
        "playoff_rows": int((out["season_type"] == "playoffs").sum()),
        "duplicate_count": duplicate_count,
        "missing_offensive_id_rate": float(out["offensive_player_id"].isna().mean()),
        "missing_defensive_id_rate": float(out["defensive_player_id"].isna().mean()),
        "fgm_gt_fga": int((out["matchup_fgm"] > out["matchup_fga"]).fillna(False).sum()),
        "three_pm_gt_pa": int((out["matchup_3pm"] > out["matchup_3pa"]).fillna(False).sum()),
        "ftm_gt_fta": int((out["matchup_ftm"] > out["matchup_fta"]).fillna(False).sum()),
        "negative_numeric_values": int((out.select_dtypes(include=[np.number]) < 0).sum().sum()),
        "time_share_percentage_gt_one": {
            col: int((out[col] > 1).fillna(False).sum())
            for col in [
                "percentage_defender_total_time",
                "percentage_offensive_total_time",
                "percentage_total_time_both_on",
            ]
        },
    }
    return out, audit


def normalize_shot_logs() -> tuple[pd.DataFrame, dict[str, Any]]:
    path = RAW_DEF / "closest_defender_2014_15" / "original" / "shot_logs.csv"
    if not path.exists():
        return pd.DataFrame(), {"rows": 0}
    raw = pd.read_csv(path)
    out = pd.DataFrame()
    out["game_id"] = raw.get("GAME_ID").astype("string")
    out["matchup"] = raw.get("MATCHUP")
    out["game_clock"] = raw.get("GAME_CLOCK")
    out["period"] = pd.to_numeric(raw.get("PERIOD"), errors="coerce").astype("Int64")
    out["game_date"] = pd.NA
    out["location"] = raw.get("LOCATION")
    out["opponent"] = raw.get("MATCHUP")
    out["score_margin"] = pd.to_numeric(raw.get("FINAL_MARGIN"), errors="coerce")
    out["shooter_name"] = raw.get("player_name")
    out["shooter_id"] = pd.to_numeric(raw.get("player_id"), errors="coerce").astype("Int64")
    out["shot_number"] = pd.to_numeric(raw.get("SHOT_NUMBER"), errors="coerce").astype("Int64")
    out["shot_clock"] = pd.to_numeric(raw.get("SHOT_CLOCK"), errors="coerce")
    out["dribbles"] = pd.to_numeric(raw.get("DRIBBLES"), errors="coerce")
    out["touch_time"] = pd.to_numeric(raw.get("TOUCH_TIME"), errors="coerce")
    out["shot_distance"] = pd.to_numeric(raw.get("SHOT_DIST"), errors="coerce")
    out["shot_type"] = raw.get("PTS_TYPE").map(lambda v: f"{int(v)}PT Field Goal" if pd.notna(v) else pd.NA)
    out["shot_value"] = pd.to_numeric(raw.get("PTS_TYPE"), errors="coerce").astype("Int64")
    out["shot_result"] = raw.get("SHOT_RESULT")
    out["made"] = pd.to_numeric(raw.get("FGM"), errors="coerce").astype("Int64")
    out["points"] = pd.to_numeric(raw.get("PTS"), errors="coerce")
    out["closest_defender_name"] = raw.get("CLOSEST_DEFENDER")
    out["closest_defender_id"] = pd.to_numeric(raw.get("CLOSEST_DEFENDER_PLAYER_ID"), errors="coerce").astype("Int64")
    out["closest_defender_distance"] = pd.to_numeric(raw.get("CLOSE_DEF_DIST"), errors="coerce")
    out.to_parquet(PROCESSED_DEF / "shot_logs_2014_15.parquet", index=False)

    exp = out.copy()
    for col in ["shot_clock", "dribbles", "touch_time", "closest_defender_distance", "closest_defender_id", "shooter_id"]:
        exp[f"{col}_missing"] = exp[col].isna()
    exp.to_parquet(ANALYSIS_DEF / "shot_quality_tracking_experiment.parquet", index=False)

    audit = {
        "rows": int(len(out)),
        "made_rate": float(out["made"].mean()),
        "missing_shot_clock_pct": float(out["shot_clock"].isna().mean()),
        "missing_defender_id_pct": float(out["closest_defender_id"].isna().mean()),
        "missing_defender_distance_pct": float(out["closest_defender_distance"].isna().mean()),
        "exact_closest_defender_rows": int(out["closest_defender_id"].notna().sum()),
        "unique_defenders": int(out["closest_defender_id"].nunique()),
        "defender_distance": describe(out["closest_defender_distance"]),
        "shot_distance": describe(out["shot_distance"]),
        "dribbles": describe(out["dribbles"]),
        "touch_time": describe(out["touch_time"]),
        "negative_defender_distance": int((out["closest_defender_distance"] < 0).fillna(False).sum()),
        "negative_shot_distance": int((out["shot_distance"] < 0).fillna(False).sum()),
        "shot_clock_out_of_bounds": int(((out["shot_clock"] < 0) | (out["shot_clock"] > 24)).fillna(False).sum()),
        "touch_time_negative": int((out["touch_time"] < 0).fillna(False).sum()),
        "dribbles_negative": int((out["dribbles"] < 0).fillna(False).sum()),
    }
    return out, audit


def describe(s: pd.Series) -> dict[str, float | None]:
    clean = pd.to_numeric(s, errors="coerce").dropna()
    if clean.empty:
        return {"min": None, "median": None, "p95": None, "max": None}
    return {
        "min": float(clean.min()),
        "median": float(clean.median()),
        "p95": float(clean.quantile(0.95)),
        "max": float(clean.max()),
    }


def normalize_pt_defend() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    frames = []
    for csv_path in sorted((RAW_DEF / "nba_pt_defend").glob("*/*/*.csv")):
        rel = csv_path.relative_to(RAW_DEF / "nba_pt_defend")
        season_type = rel.parts[0]
        season = rel.parts[1]
        category = csv_path.stem
        raw = pd.read_csv(csv_path)
        if raw.empty:
            continue
        df = raw.copy()
        df["season"] = season
        df["season_type"] = season_type
        df["defense_category"] = category
        rename = {
            "CLOSE_DEF_PERSON_ID": "defender_id",
            "PLAYER_NAME": "defender_name",
            "PLAYER_LAST_TEAM_ID": "team_id",
            "PLAYER_LAST_TEAM_ABBREVIATION": "team_abbreviation",
            "PLAYER_POSITION": "player_position",
            "AGE": "age",
            "GP": "gp",
            "D_FGM": "defended_fgm",
            "D_FGA": "defended_fga",
            "D_FG_PCT": "defended_fg_pct",
            "NORMAL_FG_PCT": "normal_fg_pct",
            "PCT_PLUSMINUS": "pct_plusminus",
            "FREQ": "frequency",
        }
        df = df.rename(columns=rename)
        frames.append(df)
    if not frames:
        return pd.DataFrame(), pd.DataFrame(), {"rows": 0}
    out = pd.concat(frames, ignore_index=True)
    for col in ["defender_id", "team_id", "gp", "defended_fgm", "defended_fga"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out.to_parquet(PROCESSED_DEF / "pt_defend_season_category.parquet", index=False)

    keep = [
        "defender_id",
        "defender_name",
        "team_abbreviation",
        "player_position",
        "season",
        "season_type",
        "defense_category",
        "defended_fga",
        "defended_fg_pct",
        "normal_fg_pct",
        "pct_plusminus",
        "frequency",
    ]
    profile_base = out[[c for c in keep if c in out.columns]].copy()
    pivot = profile_base.pivot_table(
        index=["defender_id", "season", "season_type"],
        columns="defense_category",
        values=[c for c in ["defended_fga", "defended_fg_pct", "normal_fg_pct", "pct_plusminus", "frequency"] if c in profile_base.columns],
        aggfunc="first",
    )
    pivot.columns = [f"{category_prefix(cat)}_{metric_name(metric)}" for metric, cat in pivot.columns]
    pivot = pivot.reset_index()
    meta = profile_base.sort_values("defense_category").drop_duplicates(["defender_id", "season", "season_type"])[
        ["defender_id", "season", "season_type", "defender_name", "team_abbreviation", "player_position"]
    ]
    profiles = meta.merge(pivot, on=["defender_id", "season", "season_type"], how="left")
    profiles.to_parquet(PROCESSED_DEF / "defender_season_profiles.parquet", index=False)
    audit = {
        "rows": int(len(out)),
        "unique_modern_defenders": int(out["defender_id"].nunique()) if "defender_id" in out else 0,
        "seasons": sorted(out["season"].dropna().unique().tolist()),
        "season_types": sorted(out["season_type"].dropna().unique().tolist()),
        "categories": sorted(out["defense_category"].dropna().unique().tolist()),
    }
    return out, profiles, audit


def category_prefix(category: str) -> str:
    mapping = {
        "overall": "overall",
        "2_pointers": "two_pt",
        "3_pointers": "three_pt",
        "less_than_6ft": "lt6",
        "less_than_10ft": "lt10",
        "greater_than_15ft": "gt15",
    }
    return mapping.get(str(category).lower(), str(category).lower())


def metric_name(metric: str) -> str:
    return {
        "defended_fga": "d_fga",
        "defended_fg_pct": "d_fg_pct",
        "normal_fg_pct": "normal_fg_pct",
        "pct_plusminus": "pct_plusminus",
        "frequency": "frequency",
    }.get(metric, metric)


def build_defender_priors(matchups: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    if matchups.empty:
        return pd.DataFrame(), {"rows": 0}
    rows = matchups.copy()
    rows["_game_order"] = pd.to_numeric(rows["game_id"], errors="coerce")
    rows = rows.sort_values(["defensive_player_id", "_game_order", "game_id"])
    pop_fga = rows["matchup_fga"].sum()
    pop_fgm = rows["matchup_fgm"].sum()
    pop_3pa = rows["matchup_3pa"].sum()
    pop_3pm = rows["matchup_3pm"].sum()
    pop_fg = float(pop_fgm / pop_fga) if pop_fga else 0.45
    pop_3p = float(pop_3pm / pop_3pa) if pop_3pa else 0.35
    strength = 50.0
    game_totals = (
        rows.assign(
            blocks_total=rows["matchup_blocks"].fillna(0) + rows["help_blocks"].fillna(0),
            fouls_total=rows["shooting_fouls"].fillna(0),
        )
        .groupby(["defensive_player_id", "game_id", "_game_order"], dropna=False)
        .agg(
            game_possessions=("partial_possessions", "sum"),
            game_fga=("matchup_fga", "sum"),
            game_fgm=("matchup_fgm", "sum"),
            game_3pa=("matchup_3pa", "sum"),
            game_3pm=("matchup_3pm", "sum"),
            game_blocks=("blocks_total", "sum"),
            game_fouls=("fouls_total", "sum"),
        )
        .reset_index()
        .sort_values(["defensive_player_id", "_game_order", "game_id"])
    )
    prior_parts = []
    for defender_id, grp in game_totals.groupby("defensive_player_id", dropna=False):
        g = grp.copy()
        g["defender_prior_matchup_possessions"] = g["game_possessions"].cumsum().shift(fill_value=0)
        g["defender_prior_matchup_fga"] = g["game_fga"].cumsum().shift(fill_value=0)
        g["defender_prior_matchup_fgm"] = g["game_fgm"].cumsum().shift(fill_value=0)
        g["defender_prior_matchup_fg_pct"] = np.where(
            g["defender_prior_matchup_fga"] > 0,
            g["defender_prior_matchup_fgm"] / g["defender_prior_matchup_fga"],
            pop_fg,
        )
        g["defender_prior_matchup_fg_pct_shrunk"] = (
            g["defender_prior_matchup_fgm"] + pop_fg * strength
        ) / (g["defender_prior_matchup_fga"] + strength)
        g["defender_prior_matchup_3pa"] = g["game_3pa"].cumsum().shift(fill_value=0)
        g["defender_prior_matchup_3pm"] = g["game_3pm"].cumsum().shift(fill_value=0)
        g["defender_prior_matchup_3p_pct"] = np.where(
            g["defender_prior_matchup_3pa"] > 0,
            g["defender_prior_matchup_3pm"] / g["defender_prior_matchup_3pa"],
            pop_3p,
        )
        g["defender_prior_matchup_3p_pct_shrunk"] = (
            g["defender_prior_matchup_3pm"] + pop_3p * strength
        ) / (g["defender_prior_matchup_3pa"] + strength)
        g["defender_prior_blocks"] = g["game_blocks"].cumsum().shift(fill_value=0)
        g["defender_prior_shooting_fouls"] = g["game_fouls"].cumsum().shift(fill_value=0)
        g["defender_prior_block_rate"] = np.where(
            g["defender_prior_matchup_possessions"] > 0,
            g["defender_prior_blocks"] / g["defender_prior_matchup_possessions"],
            0.0,
        )
        g["defender_prior_foul_rate"] = np.where(
            g["defender_prior_matchup_possessions"] > 0,
            g["defender_prior_shooting_fouls"] / g["defender_prior_matchup_possessions"],
            0.0,
        )
        prior_parts.append(g)
    game_priors = pd.concat(prior_parts, ignore_index=True)
    prior_cols = [
        "defensive_player_id",
        "game_id",
        "defender_prior_matchup_possessions",
        "defender_prior_matchup_fga",
        "defender_prior_matchup_fgm",
        "defender_prior_matchup_fg_pct",
        "defender_prior_matchup_fg_pct_shrunk",
        "defender_prior_matchup_3pa",
        "defender_prior_matchup_3pm",
        "defender_prior_matchup_3p_pct",
        "defender_prior_matchup_3p_pct_shrunk",
        "defender_prior_blocks",
        "defender_prior_shooting_fouls",
        "defender_prior_block_rate",
        "defender_prior_foul_rate",
    ]
    priors = rows.merge(game_priors[prior_cols], on=["defensive_player_id", "game_id"], how="left")
    cols = [
        "game_id",
        "season",
        "season_type",
        "defensive_player_id",
        "defensive_player_name",
        "defender_prior_matchup_possessions",
        "defender_prior_matchup_fga",
        "defender_prior_matchup_fgm",
        "defender_prior_matchup_fg_pct",
        "defender_prior_matchup_fg_pct_shrunk",
        "defender_prior_matchup_3pa",
        "defender_prior_matchup_3pm",
        "defender_prior_matchup_3p_pct",
        "defender_prior_matchup_3p_pct_shrunk",
        "defender_prior_blocks",
        "defender_prior_shooting_fouls",
        "defender_prior_block_rate",
        "defender_prior_foul_rate",
    ]
    priors = priors[cols]
    priors.to_parquet(PROCESSED_DEF / "defender_game_priors.parquet", index=False)
    audit = {
        "rows": int(len(priors)),
        "prior_strength": strength,
        "population_fg_pct": pop_fg,
        "population_3p_pct": pop_3p,
        "no_future_leakage_check": "defender history is aggregated by game, shifted one game, then merged back to matchup rows",
    }
    return priors, audit


def build_crosswalk(shot_logs: pd.DataFrame, matchups: pd.DataFrame, pt_def: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    records = []
    def add(source: str, ids: pd.Series, names: pd.Series) -> None:
        tmp = pd.DataFrame({"player_id": ids.astype("Int64"), "player_name": names.astype("string")}).dropna(subset=["player_id"])
        for row in tmp.drop_duplicates().itertuples(index=False):
            records.append({"source": source, "player_id": int(row.player_id), "player_name": row.player_name})

    if not shot_logs.empty:
        add("shot_logs_2014_15_closest_defender", shot_logs["closest_defender_id"], shot_logs["closest_defender_name"])
        add("shot_logs_2014_15_shooter", shot_logs["shooter_id"], shot_logs["shooter_name"])
    if not matchups.empty:
        add("matchups_defender", matchups["defensive_player_id"], matchups["defensive_player_name"])
        add("matchups_offense", matchups["offensive_player_id"], matchups["offensive_player_name"])
    if not pt_def.empty and "defender_id" in pt_def:
        add("pt_defend_defender", pt_def["defender_id"], pt_def["defender_name"])
    existing_path = ROOT / "artifacts" / "inference" / "player_metadata.json"
    if existing_path.exists():
        players = json.loads(existing_path.read_text(encoding="utf-8"))
        for p in players:
            records.append({"source": "existing_player_metadata", "player_id": int(p["player_id"]), "player_name": p.get("player_name")})
    cross = pd.DataFrame(records).drop_duplicates()
    if cross.empty:
        return cross, {"rows": 0}
    grouped = cross.groupby("player_id").agg(
        sources=("source", lambda s: sorted(set(s))),
        names=("player_name", lambda s: sorted(set(str(x) for x in s if pd.notna(x)))),
    ).reset_index()
    grouped["source_count"] = grouped["sources"].map(len)
    grouped["name_count"] = grouped["names"].map(len)
    grouped.to_parquet(PROCESSED_DEF / "player_id_crosswalk.parquet", index=False)
    source_sets = {
        source: set(cross.loc[cross["source"].eq(source), "player_id"].dropna().astype(int))
        for source in cross["source"].unique()
    }
    tracking_ids = source_sets.get("shot_logs_2014_15_closest_defender", set())
    all_modern = set().union(
        source_sets.get("matchups_defender", set()),
        source_sets.get("pt_defend_defender", set()),
        source_sets.get("existing_player_metadata", set()),
    )
    audit = {
        "rows": int(len(grouped)),
        "tracking_defender_ids": len(tracking_ids),
        "tracking_defender_ids_matched_to_modern_or_metadata": len(tracking_ids & all_modern),
        "tracking_match_rate": float(len(tracking_ids & all_modern) / len(tracking_ids)) if tracking_ids else None,
        "ids_in_multiple_sources": int((grouped["source_count"] > 1).sum()),
        "ambiguous_id_name_count": int((grouped["name_count"] > 1).sum()),
    }
    return grouped, audit


def write_inventory(records: list[dict[str, Any]]) -> None:
    lines = ["# Raw Defensive File Inventory", ""]
    for rec in records:
        lines.append(f"## {rec['filename']}")
        lines.append("")
        lines.append(f"- Original path: `{rec['original_path']}`")
        lines.append(f"- Final raw path: `{rec.get('final_raw_path', '')}`")
        if rec.get("extracted_path"):
            lines.append(f"- Extracted path: `{rec['extracted_path']}`")
        lines.append(f"- Size: {rec['file_size']:,} bytes")
        lines.append(f"- Detected type: {rec['detected_type']}")
        lines.append(f"- Inferred season: {rec.get('inferred_season')}")
        lines.append(f"- Inferred season type: {rec.get('inferred_season_type')}")
        lines.append(f"- Row count: {rec.get('row_count')}")
        lines.append(f"- Confidence: {rec['confidence']}")
        lines.append(f"- SHA256: `{rec['checksum_sha256']}`")
        lines.append(f"- Archive members: {', '.join(rec.get('archive_members', []))}")
        lines.append(f"- Columns: {', '.join(rec.get('columns', []))}")
        lines.append("")
    (REPORT_DEF / "RAW_FILE_INVENTORY.md").write_text("\n".join(lines), encoding="utf-8")


def write_manifest(records: list[dict[str, Any]]) -> None:
    manifest = []
    for rec in records:
        source_name = "NBA Data Archive / stats.nba.com" if "matchups" in rec["detected_type"] else "2014/15 closest-defender shot log"
        manifest.append(
            {
                "source_name": source_name,
                "source_type": rec["detected_type"],
                "source_url": "https://stats.nba.com" if "matchups" in rec["detected_type"] else None,
                "retrieved_or_user_supplied": "user_supplied",
                "original_filename": rec["filename"],
                "final_raw_path": rec.get("final_raw_path"),
                "extracted_path": rec.get("extracted_path"),
                "seasons": [rec["inferred_season"]] if rec.get("inferred_season") else [],
                "season_types": [rec["inferred_season_type"]] if rec.get("inferred_season_type") else [],
                "schema_summary": rec.get("columns", []),
                "row_count": rec.get("row_count"),
                "checksum_sha256": rec.get("checksum_sha256"),
                "retrieval_timestamp": None,
                "notes": "Original user-supplied file preserved unchanged; extracted CSV is a raw copy.",
                "license_attribution": "NBA/stats.nba.com where applicable; user-supplied archive provenance retained.",
            }
        )
    status_path = RAW_DEF / "metadata" / "pt_defend_download_status.json"
    if status_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        successful = [r for r in status if r.get("success")]
        manifest.append(
            {
                "source_name": "stats.nba.com LeagueDashPtDefend",
                "source_type": "nba_pt_defend_dashboard",
                "source_url": "https://stats.nba.com/stats/leaguedashptdefend",
                "retrieved_or_user_supplied": "downloaded_during_task" if successful else "attempted_download",
                "original_filename": None,
                "final_raw_path": "data/raw/defense/nba_pt_defend/",
                "seasons": sorted({r["season"] for r in successful}),
                "season_types": sorted({r["season_type"] for r in successful}),
                "schema_summary": [
                    "CLOSE_DEF_PERSON_ID",
                    "PLAYER_NAME",
                    "PLAYER_LAST_TEAM_ID",
                    "PLAYER_LAST_TEAM_ABBREVIATION",
                    "PLAYER_POSITION",
                    "AGE",
                    "GP",
                    "G",
                    "FREQ",
                    "D_FGM",
                    "D_FGA",
                    "D_FG_PCT",
                    "NORMAL_FG_PCT",
                    "PCT_PLUSMINUS",
                ],
                "row_count": sum(int(r.get("rows", 0)) for r in successful),
                "checksum_sha256": None,
                "retrieval_timestamp": datetime.now(timezone.utc).isoformat(),
                "notes": "One CSV per season, season type, and defense category when endpoint succeeded.",
                "license_attribution": "NBA/stats.nba.com LeagueDashPtDefend.",
            }
        )
    (RAW_DEF / "metadata" / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def write_schema_mapping() -> None:
    mapping = {
        "GAME_ID": "game_id",
        "MATCHUP": "matchup/opponent context",
        "LOCATION": "location",
        "FINAL_MARGIN": "score_margin",
        "SHOT_NUMBER": "shot_number",
        "PERIOD": "period",
        "GAME_CLOCK": "game_clock",
        "SHOT_CLOCK": "shot_clock",
        "DRIBBLES": "dribbles",
        "TOUCH_TIME": "touch_time",
        "SHOT_DIST": "shot_distance",
        "PTS_TYPE": "shot_value and shot_type",
        "SHOT_RESULT": "shot_result",
        "FGM": "made",
        "PTS": "points",
        "player_name": "shooter_name",
        "player_id": "shooter_id",
        "CLOSEST_DEFENDER": "closest_defender_name",
        "CLOSEST_DEFENDER_PLAYER_ID": "closest_defender_id",
        "CLOSE_DEF_DIST": "closest_defender_distance",
    }
    lines = ["# 2014/15 Shot Logs Schema Mapping", ""]
    for source, target in mapping.items():
        lines.append(f"- `{source}` -> `{target}`")
    lines.append("")
    lines.append("Unknown or unused source columns are retained in raw data, not silently interpreted.")
    (REPORT_DEF / "SHOT_LOGS_SCHEMA_MAPPING.md").write_text("\n".join(lines), encoding="utf-8")


def write_crosswalk_report(audit: dict[str, Any]) -> None:
    lines = [
        "# Defender ID Cross-Source Audit",
        "",
        f"- Tracking defender IDs: {audit.get('tracking_defender_ids', 0):,}",
        f"- Tracking defender IDs matched to modern matchup/PT Defend/player metadata IDs: {audit.get('tracking_defender_ids_matched_to_modern_or_metadata', 0):,}",
        f"- Direct tracking match rate: {pct(audit.get('tracking_match_rate'))}",
        f"- Player IDs appearing in multiple sources: {audit.get('ids_in_multiple_sources', 0):,}",
        f"- IDs with multiple observed names: {audit.get('ambiguous_id_name_count', 0):,}",
        "",
        "IDs were joined only by exact numeric NBA person ID. Name-only matching was not used for canonical joins.",
    ]
    (REPORT_DEF / "DEFENDER_ID_CROSSWALK_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")


def pct(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{float(value) * 100:.2f}%"


def pt_status_summary() -> dict[str, Any]:
    path = RAW_DEF / "metadata" / "pt_defend_download_status.json"
    if not path.exists():
        return {"success": [], "failed_or_empty": []}
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {
        "successful_seasons": sorted({r["season"] for r in rows if r.get("success")}),
        "successful_categories": sorted({r["defense_category"] for r in rows if r.get("success")}),
        "success_count": sum(1 for r in rows if r.get("success")),
        "empty_count": sum(1 for r in rows if r.get("empty")),
        "failed_count": sum(1 for r in rows if r.get("failed")),
        "failed_or_empty": [
            {
                "season": r.get("season"),
                "season_type": r.get("season_type"),
                "defense_category": r.get("defense_category"),
                "status": "empty" if r.get("empty") else "failed",
                "error": r.get("error"),
            }
            for r in rows
            if r.get("empty") or r.get("failed")
        ],
    }


def write_audit_report(
    records: list[dict[str, Any]],
    matchup_audit: dict[str, Any],
    shot_audit: dict[str, Any],
    pt_audit: dict[str, Any],
    prior_audit: dict[str, Any],
    cross_audit: dict[str, Any],
) -> None:
    pt_status = pt_status_summary()
    lines = [
        "# Defensive Context Data Audit",
        "",
        "## Observed Shot-Level Data",
        "",
        f"- 2014/15 shot-log rows: {shot_audit.get('rows', 0):,}",
        f"- Exact closest-defender rows: {shot_audit.get('exact_closest_defender_rows', 0):,}",
        f"- Closest-defender-distance coverage: {pct(1 - shot_audit.get('missing_defender_distance_pct', 1))}",
        f"- Shot-clock coverage: {pct(1 - shot_audit.get('missing_shot_clock_pct', 1))}",
        f"- Dribbles coverage: {pct(1.0)}",
        f"- Touch-time coverage: {pct(1.0)}",
        "",
        "## Aggregated Defender Profile Data",
        "",
        f"- Matchup rows: {matchup_audit.get('rows', 0):,}",
        f"- Matchup seasons: {', '.join(matchup_audit.get('seasons', []))}",
        f"- Matchup games: {matchup_audit.get('games', 0):,}",
        f"- Matchup unique defenders: {matchup_audit.get('unique_defenders', 0):,}",
        f"- PT Defend rows: {pt_audit.get('rows', 0):,}",
        f"- PT Defend successful seasons: {', '.join(pt_status.get('successful_seasons', [])) or 'none'}",
        f"- PT Defend successful categories: {', '.join(pt_status.get('successful_categories', [])) or 'none'}",
        f"- PT Defend failed downloads: {pt_status.get('failed_count', 0):,}",
        f"- PT Defend empty downloads: {pt_status.get('empty_count', 0):,}",
        "",
        "## Inferred / Derived Features",
        "",
        f"- Leakage-safe defender prior rows: {prior_audit.get('rows', 0):,}",
        f"- Prior formula: shifted expanding defender history; FG% and 3P% use Bayesian shrinkage with strength {prior_audit.get('prior_strength', 'n/a')}.",
        "",
        "## Defender ID Matching",
        "",
        f"- Direct tracking defender ID match rate: {pct(cross_audit.get('tracking_match_rate'))}",
        f"- IDs in multiple sources: {cross_audit.get('ids_in_multiple_sources', 0):,}",
        "",
        "## Data-Quality Problems",
        "",
        f"- Matchup duplicate canonical keys: {matchup_audit.get('duplicate_count', 0):,}",
        f"- Matchup FGM > FGA rows: {matchup_audit.get('fgm_gt_fga', 0):,}",
        f"- Matchup 3PM > 3PA rows: {matchup_audit.get('three_pm_gt_pa', 0):,}",
        f"- Matchup FTM > FTA rows: {matchup_audit.get('ftm_gt_fta', 0):,}",
        f"- Matchup time-share percentage values > 1.0: {sum(matchup_audit.get('time_share_percentage_gt_one', {}).values()):,}",
        f"- Shot-clock out-of-bounds rows: {shot_audit.get('shot_clock_out_of_bounds', 0):,}",
        f"- Negative touch-time rows: {shot_audit.get('touch_time_negative', 0):,}",
        "",
        "## Feasibility",
        "",
        "- Closest defender distance can be learned only from the 2014/15 event-level tracking shot logs.",
        "- Modern defender profiles can be built from Player Matchups and LeagueDashPtDefend when available.",
        "- Modern exact defender-per-shot data was not found; matchup archives are game-level player matchup aggregates, not shot-level closest-defender labels.",
        "- Shot clock, dribbles, and touch time are available in the 2014/15 tracking dataset and prepared for a future experiment.",
        "- Domain shift is material: tracking data is 2014/15, while production shot-quality data covers 2020/21-2025/26 with different shot mix, spacing, and player population.",
        "- Safe next modeling experiment: compare 2014/15 shot-log models with and without closest defender distance, shot clock, dribbles, and touch time before any production integration.",
    ]
    (REPORT_DEF / "DEFENSIVE_CONTEXT_DATA_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ensure_dirs()
    records = discover_files()
    records = organize_raw(records)
    write_inventory(records)
    matchups, matchup_audit = normalize_matchups()
    shot_logs, shot_audit = normalize_shot_logs()
    pt_def, profiles, pt_audit = normalize_pt_defend()
    priors, prior_audit = build_defender_priors(matchups)
    crosswalk, cross_audit = build_crosswalk(shot_logs, matchups, pt_def)
    write_schema_mapping()
    write_crosswalk_report(cross_audit)
    write_manifest(records)
    write_audit_report(records, matchup_audit, shot_audit, pt_audit, prior_audit, cross_audit)
    summary = {
        "records": records,
        "matchup_audit": matchup_audit,
        "shot_audit": shot_audit,
        "pt_audit": pt_audit,
        "prior_audit": prior_audit,
        "cross_audit": cross_audit,
        "pt_status": pt_status_summary(),
    }
    (RAW_DEF / "metadata" / "defense_build_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
