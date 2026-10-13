"""Defensive Shot Quality Experiment v1.

Isolated research experiment for 2014/15 tracking shot logs. This does not
modify production inference, Decision Policy, or frontend files.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[2]
sys.modules.setdefault("run_defensive_shot_quality_experiment_v1", sys.modules[__name__])
SHOT_LOG_PATH = ROOT / "data" / "processed" / "defense" / "shot_logs_2014_15.parquet"
EXPERIMENT_PATH = ROOT / "data" / "analysis" / "defense" / "shot_quality_tracking_experiment.parquet"
PT_RAW = ROOT / "data" / "raw" / "defense" / "nba_pt_defend" / "as_of_snapshots"
PROCESSED_DEF = ROOT / "data" / "processed" / "defense"
ANALYSIS_DEF = ROOT / "data" / "analysis" / "defense"
REPORT_DIR = ROOT / "reports" / "modeling"
MODEL_DIR = ROOT / "artifacts" / "models"
METADATA_DIR = ROOT / "data" / "metadata"

RANDOM_SEED = 20261011
CATEGORIES = ["Overall", "2 Pointers", "3 Pointers", "Less Than 6Ft", "Less Than 10Ft", "Greater Than 15Ft"]
PT_HEADERS = {
    "Host": "stats.nba.com",
    "Connection": "keep-alive",
    "Accept": "application/json, text/plain, */*",
    "x-nba-stats-token": "true",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    ),
    "x-nba-stats-origin": "stats",
    "Origin": "https://www.nba.com",
    "Referer": "https://www.nba.com/",
    "Accept-Language": "en-US,en;q=0.9",
}


def ensure_dirs() -> None:
    for path in [ANALYSIS_DEF, REPORT_DIR, MODEL_DIR, METADATA_DIR, PT_RAW, PROCESSED_DEF]:
        path.mkdir(parents=True, exist_ok=True)


def slug(value: str) -> str:
    return value.lower().replace(" ", "_")


def parse_table(payload: dict[str, Any]) -> pd.DataFrame:
    result_sets = payload.get("resultSets") or payload.get("resultSet") or []
    if isinstance(result_sets, dict):
        result_sets = [result_sets]
    for result in result_sets:
        headers = result.get("headers")
        rows = result.get("rowSet")
        if headers is not None and rows is not None:
            return pd.DataFrame(rows, columns=headers)
    return pd.DataFrame()


def pt_params(category: str, date_to: str) -> dict[str, str]:
    return {
        "College": "",
        "Conference": "",
        "Country": "",
        "DateFrom": "",
        "DateTo": date_to,
        "DefenseCategory": category,
        "Division": "",
        "DraftPick": "",
        "DraftYear": "",
        "GameSegment": "",
        "Height": "",
        "LastNGames": "",
        "LeagueID": "00",
        "Location": "",
        "Month": "",
        "OpponentTeamID": "",
        "Outcome": "",
        "PORound": "",
        "PerMode": "Totals",
        "Period": "",
        "PlayerExperience": "",
        "PlayerID": "",
        "PlayerPosition": "",
        "Season": "2014-15",
        "SeasonSegment": "",
        "SeasonType": "Regular Season",
        "StarterBench": "",
        "TeamID": "",
        "VsConference": "",
        "VsDivision": "",
        "Weight": "",
    }


def download_pt_asof_snapshots(force: bool = False, delay: float = 1.0) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Monthly prior-period LeagueDashPtDefend snapshots for 2014/15."""
    # Shots begin in late 2014 and run through 2015. A shot in month M uses
    # cumulative stats through the previous month end.
    cutoffs = {
        "2014-11": "10/31/2014",
        "2014-12": "11/30/2014",
        "2015-01": "12/31/2014",
        "2015-02": "01/31/2015",
        "2015-03": "02/28/2015",
        "2015-04": "03/31/2015",
    }
    status: list[dict[str, Any]] = []
    frames = []
    for snapshot_month, date_to in cutoffs.items():
        out_dir = PT_RAW / snapshot_month
        out_dir.mkdir(parents=True, exist_ok=True)
        for category in CATEGORIES:
            out_path = out_dir / f"{slug(category)}.csv"
            if out_path.exists() and not force:
                df = pd.read_csv(out_path)
                ok = not df.empty
            else:
                url = "https://stats.nba.com/stats/leaguedashptdefend"
                params = pt_params(category, date_to)
                ok = False
                error = None
                df = pd.DataFrame()
                try:
                    request = Request(f"{url}?{urlencode(params)}", headers=PT_HEADERS)
                    with urlopen(request, timeout=30) as response:  # noqa: S310 - fixed NBA endpoint.
                        body = response.read().decode("utf-8", errors="replace")
                    df = parse_table(json.loads(body))
                    ok = not df.empty
                    if ok:
                        df.to_csv(out_path, index=False)
                except Exception as exc:  # noqa: BLE001
                    error = repr(exc)
                status.append(
                    {
                        "snapshot_month": snapshot_month,
                        "date_to": date_to,
                        "category": category,
                        "success": ok,
                        "rows": int(len(df)),
                        "path": str(out_path.relative_to(ROOT)) if out_path.exists() else None,
                        "error": error,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )
                time.sleep(delay)
            if not df.empty:
                tmp = df.copy()
                tmp["snapshot_month"] = snapshot_month
                tmp["date_to"] = date_to
                tmp["defense_category"] = slug(category)
                frames.append(tmp)
                if not any(s.get("snapshot_month") == snapshot_month and s.get("category") == category for s in status):
                    status.append(
                        {
                            "snapshot_month": snapshot_month,
                            "date_to": date_to,
                            "category": category,
                            "success": True,
                            "rows": int(len(df)),
                            "path": str(out_path.relative_to(ROOT)),
                            "error": None,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                    )
    status_path = ROOT / "data" / "raw" / "defense" / "metadata" / "pt_defend_asof_snapshot_status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    if not frames:
        return pd.DataFrame(), {"success": False, "status": status}
    raw = pd.concat(frames, ignore_index=True)
    rename = {
        "CLOSE_DEF_PERSON_ID": "defender_id",
        "PLAYER_NAME": "defender_name",
        "D_FGA": "d_fga",
        "D_FG_PCT": "d_fg_pct",
        "NORMAL_FG_PCT": "normal_fg_pct",
        "PCT_PLUSMINUS": "pct_plusminus",
    }
    raw = raw.rename(columns=rename)
    keep = ["defender_id", "defender_name", "snapshot_month", "date_to", "defense_category", "d_fga", "d_fg_pct", "normal_fg_pct", "pct_plusminus"]
    raw = raw[[c for c in keep if c in raw.columns]]
    profile = raw.pivot_table(
        index=["defender_id", "snapshot_month"],
        columns="defense_category",
        values=[c for c in ["d_fga", "d_fg_pct", "normal_fg_pct", "pct_plusminus"] if c in raw.columns],
        aggfunc="first",
    )
    profile.columns = [f"defender_pt_{cat_prefix(cat)}_{metric}" for metric, cat in profile.columns]
    profile = profile.reset_index()
    profile.to_parquet(PROCESSED_DEF / "pt_defend_2014_15_asof_profiles.parquet", index=False)
    return profile, {
        "success": True,
        "status": status,
        "rows": int(len(profile)),
        "snapshots": sorted(profile["snapshot_month"].unique().tolist()),
        "categories": sorted(raw["defense_category"].unique().tolist()),
    }


def cat_prefix(cat: str) -> str:
    return {
        "overall": "overall",
        "2_pointers": "2pt",
        "3_pointers": "3pt",
        "less_than_6ft": "lt6",
        "less_than_10ft": "lt10",
        "greater_than_15ft": "gt15",
    }.get(str(cat), str(cat))


def parse_game_date(matchup: pd.Series) -> pd.Series:
    raw = matchup.astype(str).str.extract(r"^([A-Z]{3} \d{2}, \d{4})")[0]
    return pd.to_datetime(raw, format="%b %d, %Y", errors="coerce")


def seconds_remaining(game_clock: pd.Series) -> pd.Series:
    parts = game_clock.astype(str).str.extract(r"(?P<min>\d+):(?P<sec>\d+)")
    return pd.to_numeric(parts["min"], errors="coerce") * 60 + pd.to_numeric(parts["sec"], errors="coerce")


def distance_bucket(s: pd.Series) -> pd.Series:
    return pd.cut(s.astype(float), [-0.1, 4, 10, 16, 23.75, 100], labels=["0-4", "4-10", "10-16", "16-24", "24+"]).astype(str)


def pressure_bucket(s: pd.Series) -> pd.Series:
    return pd.cut(s.astype(float), [-0.1, 2, 4, 6, 200], labels=["0-2 ft", "2-4 ft", "4-6 ft", "6+ ft"]).astype(str)


def load_and_prepare(pt_profiles: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    source = pd.read_parquet(SHOT_LOG_PATH)
    exp = pd.read_parquet(EXPERIMENT_PATH)
    df = source.copy()
    df["game_date"] = parse_game_date(df["matchup"])
    df["seconds_remaining_period"] = seconds_remaining(df["game_clock"])
    df["made"] = df["made"].astype(int)
    df["shot_value"] = df["shot_value"].astype(int)
    df["is_home"] = df["location"].eq("H").astype(int)
    df["touch_time_invalid"] = df["touch_time"].lt(0)
    df["touch_time_missing"] = df["touch_time"].isna() | df["touch_time_invalid"]
    df.loc[df["touch_time_invalid"], "touch_time"] = np.nan
    df["shot_clock_missing"] = df["shot_clock"].isna()
    df["closest_defender_distance_missing"] = df["closest_defender_distance"].isna()
    df["distance_bucket"] = distance_bucket(df["shot_distance"])
    df["pressure_bucket"] = pressure_bucket(df["closest_defender_distance"])
    df["shot_clock_bucket"] = pd.cut(df["shot_clock"], [-0.1, 7, 15, 24], labels=["late", "middle", "early"]).astype(str)
    df["dribble_bucket"] = pd.cut(df["dribbles"], [-0.1, 0, 2, 6, 100], labels=["0", "1-2", "3-6", "7+"]).astype(str)
    df["touch_time_bucket"] = pd.cut(df["touch_time"], [-0.1, 2, 6, 100], labels=["quick", "moderate", "long"]).astype(str)
    df = df.sort_values(["game_date", "game_id", "period", "seconds_remaining_period"], ascending=[True, True, True, False]).reset_index(drop=True)
    if pt_profiles is not None and not pt_profiles.empty:
        df["snapshot_month"] = df["game_date"].dt.strftime("%Y-%m")
        df = df.merge(pt_profiles, left_on=["closest_defender_id", "snapshot_month"], right_on=["defender_id", "snapshot_month"], how="left")
    audit = {
        "source_rows": int(len(source)),
        "experiment_rows": int(len(exp)),
        "unique_games": int(df["game_id"].nunique()),
        "unique_dates": int(df["game_date"].nunique()),
        "unique_shooters": int(df["shooter_id"].nunique()),
        "unique_defenders": int(df["closest_defender_id"].nunique()),
        "made_distribution": df["made"].value_counts().sort_index().to_dict(),
        "shot_value_distribution": df["shot_value"].value_counts().sort_index().to_dict(),
        "negative_touch_time_rows": int(df["touch_time_invalid"].sum()),
        "missing_shot_clock_rows": int(df["shot_clock_missing"].sum()),
        "closest_defender_distance": describe(df["closest_defender_distance"]),
        "shot_distance": describe(df["shot_distance"]),
        "shot_clock": describe(df["shot_clock"]),
        "dribbles": describe(df["dribbles"]),
        "touch_time": describe(df["touch_time"]),
    }
    return df, audit


def describe(s: pd.Series) -> dict[str, float | None]:
    clean = pd.to_numeric(s, errors="coerce").dropna()
    if clean.empty:
        return {"min": None, "median": None, "p95": None, "max": None}
    return {"min": float(clean.min()), "median": float(clean.median()), "p95": float(clean.quantile(0.95)), "max": float(clean.max())}


def chronological_split(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    games = df.groupby("game_id", as_index=False).agg(game_date=("game_date", "min"), rows=("made", "size"))
    games = games.sort_values(["game_date", "game_id"]).reset_index(drop=True)
    n = len(games)
    games["split"] = "test"
    games.loc[: int(n * 0.70) - 1, "split"] = "train"
    games.loc[int(n * 0.70) : int(n * 0.85) - 1, "split"] = "validation"
    out = df.merge(games[["game_id", "split"]], on="game_id", how="left")
    summary = {}
    for split, grp in out.groupby("split"):
        summary[split] = {
            "rows": int(len(grp)),
            "games": int(grp["game_id"].nunique()),
            "date_min": str(grp["game_date"].min().date()),
            "date_max": str(grp["game_date"].max().date()),
            "actual_fg_pct": float(grp["made"].mean()),
        }
    overlap = games.groupby("game_id")["split"].nunique().max()
    summary["no_game_crosses_splits"] = bool(overlap == 1)
    return out, summary


def add_game_priors(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = df.copy()
    global_fg = float(df.loc[df["split"].eq("train"), "made"].mean())
    global_2p = float(df.loc[df["split"].eq("train") & df["shot_value"].eq(2), "made"].mean())
    global_3p = float(df.loc[df["split"].eq("train") & df["shot_value"].eq(3), "made"].mean())
    strengths = {"overall": 100.0, "value": 75.0, "bucket": 40.0, "defender": 100.0}
    df["_row_id"] = np.arange(len(df))
    game_order = df.groupby("game_id", as_index=False).agg(game_date=("game_date", "min")).sort_values(["game_date", "game_id"])
    game_order["game_order"] = np.arange(len(game_order))
    df = df.merge(game_order[["game_id", "game_order"]], on="game_id", how="left")

    df = _merge_prior(df, ["shooter_id"], "shooter", global_fg, strengths["overall"])
    df = _merge_prior(df, ["shooter_id"], "shooter_2p", global_2p, strengths["value"], mask=df["shot_value"].eq(2))
    df = _merge_prior(df, ["shooter_id"], "shooter_3p", global_3p, strengths["value"], mask=df["shot_value"].eq(3))
    df = _merge_prior(df, ["shooter_id", "distance_bucket"], "shooter_distance", global_fg, strengths["bucket"])

    df = _merge_prior(df, ["closest_defender_id"], "defender", global_fg, strengths["defender"], allowed=True)
    df = _merge_prior(df, ["closest_defender_id"], "defender_2p", global_2p, strengths["value"], mask=df["shot_value"].eq(2), allowed=True)
    df = _merge_prior(df, ["closest_defender_id"], "defender_3p", global_3p, strengths["value"], mask=df["shot_value"].eq(3), allowed=True)
    df = _merge_prior(df, ["closest_defender_id", "distance_bucket"], "defender_distance", global_fg, strengths["bucket"], allowed=True)

    rename = {
        "shooter_prior_attempts": "shooter_prior_fga",
        "shooter_prior_makes": "shooter_prior_fgm",
        "shooter_prior_pct": "shooter_prior_fg_pct",
        "shooter_prior_pct_shrunk": "shooter_prior_fg_pct_shrunk",
        "shooter_2p_prior_attempts": "shooter_prior_2pa",
        "shooter_2p_prior_pct_shrunk": "shooter_prior_2p_pct_shrunk",
        "shooter_3p_prior_attempts": "shooter_prior_3pa",
        "shooter_3p_prior_pct_shrunk": "shooter_prior_3p_pct_shrunk",
        "shooter_distance_prior_attempts": "shooter_distance_prior_attempts",
        "shooter_distance_prior_pct_shrunk": "shooter_distance_prior_fg_pct_shrunk",
        "defender_prior_attempts": "defender_prior_shots_defended",
        "defender_prior_makes": "defender_prior_fgm_allowed",
        "defender_prior_pct": "defender_prior_fg_pct_allowed",
        "defender_prior_pct_shrunk": "defender_prior_fg_pct_allowed_shrunk",
        "defender_2p_prior_attempts": "defender_prior_2pa_defended",
        "defender_2p_prior_pct_shrunk": "defender_prior_2p_pct_allowed_shrunk",
        "defender_3p_prior_attempts": "defender_prior_3pa_defended",
        "defender_3p_prior_pct_shrunk": "defender_prior_3p_pct_allowed_shrunk",
        "defender_distance_prior_attempts": "defender_distance_prior_attempts",
        "defender_distance_prior_pct_shrunk": "defender_distance_prior_fg_pct_allowed_shrunk",
    }
    df = df.rename(columns=rename)
    df["shooter_cold_start"] = df["shooter_prior_fga"].eq(0)
    df["defender_cold_start"] = df["defender_prior_shots_defended"].eq(0)
    prior_audit = {
        "population_fg_pct": global_fg,
        "population_2p_pct": global_2p,
        "population_3p_pct": global_3p,
        "strengths": strengths,
        "formula": "(prior_makes + strength * population_rate) / (prior_attempts + strength), prior stats are aggregated by completed previous games only",
        "shooter_cold_start_rows": int(df["shooter_cold_start"].sum()),
        "defender_cold_start_rows": int(df["defender_cold_start"].sum()),
    }
    return df.sort_values("_row_id").drop(columns=["_row_id"]), prior_audit


def _merge_prior(
    df: pd.DataFrame,
    group_cols: list[str],
    prefix: str,
    prior_mean: float,
    strength: float,
    mask: pd.Series | None = None,
    allowed: bool = False,
) -> pd.DataFrame:
    work = df.copy()
    attempt = pd.Series(1, index=work.index, dtype=float) if mask is None else mask.astype(float)
    made = work["made"].astype(float) * attempt
    totals = work[group_cols + ["game_id", "game_order"]].copy()
    totals["attempt"] = attempt
    totals["made"] = made
    game_totals = totals.groupby(group_cols + ["game_id", "game_order"], dropna=False, as_index=False).agg(attempt=("attempt", "sum"), made=("made", "sum"))
    game_totals = game_totals.sort_values(group_cols + ["game_order", "game_id"])
    parts = []
    for _, grp in game_totals.groupby(group_cols, dropna=False):
        g = grp.copy()
        g[f"{prefix}_prior_attempts"] = g["attempt"].cumsum().shift(fill_value=0)
        g[f"{prefix}_prior_makes"] = g["made"].cumsum().shift(fill_value=0)
        parts.append(g)
    priors = pd.concat(parts, ignore_index=True)
    priors[f"{prefix}_prior_pct"] = np.where(
        priors[f"{prefix}_prior_attempts"] > 0,
        priors[f"{prefix}_prior_makes"] / priors[f"{prefix}_prior_attempts"],
        prior_mean,
    )
    priors[f"{prefix}_prior_pct_shrunk"] = (priors[f"{prefix}_prior_makes"] + strength * prior_mean) / (
        priors[f"{prefix}_prior_attempts"] + strength
    )
    keep = group_cols + ["game_id", f"{prefix}_prior_attempts", f"{prefix}_prior_makes", f"{prefix}_prior_pct", f"{prefix}_prior_pct_shrunk"]
    return work.merge(priors[keep], on=group_cols + ["game_id"], how="left")


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns: list[str]):
        self.columns = columns

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.columns].copy()


@dataclass
class ModelSpec:
    key: str
    label: str
    families: list[str]
    columns: list[str]
    categorical: list[str]
    deployable: bool = True


def feature_specs(df: pd.DataFrame, pt_available: bool) -> list[ModelSpec]:
    base_num = [
        "shot_distance",
        "shot_value",
        "period",
        "seconds_remaining_period",
        "is_home",
        "shooter_prior_fga",
        "shooter_prior_fg_pct_shrunk",
        "shooter_prior_2pa",
        "shooter_prior_2p_pct_shrunk",
        "shooter_prior_3pa",
        "shooter_prior_3p_pct_shrunk",
        "shooter_distance_prior_attempts",
        "shooter_distance_prior_fg_pct_shrunk",
    ]
    base_cat = ["distance_bucket", "shooter_cold_start"]
    pressure = ["closest_defender_distance"]
    creation = ["shot_clock", "dribbles", "touch_time", "shot_clock_missing", "touch_time_missing", "touch_time_invalid"]
    defender = [
        "defender_prior_shots_defended",
        "defender_prior_fg_pct_allowed_shrunk",
        "defender_prior_2pa_defended",
        "defender_prior_2p_pct_allowed_shrunk",
        "defender_prior_3pa_defended",
        "defender_prior_3p_pct_allowed_shrunk",
        "defender_distance_prior_attempts",
        "defender_distance_prior_fg_pct_allowed_shrunk",
        "defender_cold_start",
    ]
    pt_cols = [c for c in df.columns if c.startswith("defender_pt_")]
    specs = [
        ModelSpec("A", "Base shot quality", ["shot", "shooter_history"], base_num + base_cat, base_cat),
        ModelSpec("B", "Defensive pressure", ["A", "closest_defender_distance"], base_num + base_cat + pressure, base_cat),
        ModelSpec("C", "Offensive creation context", ["A", "shot_clock_dribbles_touch"], base_num + base_cat + creation, base_cat),
        ModelSpec("D", "Pressure + creation", ["A", "closest_defender_distance", "shot_clock_dribbles_touch"], base_num + base_cat + pressure + creation, base_cat),
        ModelSpec("E", "Defender shot-log history", ["D", "defender_history"], base_num + base_cat + pressure + creation + defender, base_cat + ["defender_cold_start"]),
        ModelSpec("RAW_ID", "Raw defender ID ablation", ["D", "raw_defender_id"], base_num + base_cat + pressure + creation + ["closest_defender_id"], base_cat + ["closest_defender_id"], False),
    ]
    if pt_available and pt_cols:
        specs.append(ModelSpec("F", "PT Defend as-of profile", ["D", "pt_defend_asof"], base_num + base_cat + pressure + creation + pt_cols, base_cat))
        specs.append(ModelSpec("G", "Full generalizable", ["D", "defender_history", "pt_defend_asof"], base_num + base_cat + pressure + creation + defender + pt_cols, base_cat + ["defender_cold_start"]))
    else:
        specs.append(ModelSpec("G", "Full generalizable", ["D", "defender_history"], base_num + base_cat + pressure + creation + defender, base_cat + ["defender_cold_start"]))
    return specs


class CalibratedWrapper:
    def __init__(self, pipeline: Pipeline, calibration: str, calibrator: Any = None):
        self.pipeline = pipeline
        self.calibration = calibration
        self.calibrator = calibrator

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = self.pipeline.predict_proba(X)[:, 1]
        p = np.clip(p, 1e-6, 1 - 1e-6)
        if self.calibration == "sigmoid":
            z = np.log(p / (1 - p)).reshape(-1, 1)
            p = self.calibrator.predict_proba(z)[:, 1]
        elif self.calibration == "isotonic":
            p = self.calibrator.predict(p)
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.column_stack([1 - p, p])


DataFrameSelector.__module__ = "run_defensive_shot_quality_experiment_v1"
CalibratedWrapper.__module__ = "run_defensive_shot_quality_experiment_v1"
ModelSpec.__module__ = "run_defensive_shot_quality_experiment_v1"


def make_pipeline(columns: list[str], categorical: list[str], estimator_name: str) -> Pipeline:
    numeric = [c for c in columns if c not in categorical]
    if estimator_name == "logistic":
        prep = ColumnTransformer(
            [
                ("num", Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True)), ("scale", StandardScaler())]), numeric),
                ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
            ],
            remainder="drop",
        )
        est = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
    else:
        prep = ColumnTransformer(
            [
                ("num", SimpleImputer(strategy="median", add_indicator=True), numeric),
                ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))]), categorical),
            ],
            remainder="drop",
        )
        est = HistGradientBoostingClassifier(
            max_iter=90,
            learning_rate=0.06,
            l2_regularization=0.02,
            max_leaf_nodes=31,
            random_state=RANDOM_SEED,
        )
    return Pipeline([("select", DataFrameSelector(columns)), ("prep", prep), ("model", est)])


def fit_with_calibration(pipe: Pipeline, train: pd.DataFrame, val: pd.DataFrame, columns: list[str]) -> tuple[CalibratedWrapper, dict[str, Any]]:
    y_train = train["made"].astype(int)
    y_val = val["made"].astype(int)
    pipe.fit(train, y_train)
    val_p = np.clip(pipe.predict_proba(val)[:, 1], 1e-6, 1 - 1e-6)
    candidates: dict[str, CalibratedWrapper] = {"uncalibrated": CalibratedWrapper(pipe, "uncalibrated")}
    sigmoid = LogisticRegression(max_iter=1000)
    sigmoid.fit(np.log(val_p / (1 - val_p)).reshape(-1, 1), y_val)
    candidates["sigmoid"] = CalibratedWrapper(pipe, "sigmoid", sigmoid)
    isotonic = IsotonicRegression(out_of_bounds="clip")
    isotonic.fit(val_p, y_val)
    candidates["isotonic"] = CalibratedWrapper(pipe, "isotonic", isotonic)
    scored = {}
    for name, wrapper in candidates.items():
        p = wrapper.predict_proba(val)[:, 1]
        scored[name] = metric_dict(y_val.to_numpy(), p)
    best_name = min(scored, key=lambda k: scored[k]["log_loss"])
    return candidates[best_name], {"selected_calibration": best_name, "calibration_metrics": scored}


def metric_dict(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    p = np.clip(p.astype(float), 1e-6, 1 - 1e-6)
    return {
        "rows": int(len(y)),
        "actual_fg_pct": float(np.mean(y)),
        "predicted_fg_pct": float(np.mean(p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        "ece": ece_score(y, p),
    }


def ece_score(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        if mask.any():
            ece += float(mask.mean()) * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(ece)


def run_models(df: pd.DataFrame, specs: list[ModelSpec]) -> tuple[pd.DataFrame, dict[str, Any], dict[str, CalibratedWrapper], dict[str, ModelSpec]]:
    train = df[df["split"].eq("train")]
    val = df[df["split"].eq("validation")]
    test = df[df["split"].eq("test")]
    rows = []
    models: dict[str, CalibratedWrapper] = {}
    spec_by_key = {s.key: s for s in specs}
    for spec in specs:
        for estimator in ["logistic", "hgb"]:
            key = f"{spec.key}_{estimator}"
            print(f"Fitting {key}: {spec.label}", flush=True)
            pipe = make_pipeline(spec.columns, spec.categorical, estimator)
            wrapper, cal_info = fit_with_calibration(pipe, train, val, spec.columns)
            val_p = wrapper.predict_proba(val)[:, 1]
            test_p = wrapper.predict_proba(test)[:, 1]
            val_m = metric_dict(val["made"].to_numpy(), val_p)
            test_m = metric_dict(test["made"].to_numpy(), test_p)
            row = {
                "model_key": key,
                "family_key": spec.key,
                "label": spec.label,
                "estimator": estimator,
                "feature_families": ", ".join(spec.families),
                "calibration": cal_info["selected_calibration"],
                **{f"validation_{k}": v for k, v in val_m.items()},
                **{f"test_{k}": v for k, v in test_m.items()},
            }
            rows.append(row)
            models[key] = wrapper
    metrics = pd.DataFrame(rows)
    baseline = metrics.loc[metrics["family_key"].eq("A")].sort_values("validation_log_loss").iloc[0]
    metrics["delta_test_log_loss_vs_A"] = metrics["test_log_loss"] - float(baseline["test_log_loss"])
    metrics["delta_test_brier_vs_A"] = metrics["test_brier"] - float(baseline["test_brier"])
    selected_row = metrics[~metrics["family_key"].eq("RAW_ID")].sort_values(["validation_log_loss", "validation_brier"]).iloc[0]
    selected_key = str(selected_row["model_key"])
    return metrics, {"selected_model_key": selected_key, "baseline_model_key": str(baseline["model_key"])}, models, spec_by_key


def add_predictions(df: pd.DataFrame, models: dict[str, CalibratedWrapper], selected: dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    for key, model in models.items():
        out[f"p_{key}"] = model.predict_proba(out)[:, 1]
    out["p_selected"] = out[f"p_{selected['selected_model_key']}"]
    return out


def bootstrap_comparisons(preds: pd.DataFrame, selected: dict[str, Any], metrics: pd.DataFrame, n: int = 200) -> dict[str, Any]:
    rng = np.random.default_rng(RANDOM_SEED)
    test = preds[preds["split"].eq("test")].copy()
    games = test["game_id"].drop_duplicates().to_numpy()
    family_best = metrics.sort_values("validation_log_loss").groupby("family_key").head(1).set_index("family_key")["model_key"].to_dict()
    comparisons = {
        "A_vs_B": (family_best.get("A"), family_best.get("B")),
        "A_vs_D": (family_best.get("A"), family_best.get("D")),
        "D_vs_defender_profile": (family_best.get("D"), family_best.get("E") or family_best.get("F")),
        "A_vs_final_G": (family_best.get("A"), selected["selected_model_key"]),
    }
    out = {}
    for name, pair in comparisons.items():
        a, b = pair
        if not a or not b:
            continue
        deltas_ll, deltas_brier = [], []
        game_groups = {g: idx.to_numpy() for g, idx in test.groupby("game_id").groups.items()}
        for _ in range(n):
            sampled = rng.choice(games, size=len(games), replace=True)
            indices = np.concatenate([game_groups[g] for g in sampled])
            sample = test.loc[indices]
            y = sample["made"].to_numpy()
            pa = sample[f"p_{a}"].to_numpy()
            pb = sample[f"p_{b}"].to_numpy()
            deltas_ll.append(log_loss(y, pb, labels=[0, 1]) - log_loss(y, pa, labels=[0, 1]))
            deltas_brier.append(brier_score_loss(y, pb) - brier_score_loss(y, pa))
        out[name] = {
            "model_a": a,
            "model_b": b,
            "log_loss_delta_ci95": [float(np.quantile(deltas_ll, 0.025)), float(np.quantile(deltas_ll, 0.975))],
            "brier_delta_ci95": [float(np.quantile(deltas_brier, 0.025)), float(np.quantile(deltas_brier, 0.975))],
            "bootstrap_resamples": n,
        }
    return out


def analysis_tables(preds: pd.DataFrame, selected_key: str) -> dict[str, Any]:
    test = preds[preds["split"].eq("test")].copy()
    pcol = f"p_{selected_key}"
    pressure = (
        test.groupby("pressure_bucket", dropna=False)
        .agg(
            attempts=("made", "size"),
            actual_fg_pct=("made", "mean"),
            predicted_fg_pct=(pcol, "mean"),
            two_pt_fg_pct=("made", lambda s: test.loc[s.index, "made"][test.loc[s.index, "shot_value"].eq(2)].mean()),
            three_pt_fg_pct=("made", lambda s: test.loc[s.index, "made"][test.loc[s.index, "shot_value"].eq(3)].mean()),
        )
        .reset_index()
    )
    shot_clock = test.groupby("shot_clock_bucket", dropna=False).agg(attempts=("made", "size"), actual_fg_pct=("made", "mean"), predicted_fg_pct=(pcol, "mean")).reset_index()
    dribble = test.groupby("dribble_bucket", dropna=False).agg(attempts=("made", "size"), actual_fg_pct=("made", "mean"), predicted_fg_pct=(pcol, "mean")).reset_index()
    touch = test.groupby("touch_time_bucket", dropna=False).agg(attempts=("made", "size"), actual_fg_pct=("made", "mean"), predicted_fg_pct=(pcol, "mean")).reset_index()
    return {
        "pressure_buckets": pressure.to_dict(orient="records"),
        "shot_clock_buckets": shot_clock.to_dict(orient="records"),
        "dribble_buckets": dribble.to_dict(orient="records"),
        "touch_time_buckets": touch.to_dict(orient="records"),
    }


def feature_importance(model: CalibratedWrapper, spec: ModelSpec, df: pd.DataFrame) -> pd.DataFrame:
    test = df[df["split"].eq("test")].sample(min(6000, int((df["split"] == "test").sum())), random_state=RANDOM_SEED)
    y = test["made"].to_numpy()
    baseline = log_loss(y, model.predict_proba(test)[:, 1], labels=[0, 1])
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    for col in spec.columns:
        deltas = []
        for _ in range(5):
            shuffled = test.copy()
            shuffled[col] = rng.permutation(shuffled[col].to_numpy())
            score = log_loss(y, model.predict_proba(shuffled)[:, 1], labels=[0, 1])
            deltas.append(score - baseline)
        rows.append({"feature": col, "importance_log_loss_delta": float(np.mean(deltas)), "std": float(np.std(deltas))})
    return pd.DataFrame(rows).sort_values("importance_log_loss_delta", ascending=False)


def controlled_pressure_grid(model: CalibratedWrapper, spec: ModelSpec, df: pd.DataFrame) -> list[dict[str, Any]]:
    base = df[df["split"].eq("test")].copy()
    contexts = {
        "near_rim_2pt": base[(base["shot_value"].eq(2)) & (base["shot_distance"] <= 4)],
        "midrange_2pt": base[(base["shot_value"].eq(2)) & (base["shot_distance"].between(10, 18))],
        "three_pt": base[base["shot_value"].eq(3)],
    }
    grid = [0.5, 1, 2, 3, 4, 5, 6, 8, 10]
    rows = []
    for name, subset in contexts.items():
        if subset.empty:
            continue
        row = subset.sample(1, random_state=RANDOM_SEED).iloc[[0]].copy()
        for value in grid:
            x = row.copy()
            x["closest_defender_distance"] = value
            x["pressure_bucket"] = pressure_bucket(x["closest_defender_distance"])
            rows.append({"context": name, "closest_defender_distance": value, "p_make": float(model.predict_proba(x)[0, 1])})
    return rows


def domain_shift(df: pd.DataFrame) -> dict[str, Any]:
    modern_path = ROOT / "data" / "processed" / "shots_enriched.parquet"
    if not modern_path.exists():
        return {"available": False}
    modern = pd.read_parquet(modern_path, columns=["shot_distance", "shot_value", "shot_made", "player_id"])
    return {
        "available": True,
        "tracking_2014_15": {
            "rows": int(len(df)),
            "mean_shot_distance": float(df["shot_distance"].mean()),
            "three_point_rate": float(df["shot_value"].eq(3).mean()),
            "fg_pct": float(df["made"].mean()),
            "unique_shooters": int(df["shooter_id"].nunique()),
        },
        "modern_2020_26": {
            "rows": int(len(modern)),
            "mean_shot_distance": float(modern["shot_distance"].mean()),
            "three_point_rate": float(modern["shot_value"].eq(3).mean()),
            "fg_pct": float(modern["shot_made"].mean()),
            "unique_shooters": int(modern["player_id"].nunique()),
        },
    }


def modern_profile_coverage(selected_spec: ModelSpec) -> dict[str, Any]:
    path = PROCESSED_DEF / "defender_season_profiles.parquet"
    if not path.exists():
        return {"available": False}
    profiles = pd.read_parquet(path)
    modern = profiles[profiles["season"].isin(["2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26"])]
    pt_required = [c for c in selected_spec.columns if c.startswith("defender_pt_")]
    equivalent = []
    for col in pt_required:
        c = col.replace("defender_pt_", "").replace("_d_fg_pct", "_d_fg_pct").replace("_pct_plusminus", "_pct_plusminus")
        if c in modern.columns:
            equivalent.append(c)
    if equivalent:
        complete = modern[equivalent].notna().all(axis=1)
        coverage = float(complete.mean())
    else:
        coverage = 1.0
    examples = modern.sort_values("season").tail(200)[["defender_id", "defender_name", "season", "season_type"]].drop_duplicates("defender_id").head(10).to_dict(orient="records")
    return {"available": True, "modern_rows": int(len(modern)), "required_profile_columns": pt_required, "mapped_profile_columns": equivalent, "complete_required_profile_rate": coverage, "examples": examples}


def write_report(metrics: pd.DataFrame, summary: dict[str, Any], analyses: dict[str, Any]) -> None:
    display_cols = [
        "model_key",
        "label",
        "estimator",
        "calibration",
        "validation_log_loss",
        "test_log_loss",
        "validation_brier",
        "test_brier",
        "test_roc_auc",
        "test_ece",
    ]
    table = simple_markdown_table(metrics.sort_values("validation_log_loss")[display_cols])
    lines = [
        "# Defensive Shot Quality Experiment v1",
        "",
        "Experimental only. Production model, Decision Policy v1, production inference, and frontend were not changed.",
        "",
        "## Chronological Split",
    ]
    for split, info in summary["split"].items():
        if isinstance(info, dict):
            lines.append(f"- {split}: {info['rows']:,} rows, {info['games']:,} games, {info['date_min']} to {info['date_max']}")
    lines += [
        "",
        "## Model Results",
        table,
        "",
        "## Recommendation",
        summary["recommendation"]["classification"],
        "",
        summary["recommendation"]["text"],
    ]
    (REPORT_DIR / "DEFENSIVE_SHOT_QUALITY_EXPERIMENT_V1.md").write_text("\n".join(lines), encoding="utf-8")


def simple_markdown_table(df: pd.DataFrame) -> str:
    rows = []
    cols = list(df.columns)
    rows.append("| " + " | ".join(cols) + " |")
    rows.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in df.iterrows():
        vals = []
        for col in cols:
            value = row[col]
            if isinstance(value, float):
                vals.append(f"{value:.6f}")
            else:
                vals.append(str(value))
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join(rows)


def classify_recommendation(metrics: pd.DataFrame, selected: dict[str, Any]) -> dict[str, str]:
    family_best = metrics.sort_values("validation_log_loss").groupby("family_key").head(1).set_index("family_key")
    a = family_best.loc["A"]
    d = family_best.loc["D"] if "D" in family_best.index else a
    improvement = float(a["test_log_loss"] - d["test_log_loss"])
    if improvement > 0.006:
        label = "A. DEFENSIVE CONTEXT CLEARLY IMPROVES SHOT QUALITY"
    elif improvement > 0.001:
        label = "B. SMALL BUT USEFUL IMPROVEMENT"
    elif improvement >= -0.001:
        label = "C. NO MEANINGFUL IMPROVEMENT"
    else:
        label = "D. RESULTS TOO UNSTABLE / DOMAIN SHIFT TOO LARGE"
    text = "Closest-defender distance and creation context should be considered experimental inputs only until modern shot-level tracking validation is available."
    return {"classification": label, "text": text}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-pt", action="store_true")
    parser.add_argument("--skip-pt", action="store_true")
    parser.add_argument("--bootstrap", type=int, default=200)
    args = parser.parse_args()
    ensure_dirs()

    pt_profiles = pd.DataFrame()
    pt_summary: dict[str, Any] = {"success": False, "reason": "skipped"}
    if not args.skip_pt:
        print("Preparing PT Defend as-of snapshots...", flush=True)
        pt_profiles, pt_summary = download_pt_asof_snapshots(force=args.force_pt)

    print("Loading and preparing tracking shot data...", flush=True)
    df, data_audit = load_and_prepare(pt_profiles if not pt_profiles.empty else None)
    df, split_summary = chronological_split(df)
    print("Building leakage-safe shooter and defender priors...", flush=True)
    df, prior_audit = add_game_priors(df)
    pt_available = not pt_profiles.empty and any(c.startswith("defender_pt_") for c in df.columns)
    specs = feature_specs(df, pt_available)
    print("Running model grid...", flush=True)
    metrics, selected, models, spec_by_key = run_models(df, specs)
    print("Generating predictions and analyses...", flush=True)
    preds = add_predictions(df, models, selected)
    metrics_path = ANALYSIS_DEF / "defensive_shot_quality_model_metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    preds.to_parquet(ANALYSIS_DEF / "defensive_shot_quality_predictions.parquet", index=False)
    selected_spec = spec_by_key[str(metrics.loc[metrics["model_key"].eq(selected["selected_model_key"]), "family_key"].iloc[0])]
    importance = feature_importance(models[selected["selected_model_key"]], selected_spec, df)
    importance.to_csv(ANALYSIS_DEF / "defensive_shot_quality_feature_importance.csv", index=False)
    analyses = analysis_tables(preds, selected["selected_model_key"])
    analyses["controlled_pressure_grid"] = controlled_pressure_grid(models[selected["selected_model_key"]], selected_spec, df)
    print("Running game-level bootstrap comparisons...", flush=True)
    boot = bootstrap_comparisons(preds, selected, metrics, n=args.bootstrap)
    domain = domain_shift(df)
    modern_cov = modern_profile_coverage(selected_spec)
    recommendation = classify_recommendation(metrics, selected)

    artifact_path = MODEL_DIR / "experimental_defensive_shot_quality_v1.joblib"
    joblib.dump(
        {
            "model": models[selected["selected_model_key"]],
            "selected_model_key": selected["selected_model_key"],
            "spec": selected_spec,
            "version": "EXPERIMENTAL defensive-shot-quality-v1 NOT PRODUCTION",
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        artifact_path,
    )
    policy = {
        "version": "defensive_shot_quality_v1_experimental",
        "not_production": True,
        "selected_model_key": selected["selected_model_key"],
        "features": selected_spec.columns,
        "invalid_touch_time_policy": "negative touch_time set to missing with touch_time_invalid and touch_time_missing flags",
        "shot_clock_policy": "missing preserved with shot_clock_missing flag",
        "chronology_policy": "complete games split chronologically; shooter/defender priors use previous games only",
    }
    (METADATA_DIR / "defensive_shot_quality_feature_policy.json").write_text(json.dumps(policy, indent=2), encoding="utf-8")

    summary = {
        "data_audit": data_audit,
        "split": split_summary,
        "prior_audit": prior_audit,
        "pt_asof": pt_summary,
        "selected": selected,
        "metrics": metrics.to_dict(orient="records"),
        "bootstrap": boot,
        "analyses": analyses,
        "domain_shift": domain,
        "modern_profile_coverage": modern_cov,
        "recommendation": recommendation,
        "artifact_path": str(artifact_path.relative_to(ROOT)),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (REPORT_DIR / "DEFENSIVE_SHOT_QUALITY_METRICS.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    write_report(metrics, summary, analyses)
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
