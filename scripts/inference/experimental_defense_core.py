#!/usr/bin/env python3
"""Experimental defensive tracking inference for the Decision Lab.

This module is intentionally separate from production inference. It adapts
modern Decision Lab inputs into the 2014/15 Defensive Shot Quality v1 feature
contract and exposes only secondary, experimental estimates.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from inference_core import InferenceError, canonical_shot_distance


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "defense"))
import run_defensive_shot_quality_experiment_v1  # noqa: F401,E402 - required for joblib custom classes.

ARTIFACT_PATH = ROOT / "artifacts" / "models" / "experimental_defensive_shot_quality_v1.joblib"
PLAYER_PROFILE_PATH = ROOT / "artifacts" / "inference" / "player_profiles.parquet"
DEFENDER_PROFILE_PATH = ROOT / "data" / "processed" / "defense" / "defender_season_profiles.parquet"
PT_CATEGORY_PATH = ROOT / "data" / "processed" / "defense" / "pt_defend_season_category.parquet"

LIMITATIONS = [
    "Experimental model trained on 2014-15 shot-level tracking data.",
    "Not production calibrated for modern NBA decision recommendations.",
    "Defender profile signal was weak/unstable in the research experiment.",
    "Probability is an associational tracking-context estimate, not causal impact.",
]


def exp_distance_bucket(distance: float) -> str:
    d = float(distance)
    if d <= 4:
        return "0-4"
    if d <= 10:
        return "4-10"
    if d <= 16:
        return "10-16"
    if d <= 23.75:
        return "16-24"
    return "24+"


def production_distance_bucket(distance: float) -> str:
    d = float(distance)
    if d <= 3:
        return "0-3"
    if d <= 8:
        return "4-8"
    if d <= 16:
        return "9-16"
    if d <= 24:
        return "17-24"
    return "25+"


def _num(value: Any, name: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise InferenceError(400, f"{name} must be numeric") from exc
    if not math.isfinite(out):
        raise InferenceError(400, f"{name} must be finite")
    return out


@dataclass
class ExperimentalDefenseArtifacts:
    bundle: dict[str, Any]
    player_profiles: pd.DataFrame
    defender_profiles: pd.DataFrame
    pt_categories: pd.DataFrame


class ExperimentalDefenseEngine:
    def __init__(
        self,
        artifact_path: Path = ARTIFACT_PATH,
        player_profile_path: Path = PLAYER_PROFILE_PATH,
        defender_profile_path: Path = DEFENDER_PROFILE_PATH,
        pt_category_path: Path = PT_CATEGORY_PATH,
    ) -> None:
        missing = [str(p) for p in [artifact_path, player_profile_path, defender_profile_path, pt_category_path] if not p.exists()]
        if missing:
            raise FileNotFoundError("Missing experimental defense artifacts: " + ", ".join(missing))
        bundle = joblib.load(artifact_path)
        players = pd.read_parquet(player_profile_path)
        defenders = pd.read_parquet(defender_profile_path)
        pt_categories = pd.read_parquet(pt_category_path)
        players["player_id"] = players["player_id"].astype(str)
        players["distance_bucket"] = players["distance_bucket"].astype(str)
        defenders["defender_id"] = defenders["defender_id"].astype(str)
        defenders["season"] = defenders["season"].astype(str)
        defenders["season_type"] = defenders["season_type"].astype(str)
        pt_categories["defender_id"] = pt_categories["defender_id"].astype(str)
        pt_categories["season"] = pt_categories["season"].astype(str)
        pt_categories["season_type"] = pt_categories["season_type"].astype(str)
        self.artifacts = ExperimentalDefenseArtifacts(bundle, players, defenders, pt_categories)

    def feature_contract(self) -> dict[str, Any]:
        spec = self.artifacts.bundle["spec"]
        categorical = set(spec.categorical)
        numeric = [c for c in spec.columns if c not in categorical]
        return {
            "modelVersion": self.artifacts.bundle["version"],
            "selectedModelKey": self.artifacts.bundle["selected_model_key"],
            "estimator": "HistGradientBoostingClassifier",
            "calibration": "isotonic",
            "columns": spec.columns,
            "numeric": numeric,
            "categorical": spec.categorical,
            "preprocessing": {
                "numeric": "SimpleImputer(strategy='median', add_indicator=True)",
                "categorical": "SimpleImputer(strategy='most_frequent') + OrdinalEncoder(unknown_value=-1)",
                "missing": "Model pipeline handles numeric missingness; unknown categorical values encode as -1.",
            },
        }

    def analyze(self, payload: dict[str, Any]) -> dict[str, Any]:
        row, shooter_meta, defender_meta = self._feature_row(payload)
        p_make = self._predict_row(row)
        sensitivity = []
        for distance in [1, 2, 4, 6, 8]:
            varied = dict(row)
            varied["closest_defender_distance"] = float(distance)
            sensitivity.append({"closestDefenderDistance": float(distance), "makeProbability": self._predict_row(varied)})
        return {
            "status": "EXPERIMENTAL",
            "modelVersion": self.artifacts.bundle["version"],
            "selectedModelKey": self.artifacts.bundle["selected_model_key"],
            "makeProbability": p_make,
            "context": {
                "shotDistance": row["shot_distance"],
                "distanceBucket": row["distance_bucket"],
                "shotValue": row["shot_value"],
                "period": row["period"],
                "secondsRemainingPeriod": row["seconds_remaining_period"],
                "isHome": bool(row["is_home"]),
                "closestDefenderDistance": row["closest_defender_distance"],
                "shotClock": row["shot_clock"],
                "dribbles": row["dribbles"],
                "touchTime": row["touch_time"],
            },
            "shooterProfile": shooter_meta,
            "defenderProfile": defender_meta,
            "featureVector": row,
            "pressureSensitivity": sensitivity,
            "limitations": LIMITATIONS,
        }

    def defenders(self, search: str = "", limit: int = 25) -> dict[str, Any]:
        q = search.strip().lower()
        frame = self.artifacts.defender_profiles.copy()
        if q:
            frame = frame[
                frame["defender_name"].astype(str).str.lower().str.contains(q, na=False)
                | frame["defender_id"].astype(str).str.contains(q, na=False)
            ]
        sort_cols = ["defender_name", "season", "season_type"]
        latest = frame.sort_values(sort_cols).groupby("defender_id", as_index=False).tail(1)
        latest = latest.sort_values(["defender_name", "season"])
        rows = []
        for _, row in latest.head(max(1, min(int(limit), 100))).iterrows():
            rows.append(
                {
                    "defenderId": str(row["defender_id"]),
                    "defenderName": str(row["defender_name"]),
                    "team": row.get("team_abbreviation"),
                    "season": row.get("season"),
                    "seasonType": row.get("season_type"),
                    "overallDfgPct": self._optional_float(row.get("overall_d_fg_pct")),
                    "overallDefendedFga": self._optional_float(row.get("overall_d_fga")),
                }
            )
        return {"items": rows, "total": int(len(latest))}

    def defender_profile(self, defender_id: str, season: str = "2025-26", season_type: str = "regular_season") -> dict[str, Any]:
        row = self._resolve_defender(defender_id, season, season_type)
        if row is None:
            raise InferenceError(404, f"Defender profile not found for defenderId={defender_id}")
        return {"defenderProfile": self._defender_metadata(row, cold_start=False)}

    def _feature_row(self, payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        player_id = str(payload.get("playerId", "")).strip()
        if not player_id:
            raise InferenceError(400, "playerId is required")
        shot_value = int(payload.get("shotValue", 0))
        if shot_value not in {2, 3}:
            raise InferenceError(400, "shotValue must be 2 or 3")
        loc_x = _num(payload.get("locX", 0.0), "locX")
        loc_y = _num(payload.get("locY", 0.0), "locY")
        shot_distance = int(payload.get("shotDistance") or canonical_shot_distance(loc_x, loc_y))
        if shot_distance < 0 or shot_distance > 100:
            raise InferenceError(400, "shotDistance must be between 0 and 100")
        period = int(payload.get("period", 0))
        if period < 1 or period > 10:
            raise InferenceError(400, "period must be between 1 and 10")
        seconds = _num(payload.get("secondsRemaining", payload.get("secondsRemainingPeriod", 0)), "secondsRemaining")
        if seconds < 0 or seconds > 720:
            raise InferenceError(400, "secondsRemaining must be between 0 and 720")
        closest = _num(payload.get("closestDefenderDistance"), "closestDefenderDistance")
        if closest < 0 or closest > 60:
            raise InferenceError(400, "closestDefenderDistance must be between 0 and 60")
        shot_clock_raw = payload.get("shotClock")
        shot_clock_missing = shot_clock_raw is None or shot_clock_raw == ""
        shot_clock = None if shot_clock_missing else _num(shot_clock_raw, "shotClock")
        if shot_clock is not None and (shot_clock < 0 or shot_clock > 24):
            raise InferenceError(400, "shotClock must be between 0 and 24")
        dribbles = _num(payload.get("dribbles", 0), "dribbles")
        if dribbles < 0:
            raise InferenceError(400, "dribbles must be non-negative")
        touch_raw = payload.get("touchTime")
        touch_missing = touch_raw is None or touch_raw == ""
        touch = None if touch_missing else _num(touch_raw, "touchTime")
        if touch is not None and touch < 0:
            raise InferenceError(400, "touchTime must be non-negative")

        exp_bucket = exp_distance_bucket(shot_distance)
        shooter_features, shooter_meta = self._shooter_features(player_id, shot_distance)
        defender_id = str(payload.get("defenderId", "")).strip()
        season = str(payload.get("season", "2025-26"))
        season_type = str(payload.get("seasonType", "regular_season"))
        defender_row = self._resolve_defender(defender_id, season, season_type) if defender_id else None
        defender_meta = self._defender_metadata(defender_row, cold_start=defender_row is None, requested_id=defender_id or None)

        row = {
            "shot_distance": int(shot_distance),
            "shot_value": int(shot_value),
            "period": int(period),
            "seconds_remaining_period": float(seconds),
            "is_home": bool(payload.get("isHome", True)),
            **shooter_features,
            "distance_bucket": exp_bucket,
            "closest_defender_distance": float(closest),
            "shot_clock": None if shot_clock_missing else float(shot_clock),
            "dribbles": int(round(dribbles)),
            "touch_time": None if touch_missing else float(touch),
            "shot_clock_missing": bool(shot_clock_missing),
            "touch_time_missing": bool(touch_missing),
            "touch_time_invalid": False,
            "defender_pt_overall_d_fg_pct": defender_meta.get("overallDfgPct"),
            "defender_pt_overall_d_fga": defender_meta.get("overallDefendedFga"),
            "defender_pt_overall_normal_fg_pct": defender_meta.get("overallNormalFgPct"),
            "defender_pt_overall_pct_plusminus": defender_meta.get("overallPctPlusminus"),
        }
        spec_cols = self.artifacts.bundle["spec"].columns
        return {col: row.get(col) for col in spec_cols}, shooter_meta, defender_meta

    def _shooter_features(self, player_id: str, shot_distance: int) -> tuple[dict[str, Any], dict[str, Any]]:
        profiles = self.artifacts.player_profiles
        prod_bucket = production_distance_bucket(shot_distance)
        exact = profiles[profiles["player_id"].eq(player_id) & profiles["distance_bucket"].eq(prod_bucket)]
        player = profiles[profiles["player_id"].eq(player_id)]
        if not exact.empty:
            profile = exact.iloc[0].to_dict()
            cold = bool(profile.get("player_cold_start", False))
            source = "exact_player_distance_bucket"
        elif not player.empty:
            profile = player.iloc[0].to_dict()
            cold = bool(profile.get("player_cold_start", False))
            source = "player_profile_with_population_distance_fallback"
        else:
            profile = profiles[profiles["player_id"].eq("__COLD_START__")].iloc[0].to_dict()
            profile["player_id"] = player_id
            profile["player_name"] = "Unknown Player"
            cold = True
            source = "cold_start_population_profile"
        features = {
            "shooter_prior_fga": int(profile["player_prior_fga"]),
            "shooter_prior_fg_pct_shrunk": float(profile["player_prior_fg_pct_shrunk"]),
            "shooter_prior_2pa": int(profile["player_prior_2pa"]),
            "shooter_prior_2p_pct_shrunk": float(profile["player_prior_2p_pct_shrunk"]),
            "shooter_prior_3pa": int(profile["player_prior_3pa"]),
            "shooter_prior_3p_pct_shrunk": float(profile["player_prior_3p_pct_shrunk"]),
            "shooter_distance_prior_attempts": int(profile["player_distance_prior_attempts"]),
            "shooter_distance_prior_fg_pct_shrunk": float(profile["player_distance_prior_fg_pct_shrunk"]),
            "shooter_cold_start": bool(cold),
        }
        meta = {
            "playerId": str(player_id),
            "playerName": profile.get("player_name"),
            "coldStart": bool(cold),
            "source": source,
            "productionDistanceBucket": prod_bucket,
            "mappedFeatures": features,
        }
        return features, meta

    def _resolve_defender(self, defender_id: str, season: str, season_type: str) -> pd.Series | None:
        if not defender_id:
            return None
        frame = self.artifacts.defender_profiles[self.artifacts.defender_profiles["defender_id"].eq(str(defender_id))]
        if frame.empty:
            return None
        preferred = frame[frame["season_type"].eq(season_type)]
        if preferred.empty:
            preferred = frame[frame["season_type"].eq("regular_season")]
        if preferred.empty:
            preferred = frame
        eligible = preferred[preferred["season"].le(str(season))]
        if eligible.empty:
            eligible = preferred
        return eligible.sort_values(["season", "overall_d_fga"]).iloc[-1]

    def _defender_metadata(self, row: pd.Series | None, cold_start: bool, requested_id: str | None = None) -> dict[str, Any]:
        if row is None:
            return {
                "defenderId": requested_id,
                "defenderName": "No defender selected" if requested_id is None else "Unknown defender",
                "coldStart": True,
                "source": "model_median_imputation",
                "overallDfgPct": None,
                "overallNormalFgPct": None,
                "overallPctPlusminus": None,
                "overallDefendedFga": None,
                "twoPtFrequency": None,
                "twoPtDfgPct": None,
                "twoPtNormalFgPct": None,
                "twoPtDefendedFga": None,
                "threePtFrequency": None,
                "threePtDfgPct": None,
                "threePtNormalFgPct": None,
                "threePtDefendedFga": None,
                "lt6Frequency": None,
                "lt6DfgPct": None,
                "lt6NormalFgPct": None,
                "lt6DefendedFga": None,
                "lt10Frequency": None,
                "lt10DfgPct": None,
                "lt10NormalFgPct": None,
                "lt10DefendedFga": None,
                "gt15Frequency": None,
                "gt15DfgPct": None,
                "gt15NormalFgPct": None,
                "gt15DefendedFga": None,
            }
        category = self._category_values(str(row.get("defender_id")), str(row.get("season")), str(row.get("season_type")))
        return {
            "defenderId": str(row.get("defender_id")),
            "defenderName": row.get("defender_name"),
            "coldStart": bool(cold_start),
            "source": "modern_defender_season_profiles",
            "season": row.get("season"),
            "seasonType": row.get("season_type"),
            "team": row.get("team_abbreviation"),
            "overallDfgPct": self._optional_float(row.get("overall_d_fg_pct")),
            "overallNormalFgPct": self._optional_float(row.get("overall_normal_fg_pct")),
            "overallPctPlusminus": self._optional_float(row.get("overall_pct_plusminus")),
            "overallDefendedFga": self._optional_float(row.get("overall_d_fga")),
            "twoPtFrequency": self._optional_float(row.get("two_pt_frequency")),
            "twoPtDfgPct": category["two_pt"]["dfg"],
            "twoPtNormalFgPct": category["two_pt"]["normal"],
            "twoPtDefendedFga": category["two_pt"]["fga"],
            "threePtFrequency": self._optional_float(row.get("three_pt_frequency")),
            "threePtDfgPct": category["three_pt"]["dfg"],
            "threePtNormalFgPct": category["three_pt"]["normal"],
            "threePtDefendedFga": category["three_pt"]["fga"],
            "lt6Frequency": self._optional_float(row.get("lt6_frequency")),
            "lt6DfgPct": category["lt6"]["dfg"],
            "lt6NormalFgPct": category["lt6"]["normal"],
            "lt6DefendedFga": category["lt6"]["fga"],
            "lt10Frequency": self._optional_float(row.get("lt10_frequency")),
            "lt10DfgPct": category["lt10"]["dfg"],
            "lt10NormalFgPct": category["lt10"]["normal"],
            "lt10DefendedFga": category["lt10"]["fga"],
            "gt15Frequency": self._optional_float(row.get("gt15_frequency")),
            "gt15DfgPct": category["gt15"]["dfg"],
            "gt15NormalFgPct": category["gt15"]["normal"],
            "gt15DefendedFga": category["gt15"]["fga"],
        }

    def _category_values(self, defender_id: str, season: str, season_type: str) -> dict[str, dict[str, float | None]]:
        frame = self.artifacts.pt_categories
        rows = frame[
            frame["defender_id"].eq(str(defender_id))
            & frame["season"].eq(str(season))
            & frame["season_type"].eq(str(season_type))
        ]
        mapping = {
            "two_pt": ("2_pointers", "FG2_PCT", "NS_FG2_PCT", "FG2A"),
            "three_pt": ("3_pointers", "FG3_PCT", "NS_FG3_PCT", "FG3A"),
            "lt6": ("less_than_6ft", "LT_06_PCT", "NS_LT_06_PCT", "FGA_LT_06"),
            "lt10": ("less_than_10ft", "LT_10_PCT", "NS_LT_10_PCT", "FGA_LT_10"),
            "gt15": ("greater_than_15ft", "GT_15_PCT", "NS_GT_15_PCT", "FGA_GT_15"),
        }
        out: dict[str, dict[str, float | None]] = {}
        for key, (category, dfg_col, normal_col, fga_col) in mapping.items():
            cat = rows[rows["defense_category"].eq(category)]
            if cat.empty:
                out[key] = {"dfg": None, "normal": None, "fga": None}
                continue
            row = cat.iloc[0]
            out[key] = {
                "dfg": self._optional_float(row.get(dfg_col)),
                "normal": self._optional_float(row.get(normal_col)),
                "fga": self._optional_float(row.get(fga_col)),
            }
        return out

    def _predict_row(self, row: dict[str, Any]) -> float:
        frame = pd.DataFrame([row])
        return float(self.artifacts.bundle["model"].predict_proba(frame)[0, 1])

    @staticmethod
    def _optional_float(value: Any) -> float | None:
        if value is None or pd.isna(value):
            return None
        return float(value)
