#!/usr/bin/env python3
"""Isolated V1.1 experimental defensive-context inference helpers."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from inference_core import InferenceError, canonical_shot_distance


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "defense"))
import run_defensive_shot_quality_v11  # noqa: F401,E402 - required for joblib custom classes.
from run_defensive_shot_quality_v11 import support_diagnostic, training_stats  # noqa: E402


ARTIFACT = ROOT / "artifacts" / "models" / "experimental_defensive_shot_quality_v11.joblib"
POLICY = ROOT / "data" / "metadata" / "defensive_shot_quality_v11_feature_policy.json"
PRED = ROOT / "data" / "analysis" / "defense" / "defensive_shot_quality_predictions.parquet"
PLAYER_PROFILES = ROOT / "artifacts" / "inference" / "player_profiles.parquet"


def exp_distance_bucket(distance: float) -> str:
    if distance <= 4:
        return "0-4"
    if distance <= 10:
        return "4-10"
    if distance <= 16:
        return "10-16"
    if distance <= 23.75:
        return "16-24"
    return "24+"


def production_distance_bucket(distance: float) -> str:
    if distance <= 3:
        return "0-3"
    if distance <= 8:
        return "4-8"
    if distance <= 16:
        return "9-16"
    if distance <= 24:
        return "17-24"
    return "25+"


class ExperimentalDefenseV11Engine:
    def __init__(self) -> None:
        self.bundle = joblib.load(ARTIFACT)
        self.policy = json.loads(POLICY.read_text(encoding="utf-8"))
        train = pd.read_parquet(PRED)
        train = train[train["split"].eq("train")]
        self.stats = training_stats(train, self.bundle["spec"].columns)
        self.player_profiles = pd.read_parquet(PLAYER_PROFILES)
        self.player_profiles["player_id"] = self.player_profiles["player_id"].astype(str)
        self.player_profiles["distance_bucket"] = self.player_profiles["distance_bucket"].astype(str)

    def feature_contract(self) -> dict[str, Any]:
        spec = self.bundle["spec"]
        categorical = set(spec.categorical)
        numeric = [c for c in spec.columns if c not in categorical]
        return {
            "modelVersion": "defensive_shot_quality_v11",
            "artifactVersion": self.bundle["version"],
            "status": "EXPERIMENTAL",
            "selectedCandidateKey": self.bundle["selected_candidate_key"],
            "selectedVariant": self.policy.get("selected_variant"),
            "trainingEra": "2014-15 shot-level tracking data",
            "columns": spec.columns,
            "numeric": numeric,
            "categorical": spec.categorical,
            "usesAttemptCounts": spec.uses_attempt_counts,
            "usesDefenderProfile": spec.uses_defender_profile,
            "modelInputs": {
                "tracking": ["closestDefenderDistance", "shotClock", "dribbles", "touchTime"],
                "shooterRates": [
                    "shooter_prior_fg_pct_shrunk",
                    "shooter_prior_2p_pct_shrunk",
                    "shooter_prior_3p_pct_shrunk",
                    "shooter_distance_prior_fg_pct_shrunk",
                ],
                "shotContext": ["shotDistance", "shotValue", "period", "secondsRemaining", "isHome", "distanceBucket"],
            },
            "excludedFeatures": [
                "shooter_prior_fga",
                "shooter_prior_2pa",
                "shooter_prior_3pa",
                "shooter_distance_prior_attempts",
                "defender_pt_overall_d_fg_pct",
                "defender_pt_overall_d_fga",
                "defender_pt_overall_normal_fg_pct",
                "defender_pt_overall_pct_plusminus",
            ],
            "defenderProfile": {
                "usage": "context-only",
                "probabilityFeature": False,
                "closestDefenderDistance": "model feature",
            },
            "supportBehavior": {
                "IN_DISTRIBUTION": "show experimental probability",
                "LIMITED_SUPPORT": "show experimental probability with warning",
                "OUT_OF_DISTRIBUTION": "hide headline experimental probability in normal UI",
            },
            "preprocessing": {
                "calibration": "sigmoid",
                "missingFlags": ["shot_clock_missing", "touch_time_missing", "touch_time_invalid"],
                "unknownCategories": "serialized pipeline handles distance_bucket categories",
            },
        }

    def analyze(self, payload: dict[str, Any]) -> dict[str, Any]:
        row, shooter_meta = self._row_from_payload(payload)
        frame = pd.DataFrame([row])
        raw = float(self.bundle["model"].raw_predict_proba(frame)[0, 1])
        p = float(self.bundle["model"].predict_proba(frame)[0, 1])
        sparse = self._is_sparse_context(row)
        support = support_diagnostic(row, self.stats, sparse_context=sparse, semantic_mismatch=False)
        displayable = support["supportStatus"] in {"IN_DISTRIBUTION", "LIMITED_SUPPORT"}
        return {
            "status": "EXPERIMENTAL",
            "modelVersion": "defensive_shot_quality_v11",
            "artifactVersion": self.bundle["version"],
            "selectedCandidateKey": self.bundle["selected_candidate_key"],
            "makeProbability": p if displayable else None,
            "displayPolicy": "SHOW_PROBABILITY" if displayable else "HIDE_ABSOLUTE_PROBABILITY",
            "diagnostics": {
                "rawProbability": raw,
                "internalProbability": p,
            },
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
            "shooterColdStart": shooter_meta["coldStart"],
            "mappedShooterFeatures": shooter_meta["mappedFeatures"],
            "shooterProfile": shooter_meta,
            "pressureSensitivity": self._pressure_sensitivity(row),
            **support,
        }

    def _row_from_payload(self, payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        player_id = str(payload.get("playerId", "")).strip()
        if not player_id:
            raise InferenceError(400, "playerId is required")
        shot_value = int(payload.get("shotValue", 0))
        if shot_value not in {2, 3}:
            raise InferenceError(400, "shotValue must be 2 or 3")
        shot_distance = self._shot_distance(payload)
        period = int(payload.get("period", 4))
        if period < 1 or period > 10:
            raise InferenceError(400, "period must be between 1 and 10")
        seconds = _num(payload.get("secondsRemaining", payload.get("secondsRemainingPeriod", 240)), "secondsRemaining")
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

        shooter_features, shooter_meta = self._shooter_features(player_id, shot_distance)
        exp_bucket = exp_distance_bucket(shot_distance)
        row = {
            "shot_distance": int(shot_distance),
            "shot_value": shot_value,
            "period": period,
            "seconds_remaining_period": float(seconds),
            "is_home": bool(payload.get("isHome", True)),
            "closest_defender_distance": float(closest),
            "shot_clock": None if shot_clock_missing else float(shot_clock),
            "dribbles": int(round(dribbles)),
            "touch_time": None if touch_missing else float(touch),
            "shot_clock_missing": bool(shot_clock_missing),
            "touch_time_missing": bool(touch_missing),
            "touch_time_invalid": False,
            "distance_bucket": exp_bucket,
            **shooter_features,
        }
        return {col: row.get(col) for col in self.bundle["spec"].columns}, shooter_meta

    def _shooter_features(self, player_id: str, shot_distance: int) -> tuple[dict[str, Any], dict[str, Any]]:
        match = self.player_profiles[
            self.player_profiles["player_id"].eq(player_id)
            & self.player_profiles["distance_bucket"].eq(production_distance_bucket(shot_distance))
        ]
        source = "exact_player_distance_bucket"
        if match.empty:
            match = self.player_profiles[self.player_profiles["player_id"].eq(player_id)]
            source = "player_profile_with_population_distance_fallback"
        if match.empty:
            match = self.player_profiles[self.player_profiles["player_id"].eq("__COLD_START__")]
            source = "cold_start_population_profile"
        profile = match.iloc[0]
        cold_start = bool(profile.get("player_cold_start", False)) or source == "cold_start_population_profile"
        features = {
            "shooter_prior_fg_pct_shrunk": float(profile["player_prior_fg_pct_shrunk"]),
            "shooter_prior_2p_pct_shrunk": float(profile["player_prior_2p_pct_shrunk"]),
            "shooter_prior_3p_pct_shrunk": float(profile["player_prior_3p_pct_shrunk"]),
            "shooter_distance_prior_fg_pct_shrunk": float(profile["player_distance_prior_fg_pct_shrunk"]),
        }
        meta = {
            "playerId": str(player_id),
            "playerName": profile.get("player_name"),
            "coldStart": cold_start,
            "source": source,
            "productionDistanceBucket": production_distance_bucket(shot_distance),
            "mappedFeatures": features,
        }
        return features, meta

    def _pressure_sensitivity(self, row: dict[str, Any]) -> list[dict[str, Any]]:
        sensitivity = []
        for distance in [1, 2, 4, 6, 8]:
            varied = dict(row)
            varied["closest_defender_distance"] = float(distance)
            raw = float(self.bundle["model"].raw_predict_proba(pd.DataFrame([varied]))[0, 1])
            p = float(self.bundle["model"].predict_proba(pd.DataFrame([varied]))[0, 1])
            support = support_diagnostic(varied, self.stats, sparse_context=self._is_sparse_context(varied), semantic_mismatch=False)
            displayable = support["supportStatus"] in {"IN_DISTRIBUTION", "LIMITED_SUPPORT"}
            sensitivity.append(
                {
                    "closestDefenderDistance": float(distance),
                    "makeProbability": p if displayable else None,
                    "supportStatus": support["supportStatus"],
                    "supportWarnings": support["supportWarnings"],
                    "outOfRangeFeatures": support["outOfRangeFeatures"],
                    "sparseContext": support["sparseContext"],
                    "diagnostics": {"rawProbability": raw, "internalProbability": p},
                }
            )
        return sensitivity

    @staticmethod
    def _shot_distance(payload: dict[str, Any]) -> int:
        if payload.get("shotDistance") not in {None, ""}:
            distance = int(float(payload["shotDistance"]))
        else:
            loc_x = _num(payload.get("locX", 0.0), "locX")
            loc_y = _num(payload.get("locY", 0.0), "locY")
            distance = int(canonical_shot_distance(loc_x, loc_y))
        if distance < 0 or distance > 100:
            raise InferenceError(400, "shotDistance must be between 0 and 100")
        return distance

    @staticmethod
    def _is_sparse_context(row: dict[str, Any]) -> bool:
        return bool(row["shot_value"] == 2 and row["shot_distance"] <= 4 and row["closest_defender_distance"] >= 5)


def _num(value: Any, name: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise InferenceError(400, f"{name} must be numeric") from exc
    if not math.isfinite(out):
        raise InferenceError(400, f"{name} must be finite")
    return out
