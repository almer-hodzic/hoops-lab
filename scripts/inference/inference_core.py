#!/usr/bin/env python3
"""Production inference core for the NBA Clutch Agent.

The .NET app calls the HTTP wrapper, while tests import this module directly.
The model is loaded once by ``ProductionInferenceEngine`` and all request-time
feature assembly is kept here to avoid backend/API drift.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BUNDLE_PATH = ROOT / "artifacts" / "inference" / "production_inference_bundle.joblib"
PROFILE_PATH = ROOT / "artifacts" / "inference" / "player_profiles.parquet"
PLAYER_METADATA_PATH = ROOT / "artifacts" / "inference" / "player_metadata.json"
DECISION_ANALYSIS_PATH = ROOT / "data" / "analysis" / "decision_2025_26_regular.csv"

ACTION_TYPES = {
    "Alley Oop Dunk Shot",
    "Alley Oop Layup shot",
    "Cutting Dunk Shot",
    "Cutting Finger Roll Layup Shot",
    "Cutting Layup Shot",
    "Driving Bank Hook Shot",
    "Driving Dunk Shot",
    "Driving Finger Roll Layup Shot",
    "Driving Floating Bank Jump Shot",
    "Driving Floating Jump Shot",
    "Driving Hook Shot",
    "Driving Layup Shot",
    "Driving Reverse Dunk Shot",
    "Driving Reverse Layup Shot",
    "Dunk Shot",
    "Fadeaway Bank shot",
    "Fadeaway Jump Shot",
    "Finger Roll Layup Shot",
    "Floating Jump shot",
    "Hook Bank Shot",
    "Hook Shot",
    "Jump Bank Shot",
    "Jump Shot",
    "Layup Shot",
    "Pullup Jump shot",
    "Putback Dunk Shot",
    "Putback Layup Shot",
    "Reverse Dunk Shot",
    "Reverse Layup Shot",
    "Running Alley Oop Dunk Shot",
    "Running Alley Oop Layup Shot",
    "Running Dunk Shot",
    "Running Finger Roll Layup Shot",
    "Running Jump Shot",
    "Running Layup Shot",
    "Running Pull-Up Jump Shot",
    "Running Reverse Dunk Shot",
    "Running Reverse Layup Shot",
    "Step Back Bank Jump Shot",
    "Step Back Jump shot",
    "Tip Dunk Shot",
    "Tip Layup Shot",
    "Turnaround Bank Hook Shot",
    "Turnaround Bank shot",
    "Turnaround Fadeaway Bank Jump Shot",
    "Turnaround Fadeaway shot",
    "Turnaround Hook Shot",
    "Turnaround Jump Shot",
}


class InferenceError(ValueError):
    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.message = message


def distance_bucket(distance: float) -> str:
    if distance <= 3:
        return "0-3"
    if distance <= 8:
        return "4-8"
    if distance <= 16:
        return "9-16"
    if distance <= 24:
        return "17-24"
    return "25+"


def physical_distance_feet(loc_x: float, loc_y: float) -> float:
    return math.sqrt((loc_x * loc_x) + (loc_y * loc_y)) / 10.0


def canonical_shot_distance(loc_x: float, loc_y: float) -> int:
    return int(math.floor(physical_distance_feet(loc_x, loc_y)))


def canonical_shot_type(shot_value: int) -> str:
    return "3PT Field Goal" if shot_value == 3 else "2PT Field Goal"


def infer_zone(distance: float, shot_value: int, loc_x: float | None = None, loc_y: float | None = None) -> dict[str, str]:
    x = float(loc_x or 0.0)
    y = float(loc_y or 0.0)
    d = int(math.floor(distance))
    backcourt = y > 417.5 or d >= 40
    if backcourt:
        zone_range = "Back Court Shot"
    elif shot_value == 3:
        zone_range = "24+ ft."
    elif d < 8:
        zone_range = "Less Than 8 ft."
    elif d < 16:
        zone_range = "8-16 ft."
    elif d < 24:
        zone_range = "16-24 ft."
    else:
        zone_range = "24+ ft."

    if backcourt:
        zone_basic = "Backcourt"
    elif shot_value == 3:
        if abs(x) >= 220 and y <= 85:
            zone_basic = "Left Corner 3" if x < 0 else "Right Corner 3"
        else:
            zone_basic = "Above the Break 3"
    elif d <= 3:
        zone_basic = "Restricted Area"
    elif abs(x) <= 80 and y <= 140:
        zone_basic = "In The Paint (Non-RA)"
    else:
        zone_basic = "Mid-Range"

    if backcourt:
        zone_area = "Back Court(BC)"
    elif abs(x) <= 80:
        zone_area = "Center(C)"
    elif x < 0:
        zone_area = "Left Side(L)" if y <= 80 else "Left Side Center(LC)"
    else:
        zone_area = "Right Side(R)" if y <= 80 else "Right Side Center(RC)"
    return {"shot_zone_basic": zone_basic, "shot_zone_area": zone_area, "shot_zone_range": zone_range}


def validate_action_type(action_type: str, shot_value: int, model_distance: int) -> None:
    if action_type not in ACTION_TYPES:
        raise InferenceError(400, f"Unsupported actionType: {action_type}")
    lower = action_type.lower()
    near_rim_action = any(token in lower for token in ["dunk", "layup", "hook"])
    if shot_value == 3 and near_rim_action:
        raise InferenceError(400, f"{action_type} is not valid for a 3PT attempt")
    if model_distance >= 16 and near_rim_action:
        raise InferenceError(400, f"{action_type} is not valid from {model_distance} ft")


def game_state(score_diff: float) -> str:
    if score_diff > 0:
        return "leading"
    if score_diff < 0:
        return "trailing"
    return "tied"


def recommendation(margin: float, threshold: float) -> str:
    if margin >= threshold:
        return "TAKE"
    if margin <= -threshold:
        return "DEFER"
    return "REVIEW"


@dataclass
class LoadedArtifacts:
    model: Any
    calibrator: Any
    columns: list[str]
    threshold: float
    replacement_xpps: float
    metadata: dict[str, Any]
    profiles: pd.DataFrame
    player_metadata: list[dict[str, Any]]
    analysis: pd.DataFrame


class ProductionInferenceEngine:
    def __init__(
        self,
        bundle_path: Path = BUNDLE_PATH,
        profile_path: Path = PROFILE_PATH,
        player_metadata_path: Path = PLAYER_METADATA_PATH,
        analysis_path: Path = DECISION_ANALYSIS_PATH,
    ):
        missing = [str(p) for p in [bundle_path, profile_path, player_metadata_path] if not p.exists()]
        if missing:
            raise FileNotFoundError("Missing inference artifacts: " + ", ".join(missing))

        bundle = joblib.load(bundle_path)
        profiles = pd.read_parquet(profile_path)
        profiles["player_id"] = profiles["player_id"].astype(str)
        profiles["shot_zone_basic"] = profiles["shot_zone_basic"].astype(str)
        profiles["distance_bucket"] = profiles["distance_bucket"].astype(str)
        analysis = pd.read_csv(analysis_path, dtype={"game_id": str, "event_id": str}) if analysis_path.exists() else pd.DataFrame()
        self.artifacts = LoadedArtifacts(
            model=bundle["model"],
            calibrator=bundle["calibrator"],
            columns=bundle["columns"],
            threshold=float(bundle["decision_policy"]["threshold"]),
            replacement_xpps=float(bundle["decision_policy"]["replacement_xpps"]),
            metadata=bundle["metadata"],
            profiles=profiles,
            player_metadata=json.loads(player_metadata_path.read_text(encoding="utf-8")),
            analysis=analysis,
        )

    def analyze(self, payload: dict[str, Any]) -> dict[str, Any]:
        frame, profile, cold_start = self._frame_from_payload(payload)
        raw = np.clip(self.artifacts.model.predict_proba(frame[self.artifacts.columns])[:, 1], 1e-6, 1 - 1e-6)
        p_make = self._apply_calibration(raw)[0]
        shot_value = int(frame["shot_value"].iloc[0])
        shot_xpps = float(p_make * shot_value)
        replacement = self.artifacts.replacement_xpps
        margin = shot_xpps - replacement
        rec = recommendation(margin, self.artifacts.threshold)
        reason_codes = []
        if rec == "REVIEW":
            reason_codes.append("MARGIN_WITHIN_REVIEW_BAND")
        elif rec == "TAKE":
            reason_codes.append("ABOVE_REPLACEMENT_VALUE")
        else:
            reason_codes.append("BELOW_REPLACEMENT_VALUE")
        if cold_start:
            reason_codes.append("COLD_START_PROFILE")

        player_id = str(payload.get("playerId", "0"))
        player_name = str(profile.get("player_name") or payload.get("playerName") or "Unknown Player")
        return {
            "player": {"id": player_id, "name": player_name, "team": profile.get("team_name") or profile.get("team_id")},
            "shot": {
                "value": shot_value,
                "distance": float(frame["shot_distance"].iloc[0]),
                "displayDistanceFeet": float(frame["display_distance_feet"].iloc[0]),
                "modelShotDistance": int(frame["shot_distance"].iloc[0]),
                "locX": float(frame["loc_x"].iloc[0]),
                "locY": float(frame["loc_y"].iloc[0]),
                "shotType": str(frame["shot_type"].iloc[0]),
                "actionType": str(frame["action_type"].iloc[0]),
                "zoneBasic": str(frame["shot_zone_basic"].iloc[0]),
                "zoneArea": str(frame["shot_zone_area"].iloc[0]),
                "zoneRange": str(frame["shot_zone_range"].iloc[0]),
            },
            "gameContext": {
                "period": int(frame["period"].iloc[0]),
                "secondsRemaining": float(frame["seconds_remaining_period"].iloc[0]),
                "scoreDiff": float(frame["score_diff_before"].iloc[0]),
                "gameState": str(frame["game_state_before"].iloc[0]),
                "isHome": bool(frame["is_home"].iloc[0]),
                "isOvertime": bool(frame["is_overtime"].iloc[0]),
            },
            "prediction": {"makeProbability": float(p_make), "shotXPps": shot_xpps},
            "benchmark": {"replacementXPps": replacement, "margin": margin, "threshold": self.artifacts.threshold},
            "recommendation": rec,
            "reasonCodes": reason_codes,
            "profile": {
                "coldStart": bool(cold_start),
                "priorFga": int(profile["player_prior_fga"]),
                "priorFgPctShrunk": float(profile["player_prior_fg_pct_shrunk"]),
                "prior2pa": int(profile["player_prior_2pa"]),
                "prior2pPctShrunk": float(profile["player_prior_2p_pct_shrunk"]),
                "prior3pa": int(profile["player_prior_3pa"]),
                "prior3pPctShrunk": float(profile["player_prior_3p_pct_shrunk"]),
                "zonePriorAttempts": int(profile["player_zone_prior_attempts"]),
                "zonePriorFgPctShrunk": float(profile["player_zone_prior_fg_pct_shrunk"]),
                "distancePriorAttempts": int(profile["player_distance_prior_attempts"]),
                "distancePriorFgPctShrunk": float(profile["player_distance_prior_fg_pct_shrunk"]),
            },
            "resolvedContext": {
                "shotType": str(frame["shot_type"].iloc[0]),
                "actionType": str(frame["action_type"].iloc[0]),
                "modelShotDistance": int(frame["shot_distance"].iloc[0]),
                "displayDistanceFeet": float(frame["display_distance_feet"].iloc[0]),
                "shotZoneBasic": str(frame["shot_zone_basic"].iloc[0]),
                "shotZoneArea": str(frame["shot_zone_area"].iloc[0]),
                "shotZoneRange": str(frame["shot_zone_range"].iloc[0]),
                "locX": float(frame["loc_x"].iloc[0]),
                "locY": float(frame["loc_y"].iloc[0]),
            },
            "modelVersion": self.artifacts.metadata["model_version"],
            "policyVersion": self.artifacts.metadata["policy_version"],
        }

    def players(self, search: str = "", limit: int = 25) -> dict[str, Any]:
        search_l = search.strip().lower()
        rows = self.artifacts.player_metadata
        if search_l:
            rows = [r for r in rows if search_l in str(r["player_name"]).lower() or search_l in str(r["player_id"])]
        return {"items": rows[: max(1, min(limit, 100))], "total": len(rows)}

    def player_profile(self, player_id: str) -> dict[str, Any]:
        profile_rows = self.artifacts.profiles[self.artifacts.profiles["player_id"].eq(str(player_id))]
        if profile_rows.empty:
            raise InferenceError(404, f"Player profile not found for playerId={player_id}")
        meta = next((r for r in self.artifacts.player_metadata if str(r["player_id"]) == str(player_id)), None)
        return {"player": meta, "profiles": profile_rows.head(200).to_dict(orient="records")}

    def model_metadata(self) -> dict[str, Any]:
        return {
            **self.artifacts.metadata,
            "replacementXPps": self.artifacts.replacement_xpps,
            "decisionThreshold": self.artifacts.threshold,
        }

    def historical_decisions(self, filters: dict[str, Any]) -> dict[str, Any]:
        if self.artifacts.analysis.empty:
            return {"items": [], "page": 1, "pageSize": 0, "total": 0}
        frame = self.artifacts.analysis
        if filters.get("season"):
            frame = frame[frame["season"].astype(str).eq(str(filters["season"]))]
        if filters.get("playerId"):
            frame = frame[frame["player_name"].astype(str).str.contains(str(filters["playerId"]), case=False, na=False)]
        if filters.get("player"):
            frame = frame[frame["player_name"].astype(str).str.contains(str(filters["player"]), case=False, na=False)]
        if filters.get("recommendation"):
            frame = frame[frame["decision"].astype(str).eq(str(filters["recommendation"]).upper())]
        if filters.get("clutchOnly") is not None:
            want = str(filters["clutchOnly"]).lower() in {"true", "1", "yes"}
            frame = frame[frame["is_clutch"].astype(bool).eq(want)]
        page = max(1, int(filters.get("page", 1)))
        page_size = max(1, min(int(filters.get("pageSize", 50)), 200))
        total = len(frame)
        cols = [
            "season",
            "season_type",
            "game_id",
            "event_id",
            "player_name",
            "team_name",
            "is_clutch",
            "p_make",
            "shot_xpps",
            "replacement_xpps",
            "decision_margin",
            "decision",
            "points_scored",
        ]
        rows = frame.iloc[(page - 1) * page_size : page * page_size][[c for c in cols if c in frame.columns]]
        return {"items": rows.to_dict(orient="records"), "page": page, "pageSize": page_size, "total": total}

    def _apply_calibration(self, p: np.ndarray) -> np.ndarray:
        p = np.clip(p, 1e-6, 1 - 1e-6)
        cal = self.artifacts.calibrator
        if cal is None:
            return p
        logit = np.log(p / (1 - p)).reshape(-1, 1)
        return cal.predict_proba(logit)[:, 1]

    def _frame_from_payload(self, payload: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any], bool]:
        player_id = str(payload.get("playerId", "")).strip()
        if not player_id:
            raise InferenceError(400, "playerId is required")
        shot_value = int(payload.get("shotValue", 0))
        if shot_value not in {2, 3}:
            raise InferenceError(400, "shotValue must be 2 or 3")
        loc_x = float(payload.get("locX", 0.0))
        loc_y = float(payload.get("locY", 0.0))
        display_distance = physical_distance_feet(loc_x, loc_y)
        shot_distance = canonical_shot_distance(loc_x, loc_y)
        if not math.isfinite(display_distance) or shot_distance < 0 or shot_distance > 100:
            raise InferenceError(400, "shot location must resolve to a model distance between 0 and 100")
        period = int(payload.get("period", 0))
        if period < 1 or period > 10:
            raise InferenceError(400, "period must be between 1 and 10")
        seconds = float(payload.get("secondsRemaining", payload.get("secondsRemainingPeriod", -1)))
        if not math.isfinite(seconds) or seconds < 0 or seconds > 720:
            raise InferenceError(400, "secondsRemaining must be between 0 and 720")
        score_diff = float(payload.get("scoreDiff", payload.get("scoreDiffBefore", 0.0)))
        is_home = bool(payload.get("isHome", True))
        is_overtime = bool(payload.get("isOvertime", period > 4))
        zone = infer_zone(shot_distance, shot_value, loc_x, loc_y)
        action_type = str(payload.get("actionType", "Jump Shot"))
        validate_action_type(action_type, shot_value, shot_distance)
        dist_bucket = distance_bucket(shot_distance)
        profile, cold_start = self._profile(player_id, str(zone["shot_zone_basic"]), dist_bucket)
        base = {
            "shot_distance": shot_distance,
            "display_distance_feet": display_distance,
            "loc_x": loc_x,
            "loc_y": loc_y,
            "shot_value": shot_value,
            "period": period,
            "seconds_remaining_period": seconds,
            "score_diff_before": score_diff,
            "abs_score_diff_before": abs(score_diff),
            "is_home": is_home,
            "is_overtime": is_overtime,
            "season": str(payload.get("season", "2025-26")),
            "shot_type": canonical_shot_type(shot_value),
            "action_type": action_type,
            "shot_zone_basic": str(zone["shot_zone_basic"]),
            "shot_zone_area": str(zone["shot_zone_area"]),
            "shot_zone_range": str(zone["shot_zone_range"]),
            "game_state_before": str(payload.get("gameState", game_state(score_diff))),
        }
        for col in self.artifacts.metadata["skill_features"]:
            base[col] = profile[col]
        base["player_cold_start"] = bool(cold_start)
        return pd.DataFrame([base]), profile, cold_start

    def _profile(self, player_id: str, zone_basic: str, dist_bucket: str) -> tuple[dict[str, Any], bool]:
        profiles = self.artifacts.profiles
        exact = profiles[
            profiles["player_id"].eq(player_id)
            & profiles["shot_zone_basic"].eq(zone_basic)
            & profiles["distance_bucket"].eq(dist_bucket)
        ]
        if not exact.empty:
            return exact.iloc[0].to_dict(), False
        player = profiles[profiles["player_id"].eq(player_id)]
        if not player.empty:
            row = player.iloc[0].to_dict()
            row["player_zone_prior_attempts"] = 0
            row["player_zone_prior_fg_pct"] = row["global_fg"]
            row["player_zone_prior_fg_pct_shrunk"] = row["zone_prior_mean"]
            row["player_distance_prior_attempts"] = 0
            row["player_distance_prior_fg_pct"] = row["global_fg"]
            row["player_distance_prior_fg_pct_shrunk"] = row["distance_prior_mean"]
            return row, False
        cold = profiles[profiles["player_id"].eq("__COLD_START__")].iloc[0].to_dict()
        cold["player_id"] = player_id
        cold["player_name"] = "Unknown Player"
        return cold, True
