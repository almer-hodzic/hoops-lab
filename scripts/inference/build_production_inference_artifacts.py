#!/usr/bin/env python3
"""Build production inference artifacts without retraining the shooter model."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "modeling"))
sys.path.insert(0, str(ROOT / "scripts" / "decision"))
from train_shooter_skill_v11 import (  # noqa: E402
    GENERIC_BOOLEAN,
    GENERIC_CATEGORICAL,
    GENERIC_NUMERIC,
    PRIMARY_TRAIN_SEASONS,
    SKILL_FEATURES,
    TARGET,
    apply_calibration,
    build_historical_features,
    fit_calibration,
    predict,
)
from build_decision_policy_v1 import add_context_columns  # noqa: E402
from inference_core import distance_bucket  # noqa: E402


MODEL_PATH = ROOT / "artifacts" / "models" / "shooter_skill_v11_hist_gradient_boosting__historical_shooter_profile.joblib"
DECISION_PATH = ROOT / "artifacts" / "decision" / "decision_policy_v1.joblib"
SHOT_DATA = ROOT / "data" / "processed" / "shots_enriched.parquet"
OUT_DIR = ROOT / "artifacts" / "inference"


def pct(makes: float, attempts: float, fallback: float) -> float:
    return float(makes / attempts) if attempts else float(fallback)


def shrunk(makes: float, attempts: float, prior: float, strength: float) -> float:
    return float((makes + strength * prior) / (attempts + strength))


def build_profiles(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    train = df[(df["season"].isin(PRIMARY_TRAIN_SEASONS)) & (df["season_type"].eq("regular_season"))]
    global_fg = float(train[TARGET].mean())
    global_2p = float(train.loc[train["shot_value"].eq(2), TARGET].mean())
    global_3p = float(train.loc[train["shot_value"].eq(3), TARGET].mean())
    zone_mean = train.groupby("shot_zone_basic")[TARGET].mean().to_dict()
    dist_mean = train.groupby("distance_bucket")[TARGET].mean().to_dict()
    latest = df.sort_values(["game_date", "game_id", "event_id"]).groupby("player_id", as_index=False).tail(1)
    player_totals = df.groupby("player_id").agg(
        player_name=("player_name", "last"),
        team_id=("team_id", "last"),
        team_name=("team_name", "last"),
        fga=("shot_made", "size"),
        fgm=("shot_made", "sum"),
    ).reset_index()
    two = df[df["shot_value"].eq(2)].groupby("player_id").agg(two_pa=("shot_made", "size"), two_pm=("shot_made", "sum"))
    three = df[df["shot_value"].eq(3)].groupby("player_id").agg(three_pa=("shot_made", "size"), three_pm=("shot_made", "sum"))
    player_totals = player_totals.merge(two, on="player_id", how="left").merge(three, on="player_id", how="left").fillna(0)
    current_season = "2025-26"
    season = df[df["season"].eq(current_season)].groupby("player_id").agg(season_fga=("shot_made", "size"), season_fgm=("shot_made", "sum")).reset_index()
    player_totals = player_totals.merge(season, on="player_id", how="left").fillna({"season_fga": 0, "season_fgm": 0})
    prev = df[df["season"].eq("2024-25")].groupby("player_id").agg(
        prev_fga=("shot_made", "size"),
        prev_fgm=("shot_made", "sum"),
    ).reset_index()
    prev_two = df[(df["season"].eq("2024-25")) & df["shot_value"].eq(2)].groupby("player_id").agg(prev_2pa=("shot_made", "size"), prev_2pm=("shot_made", "sum"))
    prev_three = df[(df["season"].eq("2024-25")) & df["shot_value"].eq(3)].groupby("player_id").agg(prev_3pa=("shot_made", "size"), prev_3pm=("shot_made", "sum"))
    prev = prev.merge(prev_two, on="player_id", how="left").merge(prev_three, on="player_id", how="left").fillna(0)
    player_totals = player_totals.merge(prev, on="player_id", how="left").fillna(0)
    zone_totals = df.groupby(["player_id", "shot_zone_basic"]).agg(zone_attempts=("shot_made", "size"), zone_makes=("shot_made", "sum")).reset_index()
    dist_totals = df.groupby(["player_id", "distance_bucket"]).agg(distance_attempts=("shot_made", "size"), distance_makes=("shot_made", "sum")).reset_index()
    zones = sorted(df["shot_zone_basic"].dropna().astype(str).unique().tolist())
    distances = ["0-3", "4-8", "9-16", "17-24", "25+"]
    rows: list[dict[str, Any]] = []
    for p in player_totals.itertuples(index=False):
        for zone in zones:
            z = zone_totals[(zone_totals["player_id"].eq(p.player_id)) & (zone_totals["shot_zone_basic"].eq(zone))]
            z_att = int(z["zone_attempts"].iloc[0]) if not z.empty else 0
            z_makes = int(z["zone_makes"].iloc[0]) if not z.empty else 0
            zone_prior = float(zone_mean.get(zone, global_fg))
            for dist in distances:
                d = dist_totals[(dist_totals["player_id"].eq(p.player_id)) & (dist_totals["distance_bucket"].eq(dist))]
                d_att = int(d["distance_attempts"].iloc[0]) if not d.empty else 0
                d_makes = int(d["distance_makes"].iloc[0]) if not d.empty else 0
                dist_prior = float(dist_mean.get(dist, global_fg))
                rows.append(
                    {
                        "player_id": str(p.player_id),
                        "player_name": p.player_name,
                        "team_id": str(p.team_id),
                        "team_name": p.team_name,
                        "shot_zone_basic": zone,
                        "distance_bucket": dist,
                        "global_fg": global_fg,
                        "zone_prior_mean": zone_prior,
                        "distance_prior_mean": dist_prior,
                        "player_prior_fga": int(p.fga),
                        "player_prior_fg_pct": pct(p.fgm, p.fga, global_fg),
                        "player_prior_fg_pct_shrunk": shrunk(p.fgm, p.fga, global_fg, 150),
                        "player_prior_2pa": int(p.two_pa),
                        "player_prior_2p_pct": pct(p.two_pm, p.two_pa, global_2p),
                        "player_prior_2p_pct_shrunk": shrunk(p.two_pm, p.two_pa, global_2p, 100),
                        "player_prior_3pa": int(p.three_pa),
                        "player_prior_3p_pct": pct(p.three_pm, p.three_pa, global_3p),
                        "player_prior_3p_pct_shrunk": shrunk(p.three_pm, p.three_pa, global_3p, 100),
                        "player_zone_prior_attempts": z_att,
                        "player_zone_prior_fg_pct": pct(z_makes, z_att, zone_prior),
                        "player_zone_prior_fg_pct_shrunk": shrunk(z_makes, z_att, zone_prior, 40),
                        "player_distance_prior_attempts": d_att,
                        "player_distance_prior_fg_pct": pct(d_makes, d_att, dist_prior),
                        "player_distance_prior_fg_pct_shrunk": shrunk(d_makes, d_att, dist_prior, 40),
                        "player_season_prior_fga": int(p.season_fga),
                        "player_season_prior_fg_pct": pct(p.season_fgm, p.season_fga, global_fg),
                        "player_season_prior_fg_pct_shrunk": shrunk(p.season_fgm, p.season_fga, global_fg, 100),
                        "player_previous_season_fga": int(p.prev_fga),
                        "player_previous_season_fg_pct": pct(p.prev_fgm, p.prev_fga, global_fg),
                        "player_previous_season_fg_pct_shrunk": shrunk(p.prev_fgm, p.prev_fga, global_fg, 200),
                        "player_previous_season_2pa": int(p.prev_2pa),
                        "player_previous_season_2p_pct_shrunk": shrunk(p.prev_2pm, p.prev_2pa, global_2p, 120),
                        "player_previous_season_3pa": int(p.prev_3pa),
                        "player_previous_season_3p_pct_shrunk": shrunk(p.prev_3pm, p.prev_3pa, global_3p, 120),
                        "player_cold_start": False,
                    }
                )
    cold = dict(rows[0])
    cold.update(
        {
            "player_id": "__COLD_START__",
            "player_name": "Unknown Player",
            "team_id": "",
            "team_name": "",
            "shot_zone_basic": zones[0],
            "distance_bucket": distances[0],
            "player_prior_fga": 0,
            "player_prior_fg_pct": global_fg,
            "player_prior_fg_pct_shrunk": global_fg,
            "player_prior_2pa": 0,
            "player_prior_2p_pct": global_2p,
            "player_prior_2p_pct_shrunk": global_2p,
            "player_prior_3pa": 0,
            "player_prior_3p_pct": global_3p,
            "player_prior_3p_pct_shrunk": global_3p,
            "player_zone_prior_attempts": 0,
            "player_zone_prior_fg_pct": global_fg,
            "player_zone_prior_fg_pct_shrunk": global_fg,
            "player_distance_prior_attempts": 0,
            "player_distance_prior_fg_pct": global_fg,
            "player_distance_prior_fg_pct_shrunk": global_fg,
            "player_season_prior_fga": 0,
            "player_season_prior_fg_pct": global_fg,
            "player_season_prior_fg_pct_shrunk": global_fg,
            "player_previous_season_fga": 0,
            "player_previous_season_fg_pct": global_fg,
            "player_previous_season_fg_pct_shrunk": global_fg,
            "player_previous_season_2pa": 0,
            "player_previous_season_2p_pct_shrunk": global_2p,
            "player_previous_season_3pa": 0,
            "player_previous_season_3p_pct_shrunk": global_3p,
            "player_cold_start": True,
        }
    )
    rows.append(cold)
    metadata = (
        latest[["player_id", "player_name", "team_id", "team_name"]]
        .drop_duplicates("player_id")
        .sort_values("player_name")
        .assign(player_id=lambda d: d["player_id"].astype(str), team_id=lambda d: d["team_id"].astype(str))
        .to_dict(orient="records")
    )
    priors = {"global_fg": global_fg, "global_2p": global_2p, "global_3p": global_3p, "zones": zones, "distances": distances}
    return pd.DataFrame(rows), metadata, priors


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(SHOT_DATA)
    featured = build_historical_features(df)
    featured["distance_bucket"] = featured["shot_distance"].astype(float).map(distance_bucket)
    featured = add_context_columns(featured)
    model = joblib.load(MODEL_PATH)
    policy = joblib.load(DECISION_PATH)
    columns = policy["columns"]
    validation = featured[(featured["season"].eq("2024-25")) & (featured["season_type"].eq("regular_season"))].sort_values(["game_date", "game_id", "event_id"])
    calibration_late = validation.iloc[len(validation) // 2 :].copy()
    calibrator = fit_calibration(calibration_late[TARGET], predict(model, calibration_late, columns))["sigmoid"]
    profiles, player_metadata, priors = build_profiles(featured)
    profiles.to_parquet(OUT_DIR / "player_profiles.parquet", index=False)
    (OUT_DIR / "player_metadata.json").write_text(json.dumps(player_metadata, indent=2), encoding="utf-8")
    metadata = {
        "model_name": "historical_shooter_profile",
        "model_version": "shot-quality-v1.1",
        "policy_version": "decision-v1",
        "training_seasons": PRIMARY_TRAIN_SEASONS,
        "validation_season": "2024-25",
        "held_out_test_season": "2025-26",
        "skill_features": SKILL_FEATURES,
        "generic_features": GENERIC_NUMERIC + GENERIC_BOOLEAN + GENERIC_CATEGORICAL,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "profile_cutoff": "all available canonical shots through 2025-26",
        "priors": priors,
    }
    joblib.dump(
        {
            "model": model,
            "calibrator": calibrator,
            "columns": columns,
            "decision_policy": {"replacement_xpps": float(policy["global_value"]), "threshold": float(policy["threshold"])},
            "metadata": metadata,
        },
        OUT_DIR / "production_inference_bundle.joblib",
    )
    print(
        json.dumps(
            {
                "profiles": len(profiles),
                "players": len(player_metadata),
                "bundle": str((OUT_DIR / "production_inference_bundle.joblib").relative_to(ROOT)),
                "replacement_xpps": float(policy["global_value"]),
                "threshold": float(policy["threshold"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
