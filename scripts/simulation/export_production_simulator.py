"""Export approved fitted state only. No fitting, calibration, acquisition or evaluation."""
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import defensive_continuity as defense
import finalize_game_calibration as game
import improve_projection as team

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "data/simulation/production_v1"
VERSION = "hoopslab-single-game-v1.0.0"


def fingerprint(path):
    data = path.read_bytes()
    if path.suffix in {".json", ".csv", ".py", ".md"}:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def pipeline_state(candidate):
    model, columns = candidate
    if model is None:
        return dict(columns=columns, pipeline=None)
    imputer, scaler, estimator = model.steps
    return dict(columns=columns, pipeline=dict(
        impute=imputer[1].statistics_, center=scaler[1].mean_, scale=scaler[1].scale_,
        coefficients=estimator[1].coef_, intercept=estimator[1].intercept_))


def export():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    strength = joblib.load(team.BUNDLE)
    defensive_path = ROOT / "artifacts/simulation/defensive_continuity_v1.joblib"
    defensive = joblib.load(defensive_path)
    final = joblib.load(game.BUNDLE)
    protocol = json.loads(game.PROTOCOL.read_text())
    assert fingerprint(game.BUNDLE) == protocol["artifact_sha256"]
    assert final["sampler_choice"] == "D_bootstrap8"
    assert final["home_choice"] == "trailing5" and final["ot_choice"] == "exponential5"
    assert strength["rotation"] == "balanced_prior" and defensive["model"]["tail"] is None

    players = pd.read_csv(ROOT / "data/historical/processed/player_seasons.csv")
    source_path = ROOT / "data/analysis/simulation_defensive_source_fields.csv"
    source = pd.read_csv(source_path)
    fields = list(team.previous.DEFENSIVE_FIELDS.values())
    keep = ["player_id", "season_end_year", "team", "source_minutes", *fields]
    source[keep].to_csv(OUTPUT / "defensive_profiles.csv", index=False)
    profiles = team.previous.add_derived_player_features(players).merge(
        source, on=["player_id", "season_end_year", "team"], how="left", validate="one_to_one")
    profiles = profiles[profiles.source_minutes.notna() & ((profiles.source_minutes - profiles.minutes).abs() <= 1.)]
    defaults_fields = team.previous.PROFILE_FIELDS + ["team_pace", "league_pace"]
    train = profiles[(profiles.season_end_year <= 2013) & (profiles.minutes >= 100)]
    defaults = train[defaults_fields].median()
    roles = {role: rows[defaults_fields].median().fillna(defaults).to_dict()
             for role, rows in train.groupby("position")}

    sampler = final["engine_sampler"]
    overtime = final["engine_ot"]
    np.savetxt(OUTPUT / "joint_residuals.csv", sampler["samples"], delimiter=",", fmt="%.17g",
               header="pace,team_a_efficiency,team_b_efficiency", comments="")
    np.savetxt(OUTPUT / "overtime_residuals.csv", overtime["samples"], delimiter=",", fmt="%.17g",
               header="team_a,team_b", comments="")
    config = dict(modelVersion=VERSION, schemaVersion=1, textHashNormalization="CRLF_to_LF",
        researchCommit="a7491448f1f298bc39a8b475ce08db9fe85e6c85",
        defaults=dict(simulationCount=10000, minSimulations=1000, maxSimulations=100000,
                      minPlayers=8, maxPlayers=12, minutes=240),
        rotation=strength["rotation"], rookie_priors=strength["rookie_priors"],
        rotation_constants=dict(starter_floor=20, starter_cap=38, bench_floor=4, bench_cap=28,
                                prior_weight=.6, history_weight=.4, availability_floor=.6, availability_weight=.4),
        usage=strength["usage"], role_usage_means=strength["role_usage_means"],
        models={target: pipeline_state(strength["models"][target][strength["selected_models"][target]])
                for target in ["ortg", "pace"]},
        defense=pipeline_state(defensive["model"]["candidate"]),
        domain=strength["domain"], defensive_domain=defensive["domain"],
        uncertainty_thresholds=defensive["uncertainty_thresholds"],
        profile_defaults=defaults.to_dict(), role_defaults=roles,
        home=final["home"], sampler={k: v for k, v in sampler.items() if k != "samples"},
        overtime={k: v for k, v in overtime.items() if k != "samples"},
        overtime_max_periods=20, intervals=[.5, .8, .9, .95],
        sources={str(path.relative_to(ROOT)).replace("\\", "/"): fingerprint(path) for path in
                 [team.BUNDLE, defensive_path, game.BUNDLE, game.PROTOCOL, source_path]},
        kernel_hashes={f.__module__ + "." + f.__name__: hashlib.sha256(inspect.getsource(f).encode()).hexdigest()
                       for f in [team.projected_rotation, team.bounded_minutes, team.role_inputs,
                                 team.usage, team.aggregate_case, team.extrapolation, team.previous.aggregate,
                                 team.previous.roster_rows, team.previous.add_derived_player_features,
                                 team.previous.normalize_usage, team.previous.predict, defense.continuity,
                                 defense.predict, defense.uncertainty, game.means, game.base_means,
                                 game.regulation, game.draw_residuals, game.simulate]},
        canonical_hashes={name: fingerprint(ROOT / "data/historical/processed" / name) for name in
                          ["player_seasons.csv", "player_team_splits.csv", "team_seasons.csv", "league_seasons.csv"]},
        files={path.name: fingerprint(path) for path in sorted(OUTPUT.glob("*.csv"))})
    (OUTPUT / "config.json").write_text(json.dumps(clean(config), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(dict(modelVersion=VERSION, files={p.name: p.stat().st_size for p in OUTPUT.iterdir()})))


if __name__ == "__main__":
    export()
