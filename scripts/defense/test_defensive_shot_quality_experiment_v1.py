"""Deterministic checks for Defensive Shot Quality Experiment v1."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "defense"))
sys.path.insert(0, str(ROOT / "scripts" / "inference"))
import run_defensive_shot_quality_experiment_v1  # noqa: F401,E402 - required for joblib custom classes.
import run_defensive_shot_quality_v11  # noqa: F401,E402 - required for V1.1 joblib custom classes.
from experimental_defense_core import ExperimentalDefenseEngine  # noqa: E402
from experimental_defense_v11_core import ExperimentalDefenseV11Engine  # noqa: E402
from audit_experimental_defense_support import support_flag, training_stats  # noqa: E402
from inference_core import InferenceError, ProductionInferenceEngine  # noqa: E402

PRED = ROOT / "data" / "analysis" / "defense" / "defensive_shot_quality_predictions.parquet"
METRICS = ROOT / "reports" / "modeling" / "DEFENSIVE_SHOT_QUALITY_METRICS.json"
ARTIFACT = ROOT / "artifacts" / "models" / "experimental_defensive_shot_quality_v1.joblib"
V11_ARTIFACT = ROOT / "artifacts" / "models" / "experimental_defensive_shot_quality_v11.joblib"
V11_POLICY = ROOT / "data" / "metadata" / "defensive_shot_quality_v11_feature_policy.json"
V11_METRICS = ROOT / "reports" / "modeling" / "DEFENSIVE_SHOT_QUALITY_V11_METRICS.json"


def fail(msg: str) -> None:
    raise AssertionError(msg)


def main() -> int:
    if not PRED.exists():
        fail("prediction parquet missing")
    if not METRICS.exists():
        fail("metrics json missing")
    if not ARTIFACT.exists():
        fail("experimental artifact missing")
    if not V11_ARTIFACT.exists():
        fail("V1.1 experimental artifact missing")
    if not V11_POLICY.exists():
        fail("V1.1 feature policy missing")
    if not V11_METRICS.exists():
        fail("V1.1 metrics missing")
    df = pd.read_parquet(PRED)
    summary = json.loads(METRICS.read_text(encoding="utf-8"))
    split_games = df.groupby("game_id")["split"].nunique()
    if int(split_games.max()) != 1:
        fail("a game crosses train/validation/test")
    dates = df.groupby("split")["game_date"].agg(["min", "max"])
    if not (dates.loc["train", "max"] <= dates.loc["validation", "min"] <= dates.loc["validation", "max"] <= dates.loc["test", "min"]):
        fail("chronological split order invalid")
    if int(df["touch_time_invalid"].sum()) != summary["data_audit"]["negative_touch_time_rows"]:
        fail("invalid touch time count mismatch")
    if df.loc[df["touch_time_invalid"], "touch_time"].notna().any():
        fail("negative touch time was not converted to missing")
    if int(df["shot_clock_missing"].sum()) != summary["data_audit"]["missing_shot_clock_rows"]:
        fail("shot-clock missingness not preserved")
    if (df["closest_defender_distance"] < 0).any():
        fail("negative closest defender distance")
    for col in ["shooter_prior_fga", "defender_prior_shots_defended"]:
        if (df[col] < 0).any():
            fail(f"negative prior column {col}")
    first_shooter = df.sort_values(["shooter_id", "game_date", "game_id"]).groupby("shooter_id").head(1)
    if (first_shooter["shooter_prior_fga"] != 0).any():
        fail("cold-start shooter prior is not zero attempts")
    first_defender = df.sort_values(["closest_defender_id", "game_date", "game_id"]).groupby("closest_defender_id").head(1)
    if (first_defender["defender_prior_shots_defended"] != 0).any():
        fail("cold-start defender prior is not zero attempts")
    bundle = joblib.load(ARTIFACT)
    sample = df[df["split"].eq("test")].head(25)
    p1 = bundle["model"].predict_proba(sample)[:, 1]
    pcol = f"p_{bundle['selected_model_key']}"
    if not np.allclose(p1, sample[pcol].to_numpy(), atol=1e-12):
        fail("serialized model inference parity failed")
    engine = ExperimentalDefenseEngine()
    prod = ProductionInferenceEngine()
    contract = engine.feature_contract()
    if contract["selectedModelKey"] != "F_hgb":
        fail("wrong experimental model selected")
    expected = set(bundle["spec"].columns)
    if set(contract["columns"]) != expected:
        fail("experimental contract columns changed")
    payload = {
        "playerId": "201939",
        "locX": 80,
        "locY": 250,
        "shotValue": 3,
        "period": 4,
        "secondsRemaining": 240,
        "isHome": True,
        "closestDefenderDistance": 3,
        "shotClock": 12,
        "dribbles": 0,
        "touchTime": 1.2,
        "defenderId": "203110",
        "season": "2025-26",
        "seasonType": "regular_season",
    }
    exp = engine.analyze(payload)
    if not 0 <= exp["makeProbability"] <= 1:
        fail("experimental probability outside [0,1]")
    if exp["status"] != "EXPERIMENTAL":
        fail("experimental response not clearly marked")
    if "recommendation" in exp:
        fail("experimental response must not return a production recommendation")
    if exp["shooterProfile"]["source"] not in {"exact_player_distance_bucket", "player_profile_with_population_distance_fallback"}:
        fail("modern shooter adapter did not map Stephen Curry")
    if exp["defenderProfile"]["defenderName"] != "Draymond Green":
        fail("modern defender adapter did not resolve selected defender")
    if exp["defenderProfile"]["lt6DfgPct"] is None or exp["defenderProfile"]["threePtDfgPct"] is None:
        fail("display defender category DFG fields were not resolved")
    no_defender = engine.analyze({**payload, "defenderId": ""})
    if not no_defender["defenderProfile"]["coldStart"]:
        fail("missing defender fallback did not activate")
    for bad_key, bad_value in [
        ("closestDefenderDistance", -0.1),
        ("shotClock", 25),
        ("dribbles", -1),
        ("touchTime", -0.1),
    ]:
        try:
            engine.analyze({**payload, bad_key: bad_value})
            fail(f"invalid {bad_key} was accepted")
        except InferenceError:
            pass
    before = prod.analyze(
        {
            "playerId": "201939",
            "locX": 80,
            "locY": 250,
            "shotValue": 3,
            "actionType": "Jump Shot",
            "period": 4,
            "secondsRemaining": 240,
            "scoreDiff": -3,
            "isHome": True,
        }
    )["recommendation"]
    after = prod.analyze(
        {
            "playerId": "201939",
            "locX": 80,
            "locY": 250,
            "shotValue": 3,
            "actionType": "Jump Shot",
            "period": 4,
            "secondsRemaining": 240,
            "scoreDiff": -3,
            "isHome": True,
            "closestDefenderDistance": 1,
            "shotClock": 1,
            "dribbles": 20,
            "touchTime": 12,
        }
    )["recommendation"]
    if before != after:
        fail("experimental fields changed production recommendation")
    app_js = (ROOT / "nba-clutch-agent" / "AiAgents.NbaClutchAgent.Web" / "wwwroot" / "app.js").read_text(encoding="utf-8")
    if "closestDefenderDistance * court.unitsPerFoot" not in app_js:
        fail("court defensive radius does not use canonical 10 units per foot scale")
    support_sample = pd.DataFrame({"x": [0, 1, 2, 3, 4], "b": [False, False, True, True, True]})
    support_stats = training_stats(support_sample, ["x", "b"])
    if support_flag(2, support_stats["x"]) != "IN_SUPPORT":
        fail("support diagnostic failed in-range numeric")
    if support_flag(10, support_stats["x"]) != "OUTSIDE_MIN_MAX":
        fail("support diagnostic failed outside min/max numeric")
    if support_flag(2, support_stats["x"], semantic_mismatch=True) != "SEMANTIC_MISMATCH":
        fail("support diagnostic failed semantic mismatch flag")
    if support_stats["b"]["median"] != 1.0:
        fail("support diagnostic failed boolean-to-numeric handling")
    split = df.groupby("split").agg(rows=("made", "size"), games=("game_id", "nunique")).to_dict(orient="index")
    expected_split = {"train": {"rows": 90922, "games": 632}, "validation": {"rows": 19219, "games": 136}, "test": {"rows": 17928, "games": 136}}
    if {k: {"rows": int(v["rows"]), "games": int(v["games"])} for k, v in split.items()} != expected_split:
        fail("V1.1 split changed")
    v11_bundle = joblib.load(V11_ARTIFACT)
    v11_policy = json.loads(V11_POLICY.read_text(encoding="utf-8"))
    v11_features = set(v11_bundle["spec"].columns)
    if any(col in v11_features for col in ["shooter_prior_fga", "shooter_prior_2pa", "shooter_prior_3pa", "shooter_distance_prior_attempts"]):
        fail("V1.1 selected feature set includes incompatible attempt counts")
    if any(col.startswith("defender_pt_") for col in v11_features):
        fail("V1.1 selected feature set unexpectedly includes defender profile")
    if not v11_policy["excluded_attempt_counts"] or v11_policy["included_defender_profile"]:
        fail("V1.1 feature policy does not match selected deployable model")
    sample_v11 = df[df["split"].eq("test")].head(25)
    parity_a = v11_bundle["model"].predict_proba(sample_v11)[:, 1]
    parity_b = v11_bundle["model"].predict_proba(sample_v11[v11_bundle["spec"].columns])[:, 1]
    if not np.allclose(parity_a, parity_b, atol=1e-12):
        fail("V1.1 serialized model parity failed")
    v11_engine = ExperimentalDefenseV11Engine()
    contract = v11_engine.feature_contract()
    if contract["modelVersion"] != "defensive_shot_quality_v11" or contract["status"] != "EXPERIMENTAL":
        fail("V1.1 runtime contract does not expose the promoted experimental version")
    if contract["usesAttemptCounts"] or contract["usesDefenderProfile"]:
        fail("V1.1 inference contract is not deployable-clean")
    if any(col in contract["columns"] for col in ["shooter_prior_fga", "shooter_prior_2pa", "shooter_prior_3pa", "shooter_distance_prior_attempts"]):
        fail("V1.1 contract exposes shooter count features")
    if any(col.startswith("defender_pt_") for col in contract["columns"]):
        fail("V1.1 contract exposes defender profile features")
    if contract["defenderProfile"]["usage"] != "context-only":
        fail("V1.1 contract does not mark defender profile as context-only")
    jokic_v11 = v11_engine.analyze(
        {
            "playerId": "203999",
            "shotDistance": 3,
            "shotValue": 2,
            "closestDefenderDistance": 6,
            "shotClock": 12,
            "dribbles": 0,
            "touchTime": 1.2,
        }
    )
    curry_v11 = v11_engine.analyze(
        {
            "playerId": "201939",
            "shotDistance": 26,
            "shotValue": 3,
            "closestDefenderDistance": 3,
            "shotClock": 12,
            "dribbles": 0,
            "touchTime": 1.2,
        }
    )
    if jokic_v11["supportStatus"] != "OUT_OF_DISTRIBUTION":
        fail("V1.1 support diagnostic failed Jokic OOD scenario")
    if jokic_v11["makeProbability"] is not None or jokic_v11["displayPolicy"] != "HIDE_ABSOLUTE_PROBABILITY":
        fail("V1.1 OOD scenario did not hide the display probability")
    if curry_v11["supportStatus"] != "IN_DISTRIBUTION":
        fail("V1.1 support diagnostic failed Curry in-distribution scenario")
    if curry_v11["makeProbability"] is None:
        fail("V1.1 in-distribution scenario did not return a displayable probability")
    expected_curry = {
        1: 0.3023963265603195,
        3: 0.3122140864429996,
        6: 0.3555128234709174,
        8: 0.3697122738616877,
    }
    for defender_distance, expected_probability in expected_curry.items():
        result = v11_engine.analyze({**curry_v11["context"], "playerId": "201939", "closestDefenderDistance": defender_distance, "secondsRemaining": 240})
        if result["supportStatus"] != "IN_DISTRIBUTION":
            fail(f"Curry {defender_distance} ft scenario lost in-distribution support")
        if not np.isclose(result["makeProbability"], expected_probability, atol=1e-12):
            fail(f"Curry {defender_distance} ft serialized V1.1 parity failed")
    median_rows = run_defensive_shot_quality_v11.representative_rows(
        df[df["split"].eq("train")],
        v11_bundle["spec"],
    )
    median_near_rim = median_rows["median_near_rim_2pt"]
    v11_stats = run_defensive_shot_quality_v11.training_stats(df[df["split"].eq("train")], v11_bundle["spec"].columns)
    for defender_distance, expected_probability in [(5, 0.7763415575005979), (6, 0.8398947466070247)]:
        row = dict(median_near_rim)
        row["closest_defender_distance"] = defender_distance
        p = float(v11_bundle["model"].predict_proba(pd.DataFrame([row]))[0, 1])
        diag = run_defensive_shot_quality_v11.support_diagnostic(row, v11_stats, sparse_context=True)
        if diag["supportStatus"] != "LIMITED_SUPPORT" or not diag["sparseContext"]:
            fail("median near-rim limited-support diagnostic failed")
        if not np.isclose(p, expected_probability, atol=1e-12):
            fail("median near-rim limited-support probability changed")
    base_v11_payload = {
        "playerId": "201939",
        "shotDistance": 26,
        "shotValue": 3,
        "closestDefenderDistance": 3,
        "shotClock": 12,
        "dribbles": 0,
        "touchTime": 1.2,
        "season": "2025-26",
        "seasonType": "regular_season",
    }
    defender_probabilities = [
        v11_engine.analyze({**base_v11_payload, "defenderId": defender_id})["makeProbability"]
        for defender_id in ["", "203110", "203497", "202695"]
    ]
    if not all(np.isclose(defender_probabilities[0], p, atol=0.0, rtol=0.0) for p in defender_probabilities):
        fail("defender selection changed V1.1 probability")
    if "featureVector" in curry_v11:
        fail("V1.1 response exposes internal feature vector in normal API payload")
    service_py = (ROOT / "scripts" / "inference" / "inference_service.py").read_text(encoding="utf-8")
    if "ExperimentalDefenseV11Engine" not in service_py or "Handler.defense_engine = ExperimentalDefenseV11Engine()" not in service_py:
        fail("experimental HTTP runtime is not promoted to V1.1")
    if "Handler.defense_engine = ExperimentalDefenseEngine()" in service_py:
        fail("experimental HTTP runtime still loads V1 artifact")
    if "Outside training support" not in app_js or "absolute experimental probability is not shown" not in app_js:
        fail("UI does not hide/explain OOD experimental probability")
    if "Experimental context analysis unavailable." not in app_js:
        fail("UI failure isolation message missing")
    if "Context only" not in app_js or "not currently used by the experimental probability model" not in app_js:
        fail("UI does not separate defender profile from model inputs")
    print("defensive shot quality experiment checks passed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1)
