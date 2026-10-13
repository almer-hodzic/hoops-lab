"""Isolated inference harness for experimental defensive shot quality v1."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "defense"))
import run_defensive_shot_quality_experiment_v1  # noqa: F401,E402 - required for joblib custom classes.

ARTIFACT = ROOT / "artifacts" / "models" / "experimental_defensive_shot_quality_v1.joblib"


def load_model(path: Path = ARTIFACT) -> dict[str, Any]:
    return joblib.load(path)


def predict(payload: dict[str, Any], path: Path = ARTIFACT) -> dict[str, Any]:
    bundle = load_model(path)
    spec = bundle["spec"]
    row = {col: payload.get(col) for col in spec.columns}
    frame = pd.DataFrame([row])
    p = float(bundle["model"].predict_proba(frame)[0, 1])
    return {
        "p_make": p,
        "model_version": bundle["version"],
        "selected_model_key": bundle["selected_model_key"],
        "supplied_feature_context": row,
        "cold_start_flags": {
            "shooter_cold_start": bool(payload.get("shooter_cold_start", False)),
            "defender_cold_start": bool(payload.get("defender_cold_start", False)),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True, help="Scenario JSON object.")
    args = parser.parse_args()
    print(json.dumps(predict(json.loads(args.json)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
