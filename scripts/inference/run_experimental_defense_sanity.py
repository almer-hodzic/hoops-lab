#!/usr/bin/env python3
"""Small reproducible sanity checks for experimental defensive context."""

from __future__ import annotations

import json

from experimental_defense_core import ExperimentalDefenseEngine
from inference_core import ProductionInferenceEngine


def run_context(prod: ProductionInferenceEngine, exp: ExperimentalDefenseEngine, player_id: str, base: dict, distances: list[float]) -> dict:
    production = prod.analyze(base)
    rows = []
    for distance in distances:
        experimental = exp.analyze(
            {
                **base,
                "playerId": player_id,
                "closestDefenderDistance": distance,
                "shotClock": 12,
                "dribbles": 0,
                "touchTime": 1.2,
                "defenderId": "203110",
                "season": "2025-26",
                "seasonType": "regular_season",
            }
        )
        rows.append({"defenderDistance": distance, "p_make": experimental["makeProbability"]})
    return {
        "production_p_make": production["prediction"]["makeProbability"],
        "production_recommendation": production["recommendation"],
        "experimental": rows,
    }


def main() -> int:
    prod = ProductionInferenceEngine()
    exp = ExperimentalDefenseEngine()
    jokic = run_context(
        prod,
        exp,
        "203999",
        {
            "playerId": "203999",
            "locX": 0,
            "locY": 32,
            "shotValue": 2,
            "actionType": "Hook Shot",
            "period": 4,
            "secondsRemaining": 240,
            "scoreDiff": -2,
            "isHome": True,
        },
        [1, 3, 6],
    )
    curry = run_context(
        prod,
        exp,
        "201939",
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
        },
        [1, 3, 6, 8],
    )
    defender_sensitivity = []
    for defender_id in ["203110", "203497", "202695", ""]:
        result = exp.analyze(
            {
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
                "defenderId": defender_id,
                "season": "2025-26",
                "seasonType": "regular_season",
            }
        )
        defender_sensitivity.append(
            {
                "defenderId": defender_id or None,
                "defenderName": result["defenderProfile"]["defenderName"],
                "p_make": result["makeProbability"],
                "coldStart": result["defenderProfile"]["coldStart"],
            }
        )
    print(json.dumps({"jokic": jokic, "curry": curry, "defender_sensitivity": defender_sensitivity}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
