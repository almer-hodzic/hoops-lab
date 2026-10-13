#!/usr/bin/env python3
"""Deterministic tests for production inference."""

from __future__ import annotations

import unittest

import numpy as np

from inference_core import ProductionInferenceEngine


class ProductionInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.engine = ProductionInferenceEngine()
        players = cls.engine.players("Stephen", 5)["items"]
        cls.player_id = str(players[0]["player_id"]) if players else "201939"

    def request(self, **overrides):
        base = {
            "playerId": self.player_id,
            "shotDistance": 26.0,
            "locX": 80.0,
            "locY": 250.0,
            "shotValue": 3,
            "shotType": "3PT Field Goal",
            "actionType": "Jump Shot",
            "shotZoneBasic": "Above the Break 3",
            "shotZoneArea": "Center(C)",
            "shotZoneRange": "24+ ft.",
            "period": 4,
            "secondsRemaining": 240,
            "scoreDiff": -3,
            "isHome": True,
            "isOvertime": False,
            "gameState": "trailing",
        }
        base.update(overrides)
        return base

    def test_take_review_defer_cases(self):
        take = self.engine.analyze(self.request(locX=0.0, locY=10.0, shotValue=2, actionType="Dunk Shot"))
        review = self.engine.analyze(self.request(locX=-230.0, locY=10.0, shotValue=3, actionType="Jump Shot"))
        defer = self.engine.analyze(self.request(locX=0.0, locY=50.0, shotValue=2, actionType="Layup Shot"))
        self.assertEqual(take["recommendation"], "TAKE")
        self.assertEqual(review["recommendation"], "REVIEW")
        self.assertEqual(defer["recommendation"], "DEFER")

    def test_probability_and_math(self):
        result = self.engine.analyze(self.request())
        p = result["prediction"]["makeProbability"]
        self.assertGreaterEqual(p, 0)
        self.assertLessEqual(p, 1)
        self.assertAlmostEqual(result["prediction"]["shotXPps"], p * result["shot"]["value"], places=12)
        self.assertAlmostEqual(
            result["benchmark"]["margin"],
            result["prediction"]["shotXPps"] - result["benchmark"]["replacementXPps"],
            places=12,
        )

    def test_threshold_boundary_helper(self):
        threshold = self.engine.artifacts.threshold
        self.assertEqual(self.engine.analyze(self.request())["benchmark"]["threshold"], threshold)

    def test_unknown_player_cold_start(self):
        result = self.engine.analyze(self.request(playerId="999999999"))
        self.assertTrue(result["profile"]["coldStart"])
        self.assertIn("COLD_START_PROFILE", result["reasonCodes"])

    def test_invalid_request(self):
        with self.assertRaises(Exception):
            self.engine.analyze(self.request(shotValue=4))

    def test_canonical_feature_construction(self):
        result = self.engine.analyze(self.request(locX=0, locY=32, shotValue=2, actionType="Layup Shot", shotType="Layup Shot"))
        self.assertEqual(result["resolvedContext"]["shotType"], "2PT Field Goal")
        self.assertEqual(result["resolvedContext"]["modelShotDistance"], 3)
        self.assertAlmostEqual(result["resolvedContext"]["displayDistanceFeet"], 3.2, places=12)
        self.assertEqual(result["resolvedContext"]["shotZoneBasic"], "Restricted Area")
        self.assertEqual(result["resolvedContext"]["shotZoneArea"], "Center(C)")
        self.assertEqual(result["resolvedContext"]["shotZoneRange"], "Less Than 8 ft.")

    def test_three_point_canonical_shot_type(self):
        result = self.engine.analyze(self.request(locX=80, locY=250, shotValue=3, actionType="Jump Shot", shotType="Jump Shot"))
        self.assertEqual(result["resolvedContext"]["shotType"], "3PT Field Goal")

    def test_action_type_passed_unchanged_when_valid(self):
        result = self.engine.analyze(self.request(locX=0, locY=32, shotValue=2, actionType="Driving Hook Shot"))
        self.assertEqual(result["resolvedContext"]["actionType"], "Driving Hook Shot")

    def test_invalid_obvious_action_location_rejected(self):
        with self.assertRaises(Exception):
            self.engine.analyze(self.request(locX=80, locY=250, shotValue=3, actionType="Dunk Shot"))

    def test_no_unknown_shot_type_reaches_model(self):
        frame, _, _ = self.engine._frame_from_payload(self.request(locX=0, locY=32, shotValue=2, shotType="Layup Shot", actionType="Layup Shot"))
        self.assertEqual(frame["shot_type"].iloc[0], "2PT Field Goal")

    def test_model_parity_direct_python(self):
        payload = self.request()
        frame, _, _ = self.engine._frame_from_payload(payload)
        raw = np.clip(self.engine.artifacts.model.predict_proba(frame[self.engine.artifacts.columns])[:, 1], 1e-6, 1 - 1e-6)
        expected = float(self.engine._apply_calibration(raw)[0])
        actual = self.engine.analyze(payload)["prediction"]["makeProbability"]
        self.assertAlmostEqual(actual, expected, places=12)


if __name__ == "__main__":
    unittest.main()
