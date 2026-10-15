"""Production/reference parity, canonical identity and all API contract rules."""
import copy
import json
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np

import finalize_game_calibration as research
from production_simulator import ARTIFACTS, ProductionSimulator, SimulationError, text_fingerprint
from simulation_service import SimulationServer


class ProductionSimulatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = ProductionSimulator()
        cls.references = json.loads((ARTIFACTS / "reference_cases.json").read_text())

    def request(self):
        return copy.deepcopy(self.references["cases"][0]["request"])

    def reject(self, req):
        with self.assertRaises(SimulationError):
            self.engine.simulate(req)

    def test_frozen_config_reference_link(self):
        self.assertEqual(text_fingerprint(ARTIFACTS / "config.json"), self.references["configSha256"])

    def test_checkout_line_endings_do_not_break_integrity(self):
        with tempfile.TemporaryDirectory() as folder:
            from pathlib import Path
            directory = Path(folder)
            for source in ARTIFACTS.iterdir():
                (directory / source.name).write_bytes(source.read_bytes().replace(b"\r\n", b"\n"))
            engine = ProductionSimulator(artifact_dir=directory)
            self.assertEqual(engine.simulate(self.request()), self.engine.simulate(self.request()))
            residual = directory / "joint_residuals.csv"
            residual.write_bytes(residual.read_bytes()+b"0,0,0\n")
            with self.assertRaisesRegex(RuntimeError, "checksum mismatch"):
                ProductionSimulator(artifact_dir=directory)

    def test_six_original_research_references(self):
        for case in self.references["cases"]:
            with self.subTest(case=case["case"]):
                req = case["request"]
                result = self.engine.simulate(req)
                for key in ["teamA", "teamB"]:
                    actual, features = self.engine.project(self.engine.validate_team(req[key], req["ironmanMode"], key), req["targetSeason"], req["ironmanMode"])
                    expected = case[key]
                    for field, target in [("projectedORtg", "ortg"), ("projectedDRtg", "drtg"), ("projectedPace", "pace")]:
                        self.assertAlmostEqual(actual[field], expected["strengths"][target], delta=1e-9)
                    np.testing.assert_allclose([r["minutes"] for r in actual["rotation"]], expected["minutes"], atol=1e-9, rtol=0)
                    for field, value in expected["features"].items():
                        self.assertAlmostEqual(features[field], value, delta=1e-9, msg=field)
                frame, home, home_a = self.engine.game_inputs(result["teamA"], result["teamB"], req["targetSeason"], req["venue"])
                for field, value in research.means(frame, home).items():
                    self.assertAlmostEqual(float(value[0]), case["expected"][field], delta=1e-9)
                # Same NumPy RNG and research sampling path: exact empirical parity.
                simulation = case["simulation"]
                h, a = ("teamA", "teamB") if home_a else ("teamB", "teamA")
                self.assertAlmostEqual(result[h]["winProbability"], simulation["probability"], delta=1e-15)
                self.assertEqual(result["game"]["simulatedMeanScore"][h], simulation["expected_home"])
                self.assertEqual(result["game"]["simulatedMeanScore"][a], simulation["expected_away"])
                self.assertEqual(result["game"]["overtimeProbability"], simulation["simulated_ot"])
                for level in [50, 80, 90, 95]:
                    interval = result["game"]["marginIntervals"][str(level)]
                    expected = [simulation[f"lo{level}"], simulation[f"hi{level}"]]
                    if not home_a:
                        expected = [-expected[1], -expected[0]]
                    np.testing.assert_array_equal(interval, expected)

    def test_valid_eight_and_twelve_player_rosters(self):
        result = self.engine.simulate(self.request())
        self.assertEqual(len(result["teamA"]["rotation"]), 12)
        self.assertEqual(len(result["teamB"]["rotation"]), 8)
        for key in ["teamA", "teamB"]:
            self.assertAlmostEqual(result[key]["projectedMinutes"], 240.)
            rotation = result[key]["rotation"]
            self.assertAlmostEqual(sum(r["possessionShare"] for r in rotation), 1.)
            self.assertAlmostEqual(sum(r["minutes"]/48*r["normalizedUsagePct"] for r in rotation), 100.)

    def test_duplicate_canonical_player_even_different_season(self):
        req = self.request()
        req["teamA"]["bench"][0] = dict(playerId="br:curryst01", season="2014-15")
        self.reject(req)

    def test_unknown_id(self):
        req = self.request(); req["teamA"]["starters"][0]["playerId"] = "br:no-such-player"
        self.reject(req)

    def test_unavailable_season(self):
        req = self.request(); req["teamA"]["starters"][0]["season"] = "1979-80"
        self.reject(req)

    def test_bad_target_era(self):
        for season in ["1978-79", "2026-27", "neutral", None, []]:
            req = self.request(); req["targetSeason"] = season
            self.reject(req)

    def test_short_normal_rosters_rejected(self):
        for size in [5, 6, 7]:
            req = self.request(); req["teamA"]["bench"] = req["teamA"]["bench"][:size-5]
            self.reject(req)

    def test_five_only_explicit_ironman(self):
        req = copy.deepcopy(self.references["cases"][5]["request"])
        result = self.engine.simulate(req)
        for key in ["teamA", "teamB"]:
            self.assertEqual(result[key]["uncertainty"], "HIGH")
            self.assertEqual([r["minutes"] for r in result[key]["rotation"]], [48.]*5)
        req["ironmanMode"] = False
        self.reject(req)
        req = self.request(); req["ironmanMode"] = True
        self.reject(req)

    def test_over_twelve(self):
        req = self.request(); req["teamA"]["bench"].append(dict(playerId="br:looneke01", season="2015-16"))
        self.reject(req)

    def test_bad_starter_slots(self):
        for slot in ["PG", "X", None, []]:
            req = self.request(); req["teamA"]["starters"][1]["slot"] = slot
            self.reject(req)

    def test_missing_starters(self):
        req = self.request(); req["teamA"]["starters"].pop()
        self.reject(req)

    def test_bad_bench_slot(self):
        req = self.request(); req["teamA"]["bench"][0]["slot"] = "PG"
        self.reject(req)

    def test_team_builder_identity_structure(self):
        req = self.request()
        team = req["teamA"]
        team["teamName"] = team.pop("name"); team["schemaVersion"] = 1
        team["starters"] = {r["slot"]: r for r in team["starters"]}
        for i, row in enumerate(team["bench"]):
            row["slot"] = f"BENCH{i+1}"
        self.assertEqual(self.engine.simulate(req)["teamA"]["name"], "2015-16 Warriors")

    def test_caller_statistics_rejected_and_labels_never_override(self):
        req = self.request()
        expected = self.engine.simulate(req)
        req["teamA"]["starters"][0]["playerName"] = "Fabricated Label"
        self.assertEqual(self.engine.simulate(req), expected)
        req["teamA"]["starters"][0]["usage_pct"] = 100
        self.reject(req)

    def test_bad_requests_and_types(self):
        for req in [None, [], "hi", {}, dict(self.request(), unknown=True)]:
            self.reject(req)
        for value in [1, "true", None]:
            req = self.request(); req["ironmanMode"] = value
            self.reject(req)
        req = self.request(); req["teamA"]["starters"][0] = None
        self.reject(req)

    def test_same_seed_identical(self):
        req = self.request()
        self.assertEqual(self.engine.simulate(req), self.engine.simulate(req))

    def test_different_seeds(self):
        req = self.request(); first = self.engine.simulate(req)
        req["seed"] += 1
        self.assertNotEqual(first["game"], self.engine.simulate(req)["game"])

    def test_no_seed_returns_replayable_random_seed(self):
        req = self.request(); req.pop("seed")
        result = self.engine.simulate(req)
        req["seed"] = result["seed"]
        self.assertEqual(result, self.engine.simulate(req))

    def test_invalid_seeds(self):
        for seed in [-1, True, 1.5, "1", 2**53]:
            req = self.request(); req["seed"] = seed
            self.reject(req)

    def test_draw_count_bounds(self):
        for count in [999, 100001, True, "10000", 1000.5, None]:
            req = self.request(); req["simulationCount"] = count
            self.reject(req)
        for count in [1000, 100000]:
            req = self.request(); req["simulationCount"] = count
            self.assertEqual(self.engine.simulate(req)["simulationCount"], count)

    def test_venue_expectations(self):
        req = self.request()
        expected = {}
        for venue in ["neutral", "teamA_home", "teamB_home"]:
            req["venue"] = venue; expected[venue] = self.engine.simulate(req)["game"]
        neutral = expected["neutral"]["expectedMargin"]
        effect = self.engine.config["home"]["home_margin_per100"]*expected["neutral"]["expectedPace"]/100
        self.assertAlmostEqual(expected["teamA_home"]["expectedMargin"]-neutral, effect)
        self.assertAlmostEqual(expected["teamB_home"]["expectedMargin"]-neutral, -effect)
        req["venue"] = "arena"
        self.reject(req)

    def test_all_supported_target_eras(self):
        req = self.request(); req["simulationCount"] = 1000
        for season in self.engine.info()["supportedTargetSeasons"]:
            req["targetSeason"] = season
            result = self.engine.simulate(req)
            self.assertEqual(result["targetSeason"], season)
            self.assertTrue(np.isfinite(result["game"]["expectedTotal"]))

    def test_every_final_score_untied_probabilities_and_intervals(self):
        req = self.request(); result = self.engine.simulate(req)
        a, b = result["teamA"], result["teamB"]
        self.assertTrue(0 <= a["winProbability"] <= 1)
        self.assertAlmostEqual(a["winProbability"]+b["winProbability"], 1.)
        distribution = result["game"]["scoreDistributions"]
        self.assertFalse(any(row["value"] == 0 for row in distribution["margin"]))
        for key in distribution:
            self.assertEqual(sum(row["count"] for row in distribution[key]), req["simulationCount"])
        for key in ["teamA", "teamB"]:
            previous = [float("inf"), float("-inf")]
            for level in ["50", "80", "90", "95"]:
                lo, hi = result["game"]["scoreIntervals"][level][key]
                self.assertLessEqual(lo, hi)
                self.assertLessEqual(lo, previous[0]); self.assertGreaterEqual(hi, previous[1])
                previous = [lo, hi]

    def test_high_uncertainty_does_not_change_strength_or_sampling(self):
        req = self.request(); baseline = self.engine.simulate(req)
        fake = dict(category="LOW", high_flags={}, coverage={}, continuity=1.)
        with patch("production_simulator.defense.uncertainty", return_value=fake):
            alternative = self.engine.simulate(req)
        self.assertEqual(baseline["game"], alternative["game"])
        for key in ["projectedORtg", "projectedDRtg", "projectedPace"]:
            self.assertEqual(baseline["teamA"][key], alternative["teamA"][key])

    def test_no_inherited_franchise_defense(self):
        result = self.engine.simulate(self.request())
        for key in ["teamA", "teamB"]:
            self.assertEqual(result[key]["diagnostics"]["returning_minutes_share"], 0.)
            self.assertEqual(result[key]["diagnostics"]["retained_defensive_context"], 0.)

    def test_traded_season_uses_weighted_source_context(self):
        req = self.request()
        req["teamB"]["bench"][-1] = dict(playerId="br:fryech01", season="2015-16")
        result = self.engine.simulate(req)
        self.assertEqual(result["teamB"]["rotation"][-1]["playerId"], "br:fryech01")
        self.assertTrue(np.isfinite(self.engine.environments[(2016, "br:fryech01")]))
        self.assertNotIn("nan", self.engine.origins[(2016, "br:fryech01")])

    def test_high_extrapolation_case(self):
        req = copy.deepcopy(self.references["cases"][1]["request"])
        result = self.engine.simulate(req)
        self.assertEqual(result["teamA"]["uncertainty"], "HIGH")
        self.assertTrue(result["teamA"]["uncertaintyReasons"])

    def test_runtime_never_loads_research_artifacts_or_fits(self):
        with patch("joblib.load", side_effect=AssertionError("Research artifact loaded")), \
                patch("sklearn.linear_model.LinearRegression.fit", side_effect=AssertionError("Fit called")), \
                patch("sklearn.linear_model.Ridge.fit", side_effect=AssertionError("Fit called")):
            engine = ProductionSimulator()
            engine.simulate(self.request())

    def test_concurrent_seeded_requests_share_no_rng_state(self):
        req = self.request(); expected = self.engine.simulate(req)
        with ThreadPoolExecutor(max_workers=4) as pool:
            for result in pool.map(self.engine.simulate, [req]*8):
                self.assertEqual(result, expected)

    def test_data_not_reparsed_per_request(self):
        with patch("pandas.read_csv", side_effect=AssertionError("Per-request read")):
            self.engine.simulate(self.request())

    def test_internal_http_validation_and_metadata(self):
        server = SimulationServer(("127.0.0.1", 0), self.engine)
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        try:
            with urlopen(base+"/api/simulation/info") as response:
                self.assertEqual(json.load(response)["modelVersion"], self.engine.config["modelVersion"])
            for body, status in [(json.dumps(self.request()).encode(), 200), (b"{}", 400), (b"[1]", 400),
                                 (b"NaN", 400), (b"x", 400), (b"x"*65537, 413)]:
                request = Request(base+"/api/simulation/game", data=body, headers={"Content-Type": "application/json"})
                try:
                    with urlopen(request, timeout=30) as response:
                        self.assertEqual(response.status, status)
                        self.assertEqual(json.load(response)["game"]["finalTies"], 0)
                except HTTPError as exc:
                    self.assertEqual(exc.code, status)
        finally:
            server.shutdown(); server.server_close(); thread.join()


if __name__ == "__main__":
    unittest.main()
