"""Verify input boundaries, roster semantics and frozen-test controls."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import improve_projection as model


class ProjectionImprovementTests(unittest.TestCase):
    def setUp(self):
        self.prior = pd.DataFrame({field: np.ones(10) for field in model.previous.PROFILE_FIELDS})
        self.prior["prior_mpg"] = np.linspace(36, 8, 10)
        self.prior["history_mpg"] = self.prior.prior_mpg
        self.prior["prior_availability"] = .85
        self.prior["projected_age"] = 27.
        self.prior["prior_start_share"] = .5
        self.prior["prior_rank"] = np.arange(1, 11)
        self.prior["roster_depth"] = 10
        self.prior["known"], self.prior["returner"] = 1., 0.
        self.prior["usage_pct"] = np.linspace(35, 10, 10)
        self.priors = dict(unknown_role_mixture_mpg=8., roles={
            "starter": dict(equivalent_mpg=25.), "primary_bench": dict(equivalent_mpg=13.),
            "deep_bench": dict(equivalent_mpg=4.)})
        self.roles = dict(starter=24., primary_bench=18., deep_bench=12.)

    def test_exact_minutes_caps_and_depth(self):
        for size in range(8, 13):
            prior = pd.concat([self.prior, self.prior.iloc[:2]], ignore_index=True).iloc[:size].copy()
            minutes = model.projected_rotation(prior, "balanced_prior", self.priors, mode="full",
                                                slots=["PG", "SG", "SF", "PF", "C"] + ["bench"] * (size - 5))
            self.assertAlmostEqual(minutes.sum(), 240.)
            self.assertTrue((minutes[:5] >= 20.).all() and (minutes[:5] <= 38.).all())
            self.assertTrue((minutes[5:] >= 4.).all() and (minutes[5:] <= 28.).all())

    def test_reject_invalid_normal_rosters_and_require_explicit_ironman(self):
        for size in [5, 6, 7, 13]:
            prior = pd.concat([self.prior, self.prior.iloc[:3]]).iloc[:size]
            with self.assertRaises(ValueError):
                model.projected_rotation(prior, "balanced_prior", self.priors, mode="full")
        np.testing.assert_array_equal(model.projected_rotation(self.prior.iloc[:5], "balanced_prior", self.priors, mode="ironman"), [48.] * 5)
        with self.assertRaises(ValueError):
            model.projected_rotation(self.prior.iloc[:6], "balanced_prior", self.priors, mode="ironman")

    def test_unknown_slot_prior_changes_minutes_not_skill(self):
        prior = self.prior.copy()
        prior.loc[0, "known"] = 0.
        original = prior.copy(deep=True)
        minutes = model.projected_rotation(prior, "balanced_prior", self.priors, mode="full",
                                            slots=["PG", "SG", "SF", "PF", "C"] + ["bench"] * 5)
        self.assertGreaterEqual(minutes[0], 20.)
        pd.testing.assert_frame_equal(prior, original)

    def test_bounded_projection_is_deterministic_and_feasible(self):
        raw = np.array([1000.] + [-100.] * 7)
        result = model.bounded_minutes(raw, np.ones(8) * 4, np.ones(8) * 38)
        self.assertAlmostEqual(sum(result), 240.)
        self.assertTrue((result >= 4).all() and (result <= 38).all())
        with self.assertRaises(ValueError):
            model.bounded_minutes([30] * 5, [0] * 5, [40] * 5)

    def test_usage_conservation_for_every_candidate(self):
        minutes = np.full(10, 24.)
        for config in [dict(alpha=1., shrink=0.), dict(alpha=.75, shrink=0.), dict(alpha=1., shrink=.1)]:
            effective, shares = model.usage(self.prior, minutes, config, self.roles)
            self.assertAlmostEqual(sum(shares), 1.)
            self.assertAlmostEqual(np.dot(minutes / 48., effective), 100.)
            self.assertGreater(effective[0], effective[-1])

    def test_no_target_performance_in_profile_inputs(self):
        defaults = pd.Series({f: 1. for f in model.previous.PROFILE_FIELDS})
        defaults["team_pace"], defaults["league_pace"] = 98., 97.
        snapshot = dict(defaults, player_id="a", player_name="A", season_end_year=2013,
                        season="2012-13", minutes=2000., minutes_per_game=30., games=70.,
                        games_started=60., age=25., position="SG")
        future = dict(snapshot, season_end_year=2015, usage_pct=99.)
        data = dict(histories={"a": pd.DataFrame([snapshot, future])}, defaults=defaults, roles={},
                    schedules={2013: 82.}, environments={(2013, "a"): -1.})
        member = pd.DataFrame([dict(player_id="a", player_name="A", minutes=1., usage_pct=99., games_started=0)])
        first = model.profile_inputs(member, 2014, data)
        member[["minutes", "usage_pct", "games_started"]] = [99999., 1., 82.]
        second = model.profile_inputs(member, 2014, data)
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(first.iloc[0].season_end_year, 2013)

    def test_priors_and_model_fit_reject_future_outcomes(self):
        with self.assertRaises(ValueError):
            model.rookie_priors([dict(year=2020)])
        with self.assertRaises(ValueError):
            model.fit_team_models(pd.DataFrame({"season_end_year": [2014]}),
                                  pd.DataFrame({"season_end_year": [2020]}), pd.DataFrame())

    def test_extrapolation_is_domain_distance_not_quality(self):
        reference = {"x": dict(min=0., max=10., p01=1., p99=9., scale=5.)}
        self.assertEqual(model.extrapolation({"x": 5.}, reference)["risk"], "within_marginal_support")
        self.assertEqual(model.extrapolation({"x": 10.}, reference)["features"][0]["status"], "near_edge")
        result = model.extrapolation({"x": 20.}, reference)
        self.assertEqual(result["risk"], "high")
        self.assertEqual(result["maximum_exceed_iqr_units"], 2.)
        self.assertNotIn("quality", result)

    def test_final_guard_prevents_repeated_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = Path(directory) / "ledger.json"
            final = Path(directory) / "final.json"
            ledger.write_text(json.dumps(dict(evaluation_count=1)), encoding="utf-8")
            with patch.object(model, "LEDGER", ledger), patch.object(model, "FINAL", final):
                with self.assertRaisesRegex(RuntimeError, "already consumed"):
                    model.final_stage()
                with self.assertRaisesRegex(RuntimeError, "cannot be retuned"):
                    model.validation_stage()

    def test_final_guard_rejects_changed_frozen_inputs_before_loading_test(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = root / "protocol.json"
            protocol.write_text(json.dumps(dict(inputs={"source": "old"})), encoding="utf-8")
            with patch.object(model, "LEDGER", root / "ledger.json"), patch.object(model, "FINAL", root / "final.json"), \
                    patch.object(model, "PROTOCOL", protocol), patch.object(model, "input_fingerprints", return_value={"source": "changed"}), \
                    patch.object(model, "prepare") as prepare:
                with self.assertRaisesRegex(RuntimeError, "Frozen inputs/model/code/results changed"):
                    model.final_stage()
                prepare.assert_not_called()


if __name__ == "__main__":
    unittest.main()
