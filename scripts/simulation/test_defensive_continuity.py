"""Continuity, training boundaries and defensive uncertainty invariants."""
import unittest

import numpy as np
import pandas as pd

import defensive_continuity as defense


class DefensiveContinuityTests(unittest.TestCase):
    def test_retained_context_scales_with_returning_minutes(self):
        minutes = np.full(10, 24.)
        none = defense.continuity(minutes, np.zeros(10, bool), -6.)
        half = defense.continuity(minutes, np.arange(10) < 5, -6.)
        all_players = defense.continuity(minutes, np.ones(10, bool), -6.)
        self.assertEqual(none["retained_defensive_context"], 0.)
        self.assertAlmostEqual(half["returning_minutes_share"], .5)
        self.assertAlmostEqual(half["retained_defensive_context"], -3.)
        self.assertAlmostEqual(all_players["retained_defensive_context"], -6.)
        self.assertEqual(half["returning_rotation_players"], 5)

    def test_synthetic_roster_never_inherits_team_context(self):
        for environment in [-100., 0., 100.]:
            result = defense.continuity(np.full(5, 48.), np.ones(5, bool), environment, synthetic=True)
            self.assertEqual(result["returning_minutes_share"], 0.)
            self.assertEqual(result["retained_defensive_context"], 0.)
            self.assertEqual(result["returning_top5_share"], 0.)

    def test_continuity_rejects_invalid_rotation(self):
        with self.assertRaises(ValueError):
            defense.continuity([30.] * 5, [True] * 5, 0.)

    def test_coefficients_cannot_fit_validation_or_final_test(self):
        with self.assertRaises(ValueError):
            defense.fit_linear(pd.DataFrame({"season_end_year": [2015]}), [])
        with self.assertRaises(ValueError):
            defense.require_development(pd.DataFrame({"season_end_year": [2020]}))
        with self.assertRaises(ValueError):
            defense.fit_tail(pd.DataFrame({"season_end_year": [2015]}))

    def test_both_bucket_definitions_remain_visible(self):
        frame = pd.DataFrame(dict(season_end_year=[2018] * 30, actual_drtg=np.arange(100., 130.)))
        ranked = defense.bucket_labels(frame, "ranking")
        legacy = defense.bucket_labels(frame, "legacy")
        self.assertTrue((ranked.value_counts() == 6).all())
        self.assertEqual(int((legacy == "poor").sum()), 12)
        self.assertEqual(int((ranked == "poor").sum()), 6)

    def test_deployable_equation_cannot_read_unmasked_source_context(self):
        rng = np.random.default_rng(4)
        frame = pd.DataFrame(rng.normal(size=(40, len(defense.CONTINUITY_COLUMNS))), columns=defense.CONTINUITY_COLUMNS)
        frame["season_end_year"] = 2014
        frame["anchor_drtg"] = 106.
        frame["actual_drtg"] = 106. - frame.w_defensive_box_plus_minus + frame.retained_defensive_context
        model = defense.fit_linear(frame, defense.CONTINUITY_COLUMNS)
        first = frame.iloc[:1].copy()
        first["retained_defensive_context"], first["returning_minutes_share"] = 0., 0.
        first["prior_defensive_environment"] = -100.
        second = first.copy()
        second["prior_defensive_environment"] = 100.
        np.testing.assert_allclose(defense.predict(first, model), defense.predict(second, model))
        self.assertNotIn("prior_defensive_environment", defense.CONTINUITY_COLUMNS)
        batch = pd.concat([first, second])
        self.assertAlmostEqual(defense.predict(first, model)[0], defense.predict(batch, model)[0])

    def test_uncertainty_has_documented_monotonic_warning_flags(self):
        fields = defense.CONTINUITY_COLUMNS + ["defensive_missing_share", "source_context_sd", "source_diversity"]
        reference = {field: dict(min=0., max=10., p01=0., p99=10., scale=5.) for field in fields}
        base = {field: 1. for field in fields}
        base.update(returning_minutes_share=.9, defensive_missing_share=0., source_context_sd=0., source_diversity=.1)
        thresholds = dict(context_sd_p75=1., context_sd_p90=2., diversity_p75=.5)
        self.assertEqual(defense.uncertainty(base, reference, thresholds)["category"], "LOW")
        for changes in [dict(returning_minutes_share=0.), dict(defensive_missing_share=.3),
                        dict(source_context_sd=3., source_diversity=.8), dict(w_age=20.)]:
            self.assertEqual(defense.uncertainty({**base, **changes}, reference, thresholds)["category"], "HIGH")
        self.assertEqual(defense.uncertainty({**base, "returning_minutes_share": .5}, reference, thresholds)["category"], "MODERATE")


if __name__ == "__main__":
    unittest.main()
