"""Domain invariants for lagged roster construction and prediction deployment."""
import unittest

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from validate_deployability import (
    DEPLOYABLE_FEATURE_SET, PROFILE_FIELDS, aggregate, build_lagged, calibrate_usage,
    normalize_usage, pipeline, predict, prior_profiles, rotation,
)
from export_cached_defensive_profiles import export_cached_profiles


class DeployabilityTests(unittest.TestCase):
    def setUp(self):
        self.defaults = pd.Series({field: 1. for field in PROFILE_FIELDS})
        self.defaults["usage_pct"] = 20.
        self.defaults["team_pace"] = self.defaults["league_pace"] = 100.
        self.members = pd.DataFrame({"player_id": [str(i) for i in range(10)],
                                     "player_name": [str(i) for i in range(10)],
                                     "minutes": np.arange(10) * 100 + 100.,
                                     "season_end_year": 2016, "team_id": "A"})
        self.histories = {}
        for i in range(10):
            row = dict(self.defaults, player_id=str(i), season_end_year=2015,
                       season="2014-15", minutes=1000., minutes_per_game=32. - i,
                       position="SG")
            self.histories[str(i)] = pd.DataFrame([row, {**row, "season_end_year": 2016, "usage_pct": 99.}])
        self.context = pd.Series({"league_points_per_75": 81., "league_pace": 98.})

    def test_rotation_capacity(self):
        for size in [5, 6, 7, 8, 10, 15, 25]:
            minutes = rotation(np.linspace(36, 2, size))
            self.assertAlmostEqual(sum(minutes), 240.)
            self.assertTrue((minutes >= 0).all() and (minutes <= 48).all())
            if size >= 8:
                self.assertTrue((minutes[:5] >= 24).all() and (minutes[:5] <= 40).all())
                self.assertTrue((minutes[5:10] <= 26).all())
        np.testing.assert_array_equal(rotation([30] * 5), [48] * 5)
        with self.assertRaises(ValueError):
            rotation([30] * 4)

    def test_usage_conservation_and_shrinkage(self):
        effective, share, compression = normalize_usage([35, 30, 25, 15, 10], [48] * 5)
        self.assertAlmostEqual(sum(effective), 100.)
        self.assertAlmostEqual(sum(share), 1.)
        self.assertGreater(effective[0], effective[1])
        self.assertLess(effective[0] / effective[-1], 35 / 10)
        self.assertGreater(compression, 0)

    def test_no_current_or_future_profile(self):
        profiles = prior_profiles(self.members, 2016, self.histories, self.defaults, {}, 3)
        self.assertTrue((profiles.season_end_year == 2015).all())
        self.assertTrue((profiles.usage_pct == 20).all())
        self.histories["0"] = self.histories["0"].query("season_end_year == 2016")
        profiles = prior_profiles(self.members, 2016, self.histories, self.defaults, {}, 3)
        self.assertEqual(profiles.iloc[0].known, 0.)
        self.assertEqual(profiles.iloc[0].prior_mpg, 8.)

    def test_missing_and_returning_players(self):
        self.histories["0"]["season_end_year"] = 2013
        prior = prior_profiles(self.members, 2016, self.histories, self.defaults, {}, 3)
        self.assertEqual(prior.iloc[0].profile_lag, 3)
        self.assertEqual(prior.iloc[0].returner, 1)
        previous_only = prior_profiles(self.members, 2016, self.histories, self.defaults, {}, 1)
        self.assertEqual(previous_only.iloc[0].known, 0)

    def test_target_performance_and_minutes_do_not_change_deployable_inputs(self):
        teams = pd.DataFrame([dict(season_end_year=2016, season="2015-16", team_id="A",
                                   team_name="Test", off_rtg=110., def_rtg=108., pace=95.)])
        leagues = pd.DataFrame([self.context], index=[2015])
        original, _ = build_lagged(self.members, teams, leagues, self.histories, self.defaults, {})
        mutated = self.members.copy()
        mutated["minutes"] = 1000.
        mutated.loc[0, "minutes"] = 100000.
        mutated["usage_pct"], mutated["minutes_per_game"], mutated["relative_ts"] = 99., 47., .8
        teams[["off_rtg", "def_rtg", "pace"]] = [180., 50., 150.]
        changed, _ = build_lagged(mutated, teams, leagues, self.histories, self.defaults, {})
        a = original.query("mode == 'deployable'")[DEPLOYABLE_FEATURE_SET].to_numpy()
        b = changed.query("mode == 'deployable'")[DEPLOYABLE_FEATURE_SET].to_numpy()
        np.testing.assert_allclose(a, b)
        oracle_a = original.query("mode == 'oracle'")[DEPLOYABLE_FEATURE_SET].to_numpy()
        oracle_b = changed.query("mode == 'oracle'")[DEPLOYABLE_FEATURE_SET].to_numpy()
        self.assertFalse(np.allclose(oracle_a, oracle_b))

    def test_batch_independent_predictions(self):
        profiles = prior_profiles(self.members, 2016, self.histories, self.defaults, {}, 3)
        features, _, _ = aggregate(profiles, rotation(profiles.prior_mpg), self.context)
        frames = pd.DataFrame([features] * 5)
        model = pipeline(Ridge(alpha=30)).fit(frames[DEPLOYABLE_FEATURE_SET], np.arange(5))
        candidate = model, DEPLOYABLE_FEATURE_SET
        self.assertAlmostEqual(predict(frames.iloc[:1], "ortg", candidate)[0], predict(frames, "ortg", candidate)[0])
        other = frames.copy()
        other.loc[1:, "w_assist_pct"] = 9999.
        self.assertAlmostEqual(predict(frames.iloc[:1], "ortg", candidate)[0], predict(other, "ortg", candidate)[0])

    def test_usage_calibration_cannot_use_validation_or_test_outcomes(self):
        training = self.members.copy()
        training["season_end_year"] = 2014
        training["usage_pct"] = np.linspace(30, 10, len(training))
        histories = {pid: history.assign(season_end_year=[2013, 2016]) for pid, history in self.histories.items()}
        future = training.copy()
        future["season_end_year"], future["usage_pct"] = 2024, 99.
        expected = calibrate_usage(training, histories, self.defaults, {})
        actual = calibrate_usage(pd.concat([training, future]), histories, self.defaults, {})
        self.assertEqual(expected, actual)

    def test_source_identity_does_not_merge_same_name_players(self):
        source = export_cached_profiles([1989]).set_index(["player_id", "team"])
        first = source.loc[("br:jonesch01", "WSB")]
        second = source.loc[("br:jonesch02", "WSB")]
        self.assertEqual(first.source_minutes, 1154.)
        self.assertEqual(second.source_minutes, 516.)
        self.assertEqual(first.defensive_box_plus_minus, 2.3)
        self.assertEqual(second.defensive_box_plus_minus, .8)


if __name__ == "__main__":
    unittest.main()
