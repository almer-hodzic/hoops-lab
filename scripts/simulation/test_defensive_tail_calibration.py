"""Training-OOF provenance, mean calibration, stop rules and extrapolation guards."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import calibrate_defensive_tails as calibration


class DefensiveTailCalibrationTests(unittest.TestCase):
    def pairs(self, gamma=1.):
        x = np.linspace(-4., 4., 40)
        return pd.DataFrame(dict(season_end_year=np.repeat([2000, 2001], 20),
                                 team_id=[str(i) for i in range(40)], base_train_end=1999,
                                 anchor_drtg=107., base_prediction=107. + x,
                                 actual_drtg=107. + gamma * x))

    def test_gamma_is_estimated_not_forced_above_one(self):
        for gamma in [.8, 1., 1.25]:
            params = calibration.fit_calibration(self.pairs(gamma), "B_symmetric")
            self.assertAlmostEqual(params["gamma"], gamma)

    def test_calibrator_rejects_in_sample_duplicate_and_validation_pairs(self):
        frame = self.pairs()
        frame["base_train_end"] = frame.season_end_year
        with self.assertRaises(ValueError):
            calibration.fit_calibration(frame, "B_symmetric")
        with self.assertRaises(ValueError):
            calibration.fit_calibration(pd.concat([self.pairs(), self.pairs()]), "B_symmetric")
        frame = self.pairs().assign(season_end_year=2015, base_train_end=2014)
        with self.assertRaises(ValueError):
            calibration.fit_calibration(frame, "B_symmetric")

    def test_asymmetry_depends_only_on_predicted_residual_sign(self):
        frame = self.pairs()
        x = frame.base_prediction - frame.anchor_drtg
        frame["actual_drtg"] = 107. + np.where(x < 0, 1.1 * x, .7 * x)
        params = calibration.fit_calibration(frame, "C_asymmetric")
        self.assertAlmostEqual(params["gamma_better"], 1.1)
        self.assertAlmostEqual(params["gamma_worse"], .7)
        np.testing.assert_allclose(calibration.apply_calibration([105., 109.], [107., 107.], params), [104.8, 108.4])

    def test_unsupported_synthetic_value_is_preserved_without_universal_clamp(self):
        params = dict(kind="B_symmetric", gamma=1.5)
        raw = calibration.apply_calibration([99.], [107.], params)
        guarded = calibration.apply_calibration([99.], [107.], params, unsupported=[True])
        supported = calibration.apply_calibration([99.], [107.], params, unsupported=[False])
        self.assertEqual(raw[0], 95.)
        self.assertEqual(guarded[0], 99.)
        self.assertEqual(supported[0], raw[0])

    def test_stability_rejects_wild_coefficients_without_clipping(self):
        stable = [dict(gamma=v) for v in [.8, .9, 1., 1.05, 1.1, .95]]
        unstable = [dict(gamma=v) for v in [.2, 2.4, .4, 2.3, .3, 2.5]]
        self.assertTrue(calibration.coefficient_stability(stable)["stable"])
        self.assertFalse(calibration.coefficient_stability(unstable)["stable"])
        self.assertEqual(unstable[0]["gamma"], .2)

    def test_product_exception_does_not_mutate_hard_gate_failure(self):
        record = dict(legacy=dict(checks=dict(overall=True, bucket_bias=False, extremes=True,
                                              p90=True, suberas=True, box_heavy=True),
                                  diagnostics=dict(buckets={"poor": dict(bias=-3.1)})),
                      ranking=dict(accepted=True), residual_metrics=dict(residual_spearman=.5), accepted=False)
        self.assertTrue(calibration.marginal_product_rule(record, True, 6))
        self.assertFalse(record["accepted"])
        self.assertFalse(record["legacy"]["checks"]["bucket_bias"])
        record["legacy"]["diagnostics"]["buckets"]["poor"]["bias"] = -3.26
        self.assertFalse(calibration.marginal_product_rule(record, True, 6))

    def test_validation_is_single_use_and_blocks_further_preparation(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger, result = Path(directory) / "ledger.json", Path(directory) / "result.json"
            ledger.write_text("{}", encoding="utf-8")
            with patch.object(calibration, "LEDGER", ledger), patch.object(calibration, "RESULT", result):
                with self.assertRaisesRegex(RuntimeError, "already consumed"):
                    calibration.validate_stage()
                with self.assertRaisesRegex(RuntimeError, "already consumed"):
                    calibration.prepare_stage()

    def test_actual_training_output_has_strict_forward_lineage(self):
        if not calibration.TRAINING.exists() or not calibration.OOF_PATH.exists():
            self.skipTest("Run --prepare first for generated-data integration check")
        oof = pd.read_csv(calibration.OOF_PATH)
        calibration.assert_oof(oof)
        result = json.loads(calibration.TRAINING.read_text(encoding="utf-8"))
        for candidate in result["candidates"].values():
            for row in candidate["prequential"]:
                first_year = oof.loc[oof.fold == row["fold"], "season_end_year"].min()
                self.assertLess(row["parameters"]["fitting_max_year"], first_year)


if __name__ == "__main__":
    unittest.main()
