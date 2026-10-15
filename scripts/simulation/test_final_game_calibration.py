"""Final-pass leakage boundaries, joint sampling, roster transport and stop rules."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import finalize_game_calibration as model
import final_game_data as data
import final_game_engine as bridge


class FinalGameCalibrationTests(unittest.TestCase):
    def historical(self):
        rows=[]
        for year in [2010,2011,2012,2013]:
            for _ in range(30):
                rows.append(dict(season_end_year=year,home_pace=100.,away_pace=100.,league_pace=100.,league_ortg=100.,
                    home_ortg=100.,away_ortg=100.,home_drtg=100.,away_drtg=100.,
                    home_regulation_points=103.,away_regulation_points=99.))
        return pd.DataFrame(rows)

    def matchup(self):
        return pd.DataFrame(dict(season_end_year=[2026],home_pace=[100.],away_pace=[100.],league_pace=[100.],league_ortg=[110.],
            home_ortg=[110.],away_ortg=[110.],home_drtg=[110.],away_drtg=[110.]))

    def sampler(self,kind="A_pooled_gaussian"):
        covariance=np.array([[.002,.0001,.0001],[.0001,.01,.004],[.0001,.004,.01]])
        return dict(kind=kind,covariance=covariance,cholesky=np.linalg.cholesky(covariance),df=8.)

    def home(self):return dict(common_offset=0.,home_margin_per100=0.)

    def ot(self):return dict(extra=0.,thinning=0.,samples=np.array([[-.15,.15],[.15,-.15]]))

    def test_home_effect_uses_strictly_earlier_seasons(self):
        h=self.historical()
        first=model.home_parameters(h,2013,"trailing5")
        h.loc[h.season_end_year==2013,"home_regulation_points"]=1000
        second=model.home_parameters(h,2013,"trailing5")
        self.assertEqual(first,second)
        self.assertEqual(first["fitting_max_year"],2012)
        self.assertAlmostEqual(first["home_margin_per100"],4.)
        self.assertAlmostEqual(first["common_offset"],1.)

    def test_old_test_years_never_enter_home_parameter_fit(self):
        h=self.historical()
        future=h.iloc[:1].assign(season_end_year=2025,home_regulation_points=10000)
        a=model.home_parameters(h,2026,"exponential5")
        b=model.home_parameters(pd.concat([h,future]),2026,"exponential5")
        self.assertEqual(a,b)

    def test_all_home_models_remain_prospective(self):
        for kind in model.HOME_MODELS:
            p=model.home_parameters(self.historical(),2013,kind)
            self.assertLess(p["fitting_max_year"],2013)
            self.assertAlmostEqual(p["home_margin_per100"],4.)

    def test_history_guard_rejects_current_and_post2019_targets(self):
        for frame,year in [(pd.DataFrame(dict(season_end_year=[2015])),2015),
                           (pd.DataFrame(dict(season_end_year=[2020])),2026)]:
            with self.assertRaises(ValueError):model.assert_history(frame,year)
        with self.assertRaises(ValueError):model.residual_frame(pd.DataFrame(dict(season_end_year=[2026])),"trailing5")

    def test_sampler_fit_excludes_old_and_new_holdouts(self):
        rng=np.random.default_rng(3)
        history=pd.DataFrame(rng.normal(size=(600,3)),columns=model.RESID)
        history["season_end_year"]=2014
        future=history.iloc[:1].copy();future[model.RESID]=10000.;future["season_end_year"]=2026
        a=model.fit_sampler(history,2026,"A_pooled_gaussian")
        b=model.fit_sampler(pd.concat([history,future]),2026,"A_pooled_gaussian")
        np.testing.assert_allclose(a["covariance"],b["covariance"])
        self.assertEqual(a["fitting_max_year"],2014)

    def test_bootstrap_draws_whole_joint_vectors(self):
        vectors=np.array([[.01,.02,.03],[-.01,-.02,-.03],[0.,0.,0.]])
        model_state=dict(kind="D_bootstrap8",samples=vectors)
        draws=model.draw_residuals(model_state,(100,20),np.random.default_rng(4))
        np.testing.assert_allclose(draws[:,:,1],draws[:,:,0]*2)
        np.testing.assert_allclose(draws[:,:,2],draws[:,:,0]*3)

    def test_student_shared_scale_preserves_empirical_covariance(self):
        state=self.sampler("E_student_t")
        draws=model.draw_residuals(state,(1,100000),np.random.default_rng(5))[0]
        np.testing.assert_allclose(np.cov(draws,rowvar=False),state["covariance"],rtol=.10,atol=.00015)

    def test_environment_scaling_is_not_independent_score_noise(self):
        state=self.sampler();frame=self.matchup()
        a=model.simulate(frame,self.home(),state,self.ot(),5000,seed=9)
        higher=frame.copy();higher[["league_ortg","home_ortg","away_ortg","home_drtg","away_drtg"]]*=1.2
        b=model.simulate(higher,self.home(),state,self.ot(),5000,seed=9)
        self.assertGreater(b.hi95.iloc[0]-b.lo95.iloc[0],a.hi95.iloc[0]-a.lo95.iloc[0])

    def test_seeded_sampling_and_high_flag_do_not_change_quality(self):
        frame=self.matchup();state=self.sampler()
        a=model.simulate(frame,self.home(),state,self.ot(),1000,seed=12)
        b=model.simulate(frame,self.home(),state,self.ot(),1000,seed=12,uncertainty="HIGH")
        pd.testing.assert_frame_equal(a.select_dtypes("number"),b.select_dtypes("number"))
        self.assertEqual(b.uncertainty_label.iloc[0],"HIGH")
        self.assertTrue(b.uncertainty_warning.iloc[0])
        with patch.object(model.np.random,"default_rng",wraps=np.random.default_rng) as random_generator:
            model.simulate(frame,self.home(),state,self.ot(),1000,seed=None)
            random_generator.assert_called_once_with(None)

    def test_ties_resolved_or_fail_loudly(self):
        frame=self.matchup()
        state=dict(kind="A_pooled_gaussian",covariance=np.zeros((3,3)),cholesky=np.zeros((3,3)))
        a=model.simulate(frame,self.home(),state,self.ot(),1000,seed=2)
        self.assertEqual(a.simulated_ot.iloc[0],1.)
        self.assertEqual(a.final_ties.iloc[0],0)
        with self.assertRaises(RuntimeError):
            model.simulate(frame,self.home(),state,dict(extra=0.,thinning=0.,samples=np.zeros((2,2))),1000,seed=2)

    def test_transport_preserves_strength_deviations(self):
        frame=pd.DataFrame(dict(season_end_year=[2026],home_team_id=["a"],away_team_id=["b"],league_ortg=[120.],league_pace=[100.]))
        strength=pd.DataFrame([dict(season_end_year=2026,team_id=tid,ortg=110.,drtg=104.,pace=92.,
            source_anchor_ortg=105.,source_anchor_pace=90.,uncertainty="HIGH") for tid in ["a","b"]])
        transformed=bridge.transport_games(frame,strength)
        self.assertEqual(transformed.home_ortg.iloc[0],125.)
        self.assertEqual(transformed.away_drtg.iloc[0],119.)
        self.assertEqual(transformed.home_pace.iloc[0],102.)

    def test_final_interval_metrics_are_not_regulation_proxy(self):
        frame=pd.DataFrame(dict(home_win=[1],home_points=[100.],away_points=[90.],margin=[10.],regulation_margin=[0.],overtime_periods=[1]))
        pred=pd.DataFrame(dict(probability=[.6],expected_home=[100.],expected_away=[90.],simulated_ot=[.05]))
        for level in model.LEVELS:
            n=str(int(level*100));pred["lo"+n]=-1;pred["hi"+n]=5;pred["reg_lo"+n]=-1;pred["reg_hi"+n]=1
        metric=model.metrics(frame,pred)
        self.assertEqual(metric["coverage95"],0.)
        self.assertEqual(metric["regulation_coverage95"],1.)

    def test_coverage_gate_cannot_be_loosened_after_holdout(self):
        metric=dict(coverage95=.94,ece=.02,log_loss=.6,brier=.22,score_mae=9.,margin_mae=11.,total_mae=15.,
                    observed_ot=.05,simulated_ot=.05,home_margin_bias=0.)
        verdict,_=model.decision(metric,metric,.25)
        self.assertTrue(verdict.startswith("B."))
        metric["coverage95"]=.919
        verdict,_=model.decision(metric,metric,.25)
        self.assertTrue(verdict.startswith("C."))

    def test_holdout_outcome_reader_is_sealed(self):
        with self.assertRaises(RuntimeError):data.read_holdout_outcomes()

    def test_holdout_is_single_use_and_prevents_further_development(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger=Path(directory)/"ledger.json";ledger.write_text("{}")
            with patch.object(model,"LEDGER",ledger):
                with self.assertRaisesRegex(RuntimeError,"consumed"):model.develop()
                with self.assertRaisesRegex(RuntimeError,"consumed"):model.final_stage()

    def test_identity_rosters_contain_no_target_performance(self):
        if not data.ROSTERS.exists():self.skipTest("Acquire blind rosters first")
        roster=pd.read_csv(data.ROSTERS)
        self.assertFalse(set(roster.columns)&{"points","numMinutes","startingPosition","win","fieldGoalsMade"})
        acquisition=json.loads(data.ACQUISITION.read_text())
        self.assertFalse(acquisition["outcome_columns_accessed_during_acquisition"])
        self.assertEqual(acquisition["holdout_games"],1230)

    def test_frozen_roster_projection_lineage(self):
        if not bridge.OUTPUT.exists():self.skipTest("Build frozen roster projections first")
        strength=pd.read_csv(bridge.OUTPUT)
        known=strength.dropna(subset=["latest_player_profile_year"])
        self.assertTrue((known.latest_player_profile_year<known.season_end_year).all())
        np.testing.assert_allclose(strength.minutes_sum,240.)
        self.assertEqual(len(strength[strength.season_end_year==2026]),30)


if __name__=="__main__":unittest.main()
