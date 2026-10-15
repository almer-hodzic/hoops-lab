"""Temporal lineage, score semantics, sampling and frozen-test guards."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import build_game_logs as games
import game_log_calibration as calibration


class GameCalibrationTests(unittest.TestCase):
    def history(self):
        rows=[]
        for gid, year, date, hp, ap in [(1,2010,"2010-04-01",100,95),
                                       (2,2011,"2010-11-01",110,100),
                                       (3,2011,"2010-11-01",90,95),
                                       (4,2011,"2010-11-03",102,98)]:
            rows.append(dict(game_id=str(gid),season_end_year=year,date=date,eligible=True,
                             home_franchise_id="A",away_franchise_id="B",pace=95.,
                             home_regulation_points=hp,away_regulation_points=ap))
        return pd.DataFrame(rows)

    def matchup(self):
        return pd.DataFrame(dict(season_end_year=[2016],league_ortg=[105.],league_pace=[95.],
                                 home_ortg=[110.],home_drtg=[103.],away_ortg=[105.],away_drtg=[108.],
                                 home_pace=[95.],away_pace=[95.]))

    def state(self):
        return dict(mean_model=dict(kind="A_formula",coefficient=np.array([1.,1.,0.,0.,0.,3.]),intercept=0.),
                    covariance=dict(pooled=np.diag([9.,100.,100.]),eras={}),variance_mode="pooled",
                    overtime=dict(samples=np.array([[-2.,1.],[1.,-2.],[0.,1.],[1.,0.]]),repeat_probability=.1))

    def test_existing_historical_identity_mapping_and_season_ids(self):
        self.assertEqual(games.season_end("0027900001"),1980)
        self.assertEqual(games.season_end("0022400001"),2025)
        seattle=games.identity("Seattle","SuperSonics",2007)
        okc=games.identity("Oklahoma City","Thunder",2010)
        self.assertEqual(seattle["franchise_id"],okc["franchise_id"])
        self.assertNotEqual(seattle["team_id"],okc["team_id"])
        old=games.identity("Charlotte","Hornets",2002)
        new=games.identity("Charlotte","Hornets",2016)
        self.assertNotEqual(old["franchise_id"],new["franchise_id"])
        self.assertEqual(games.identity("New Orleans","Hornets",2006)["team_id"],"team:new-orleans-oklahoma-city-hornets")

    def test_current_and_same_date_outcomes_cannot_change_pregame_features(self):
        original=self.history()
        before=games.add_pregame_strength(original)
        changed=original.copy()
        changed.loc[changed.game_id.isin(["2","3"]),"home_regulation_points"]=140
        after=games.add_pregame_strength(changed)
        fields=["home_ortg","away_drtg","league_ortg","home_pace","home_prior_games"]
        np.testing.assert_allclose(before.loc[before.game_id.isin(["2","3"]),fields],
                                   after.loc[after.game_id.isin(["2","3"]),fields])
        self.assertNotEqual(before.iloc[-1].home_ortg,after.iloc[-1].home_ortg)
        self.assertTrue((before.loc[before.home_ortg.notna(),"home_last_information_date"]<
                         before.loc[before.home_ortg.notna(),"date"]).all())

    def test_first_game_uses_previous_season_and_expansion_league_fallback(self):
        frame=self.history()
        frame.loc[frame.game_id=="2","home_franchise_id"]="expansion"
        result=games.add_pregame_strength(frame)
        first=result[result.game_id=="2"].iloc[0]
        self.assertEqual(first.home_prior_games,0)
        self.assertAlmostEqual(first.home_ortg,first.league_ortg)
        self.assertEqual(first.home_last_information_date,"2010-04-01")

    def test_formula_defensive_sign_and_pace_units(self):
        frame=self.matchup()
        result=calibration.mean_projection(frame,self.state()["mean_model"])
        self.assertEqual(result["pace"][0],95.)
        self.assertAlmostEqual(result["home_points"][0],(105+5+3+1.5)*.95)
        frame["away_drtg"]-=5
        better_defense=calibration.mean_projection(frame,self.state()["mean_model"])
        self.assertAlmostEqual(result["home_points"][0]-better_defense["home_points"][0],4.75)

    def test_high_uncertainty_never_changes_means_or_hidden_variance(self):
        frame,state=self.matchup(),self.state()
        standard=calibration.simulate(frame,state,count=1000,seed=42)
        high=calibration.simulate(frame,state,count=1000,seed=42,uncertainty="HIGH")
        np.testing.assert_array_equal(standard["home_scores"],high["home_scores"])
        np.testing.assert_array_equal(standard["away_scores"],high["away_scores"])
        self.assertIsNotNone(high["warning"])

    def test_monte_carlo_reproducibility_covariance_and_no_tied_final_scores(self):
        frame,state=self.matchup(),self.state()
        a=calibration.simulate(frame,state,count=10000,seed=7)
        b=calibration.simulate(frame,state,count=10000,seed=7)
        np.testing.assert_array_equal(a["home_scores"],b["home_scores"])
        self.assertTrue((a["home_scores"]!=a["away_scores"]).all())
        self.assertTrue((a["home_scores"]>=0).all())
        self.assertLess(abs(a["home_win_probability"]-a["analytic_probability"]),.025)
        self.assertGreater(np.corrcoef(a["home_scores"],a["away_scores"])[0,1],0.)

    def test_overtime_is_explicit_not_random_winner_coin_flip(self):
        frame=self.matchup().assign(home_ortg=105.,home_drtg=105.,away_ortg=105.,away_drtg=105.)
        state=self.state()
        state["mean_model"]["coefficient"][2:]=0
        state["covariance"]["pooled"]=np.eye(3)*.0001
        state["overtime"]["samples"]=np.array([[-1.,1.]])
        result=calibration.simulate(frame,state,count=1000,seed=11)
        self.assertGreater(result["ot_rate"],.99)
        self.assertTrue((result["away_scores"]>result["home_scores"]).all())

    def test_product_covariance_matches_empirical_sampling(self):
        means=dict(pace=np.array([95.]),home_efficiency=np.array([112.]),away_efficiency=np.array([108.]))
        covariance=np.array([[16.,1.,-2.],[1.,100.,5.],[-2.,5.,121.]])
        expected=calibration.score_covariance(means,covariance[None])[0]
        z=np.random.default_rng(12).multivariate_normal(np.zeros(3),covariance,200000)
        hp=(95+z[:,0])*(112+z[:,1])/100
        ap=(95+z[:,0])*(108+z[:,2])/100
        np.testing.assert_allclose(np.cov(np.column_stack([hp,ap]),rowvar=False),expected,rtol=.025,atol=.1)

    def test_empirical_close_game_atom_increases_ot_without_final_ties(self):
        frame,state=self.matchup(),self.state()
        baseline=calibration.simulate(frame,state,count=30000,seed=5)
        state["overtime"]["tie_calibration"]={"extra_tie_probability":.2}
        corrected=calibration.simulate(frame,state,count=30000,seed=5)
        self.assertGreater(corrected["ot_rate"],baseline["ot_rate"]+.015)
        self.assertTrue((corrected["home_scores"]!=corrected["away_scores"]).all())

    def test_train_only_fit_and_single_use_test_guards(self):
        with self.assertRaises(ValueError):
            calibration.fit_mean(pd.DataFrame(dict(season_end_year=[2015])),"A_formula")
        with tempfile.TemporaryDirectory() as directory:
            ledger=Path(directory)/"ledger.json"
            ledger.write_text("{}")
            with patch.object(calibration,"LEDGER",ledger):
                with self.assertRaisesRegex(RuntimeError,"consumed"):
                    calibration.final_stage()
                with self.assertRaisesRegex(RuntimeError,"consumed"):
                    calibration.develop()

    def test_calibration_bins_include_decimal_boundaries_exactly_once(self):
        p=np.array([0.,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.])
        bins=calibration.calibration_bins(np.ones(len(p)),p)
        self.assertEqual(sum(row["n"] for row in bins),len(p))
        self.assertEqual(bins[-1]["n"],2)

    def test_zero_coded_unavailable_shooting_does_not_create_fake_possessions(self):
        index=pd.DataFrame([dict(gameId=21300001,gameDateTimeEst="2013-10-30 19:00:00",
            hometeamId=1,awayteamId=2,hometeamCity="Boston",hometeamName="Celtics",
            awayteamCity="New York",awayteamName="Knicks",homeScore=100,awayScore=90)])
        boxes=[]
        for tid,score,quarters in [(1,100,[25,25,25,25]),(2,90,[20,20,25,25])]:
            boxes.append(dict(gameId=21300001,teamId=tid,teamScore=score,numMinutes=240.,
                **{f"q{i+1}Points":v for i,v in enumerate(quarters)},fieldGoalsAttempted=0,
                fieldGoalsMade=0,freeThrowsAttempted=20,reboundsOffensive=0,turnovers=0,possessions=np.nan))
        result,rejected=games.canonical_games(index,pd.DataFrame(boxes))
        self.assertEqual(len(rejected),0)
        self.assertTrue(pd.isna(result.home_possessions.iloc[0]))
        self.assertTrue(pd.isna(result.home_game_ortg.iloc[0]))
        self.assertEqual(result.home_possession_method.iloc[0],"unavailable_zero_coded_box")
        self.assertFalse(result.eligible.iloc[0])

    def test_source_cache_pinned_and_canonical_lineage_if_built(self):
        if not games.QUALITY.exists():
            self.skipTest("Canonical games not built")
        quality=json.loads(games.QUALITY.read_text())
        frame=pd.read_csv(games.OUTPUT,dtype={"game_id":str})
        self.assertEqual(len(frame),quality["games"])
        self.assertFalse(frame.game_id.duplicated().any())
        self.assertTrue(frame.game_id.str.startswith("002").all())
        self.assertTrue(frame.season_end_year.between(1980,2025).all())
        for side in ("home","away"):
            supported=frame[frame[side+"_last_information_date"].notna()]
            self.assertTrue((supported[side+"_last_information_date"]<supported.date).all())
        self.assertTrue((frame.home_points!=frame.away_points).all())
        self.assertFalse(calibration.ARTIFACT.parent.relative_to(games.ROOT).as_posix().startswith("data"))


if __name__=="__main__":
    unittest.main()
