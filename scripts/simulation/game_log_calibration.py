"""Research-only game calibration: --develop, tests, --final once, --report offline.

No team-strength retraining, production API, or UI. The final game test is separate
from the old team-model benchmark, whose artifacts are never changed/evaluated.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import ndtr
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from acquire_game_logs import ROOT, MANIFEST, fingerprint
from build_game_logs import OUTPUT, QUALITY

OUT = ROOT / "data/simulation"
SUMMARY = OUT / "game_calibration_summary.json"
PROTOCOL = OUT / "game_calibration_protocol.json"
LEDGER = OUT / "game_calibration_final_run.json"
ARTIFACT = ROOT / "artifacts/simulation/game_calibration_v1.joblib"
REPORT = ROOT / "reports/simulation/GAME_LOG_CALIBRATION.md"
KINDS = ["A_formula", "B_ridge30"]
FOLDS = [(1985, 1990), (1990, 1995), (1995, 2000), (2000, 2005), (2005, 2010), (2010, 2014)]
SEED = 20261015


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def write_json(path, value):
    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def protected_hashes():
    paths = list((ROOT / "data/historical/processed").glob("*.csv"))
    paths += [p for p in (ROOT / "artifacts/simulation").glob("*.joblib") if not p.name.startswith("game_calibration")]
    paths += [p for p in OUT.glob("*") if p.is_file() and not p.name.startswith("game_")]
    paths += [ROOT / "reports/simulation/SIMULATION_ENGINE_DESIGN.md"]
    return {str(p.relative_to(ROOT)).replace("\\", "/"): fingerprint(p) for p in paths}


def era(year):
    return np.searchsorted([1989, 1999, 2009], np.asarray(year), side="left")


def design(frame, side):
    other = "away" if side == "home" else "home"
    e = era(frame.season_end_year)
    sign = .5 if side == "home" else -.5
    return np.column_stack([frame[side + "_ortg"] - frame.league_ortg,
                            frame[other + "_drtg"] - frame.league_ortg,
                            *[sign * (e == i) for i in range(4)]])


def fit_mean(frame, kind):
    if frame.season_end_year.max() > 2014:
        raise ValueError("Game coefficients must use TRAIN only")
    x = np.vstack([design(frame, "home"), design(frame, "away")])
    y = np.concatenate([frame[side + "_regulation_points"] / frame.pace * 100 - frame.league_ortg
                        for side in ("home", "away")])
    if kind == "A_formula":
        residual = y - x[:, :2].sum(axis=1)
        common = float(residual.mean())
        h = np.linalg.lstsq(x[:, 2:], residual-common, rcond=None)[0]
        model = dict(kind=kind, intercept=common, coefficient=np.r_[1., 1., h])
    else:
        model = dict(kind=kind, estimator=make_pipeline(StandardScaler(), Ridge(alpha=30.)).fit(x, y))
    # Formula pace is the team mean; ridge calibrates how it maps to game pace.
    p = ((frame.home_pace + frame.away_pace) / 2 - frame.league_pace).to_numpy()[:, None]
    model["pace_model"] = make_pipeline(StandardScaler(), Ridge(alpha=30.)).fit(p, frame.pace-frame.league_pace)
    return model


def mean_projection(frame, model):
    pdev = ((frame.home_pace+frame.away_pace)/2-frame.league_pace).to_numpy()[:, None]
    p = ((frame.home_pace+frame.away_pace)/2).to_numpy() if model["kind"] == "A_formula" else (
        frame.league_pace.to_numpy()+model["pace_model"].predict(pdev))
    result = {"pace": p}
    for side in ("home", "away"):
        x = design(frame, side)
        delta = x @ model["coefficient"] + model["intercept"] if model["kind"] == "A_formula" else model["estimator"].predict(x)
        result[side + "_efficiency"] = frame.league_ortg.to_numpy() + delta
        result[side + "_points"] = p * result[side + "_efficiency"] / 100
    return result


def residuals(frame, means):
    return np.column_stack([frame.pace-means["pace"],
                            frame.home_regulation_points/frame.pace*100-means["home_efficiency"],
                            frame.away_regulation_points/frame.pace*100-means["away_efficiency"]])


def expanding_oof(train, kind):
    frames = []
    for end, stop in FOLDS:
        before, held = train[train.season_end_year <= end], train[train.season_end_year.between(end+1, stop)]
        model = fit_mean(before, kind)
        means = mean_projection(held, model)
        output = held.copy()
        for k, v in means.items():
            output["expected_" + k] = v
        output[["pace_residual", "home_eff_residual", "away_eff_residual"]] = residuals(held, means)
        output["base_train_end"] = end
        frames.append(output)
    result = pd.concat(frames, ignore_index=True)
    if not (result.base_train_end < result.season_end_year).all():
        raise ValueError("Non-forward residual estimation")
    return result


def covariance_model(oof):
    columns = ["pace_residual", "home_eff_residual", "away_eff_residual"]
    pooled = np.cov(oof[columns].to_numpy(), rowvar=False)
    groups = {}
    for index, group in oof.groupby(era(oof.season_end_year)):
        weight = len(group) / (len(group)+500.)
        groups[int(index)] = weight*np.cov(group[columns].to_numpy(), rowvar=False)+(1-weight)*pooled
    return dict(pooled=pooled, eras=groups, shrinkage_pseudo_games=500)


def covariance_for(frame, model, mode):
    if mode == "pooled":
        return np.repeat(model["pooled"][None], len(frame), axis=0)
    return np.array([model["eras"].get(int(i), model["pooled"]) for i in era(frame.season_end_year)])


def score_covariance(means, latent):
    p = means["pace"]
    jac = np.zeros((len(p), 2, 3))
    jac[:, 0, 0], jac[:, 1, 0] = means["home_efficiency"]/100, means["away_efficiency"]/100
    jac[:, 0, 1], jac[:, 1, 2] = p/100, p/100
    covariance = np.einsum("nik,nkl,njl->nij", jac, latent, jac)
    # Exact second-order product variance for centered multivariate normals.
    for i in range(2):
        for j in range(2):
            covariance[:, i, j] += (latent[:, 0, 0]*latent[:, i+1, j+1] + latent[:, 0, i+1]*latent[:, 0, j+1])/10000
    return covariance


def overtime_model(oof):
    ot = oof[oof.overtime_periods == 1]
    samples = np.column_stack([ot[side + "_points"]-ot[side + "_regulation_points"]-
                               ot["expected_" + side + "_points"]*5/48 for side in ("home", "away")])
    if len(samples) < 100:
        raise ValueError("Insufficient genuine overtime support")
    samples -= samples.mean(axis=0)
    all_ot = oof[oof.overtime_periods > 0]
    return dict(samples=samples, fitting_games=len(ot), repeat_probability=float((all_ot.overtime_periods > 1).mean()),
                observed_game_ot_rate=float((oof.overtime_periods > 0).mean()))


def fit_tie_mass(oof, covariance, mode):
    if oof.season_end_year.max()>2014 or not (oof.base_train_end<oof.season_end_year).all():
        raise ValueError("Tie correction must use genuine TRAIN-OOF pairs")
    means={k:oof["expected_"+k].to_numpy() for k in ["pace","home_efficiency","away_efficiency","home_points","away_points"]}
    c=score_covariance(means,covariance_for(oof,covariance,mode))
    mu=means["home_points"]-means["away_points"]
    sd=np.sqrt(c[:,0,0]+c[:,1,1]-2*c[:,0,1])
    base=ndtr((.5-mu)/sd)-ndtr((-.5-mu)/sd)
    near=ndtr((3.5-mu)/sd)-ndtr((.5-mu)/sd)+ndtr((-.5-mu)/sd)-ndtr((-3.5-mu)/sd)
    observed=float((oof.regulation_margin==0).mean())
    probability=max(0.,float((observed-base.mean())/near.mean()))
    if probability>1:
        raise ValueError("Close-game tie model has insufficient support")
    return dict(extra_tie_probability=probability, gaussian_tie_rate=float(base.mean()), observed_tie_rate=observed,
                fitting_rows=len(oof), definition="One fitted Bernoulli tie atom for rounded1-3 point margins; TRAIN only")


def projection(frame, state):
    means = mean_projection(frame, state["mean_model"])
    latent = covariance_for(frame, state["covariance"], state["variance_mode"])
    covariance = score_covariance(means, latent)
    margin = means["home_points"]-means["away_points"]
    sd = np.sqrt(covariance[:, 0, 0]+covariance[:, 1, 1]-2*covariance[:, 0, 1])
    positive=1-ndtr((.5-margin)/sd)
    near_positive=ndtr((3.5-margin)/sd)-ndtr((.5-margin)/sd)
    near_negative=ndtr((-.5-margin)/sd)-ndtr((-3.5-margin)/sd)
    extra=state["overtime"].get("tie_calibration",{}).get("extra_tie_probability",0.)
    tie=ndtr((.5-margin)/sd)-ndtr((-.5-margin)/sd)+extra*(near_positive+near_negative)
    ot_samples=state["overtime"]["samples"]
    ot_sd=float(np.std(np.diff(ot_samples,axis=1),ddof=1)) if len(ot_samples)>1 else 0.
    ot_probability=ndtr(margin*5/48/ot_sd) if ot_sd>0 else .5+.5*np.sign(margin)
    p = positive-extra*near_positive+tie*ot_probability
    # Analytic final-score expectation approximates discrete ties; MC resolves them explicitly.
    repeats = 1/(1-state["overtime"]["repeat_probability"])
    home = means["home_points"]*(1+tie*5/48*repeats)
    away = means["away_points"]*(1+tie*5/48*repeats)
    return dict(means=means, latent=latent, covariance=covariance, probability=p, margin_sd=sd,
                home=home, away=away)


def calibration_bins(y, p):
    result = []
    index = np.minimum(np.floor(np.asarray(p)*10).astype(int), 9)
    for i in range(10):
        low, mask = i/10., index == i
        if mask.any():
            result.append(dict(low=float(low), n=int(mask.sum()), predicted=float(p[mask].mean()), actual=float(y[mask].mean())))
    return result


def metrics(frame, pred, probability=None):
    p = np.clip(pred["probability"] if probability is None else probability, 1e-9, 1-1e-9)
    y = frame.home_win.to_numpy()
    hp, ap = pred["home"], pred["away"]
    bins = calibration_bins(y, p)
    error = frame.regulation_margin.to_numpy()-(pred["means"]["home_points"]-pred["means"]["away_points"])
    sd = pred["margin_sd"]
    return dict(n=len(frame), score_mae=float(np.mean(np.r_[abs(hp-frame.home_points), abs(ap-frame.away_points)])),
                margin_mae=float(np.mean(abs(hp-ap-frame.margin))),
                total_mae=float(np.mean(abs(hp+ap-frame.home_points-frame.away_points))),
                margin_bias=float(np.mean(hp-ap-frame.margin)),
                brier=float(np.mean((p-y)**2)), log_loss=float(np.mean(-y*np.log(p)-(1-y)*np.log(1-p))),
                accuracy=float(np.mean((p>=.5)==y)),
                ece=sum(row["n"]*abs(row["predicted"]-row["actual"]) for row in bins)/len(y),
                margin_coverage80=float(np.mean(abs(error)<=1.281551565545*sd)),
                margin_coverage95=float(np.mean(abs(error)<=1.95996398454*sd)),
                calibration_bins=bins)


def simulate(frame, state, count=10000, seed=SEED, uncertainty="STANDARD"):
    if len(frame) != 1 or count < 100:
        raise ValueError("One matchup and at least100 draws required")
    if uncertainty not in {"STANDARD", "HIGH"}:
        raise ValueError("Unknown uncertainty state")
    pred = projection(frame, state)
    m, c = pred["means"], pred["latent"][0]
    rng = np.random.default_rng(seed)
    z = rng.multivariate_normal(np.zeros(3), c, size=count)
    pace = np.maximum(1., m["pace"][0]+z[:, 0])
    hp = np.maximum(0, np.rint(pace*(m["home_efficiency"][0]+z[:, 1])/100-c[0, 1]/100)).astype(int)
    ap = np.maximum(0, np.rint(pace*(m["away_efficiency"][0]+z[:, 2])/100-c[0, 2]/100)).astype(int)
    extra=state["overtime"].get("tie_calibration",{}).get("extra_tie_probability",0.)
    if extra>0:
        close=(abs(hp-ap)>=1)&(abs(hp-ap)<=3)
        forced=close & (rng.random(count)<extra)
        hp[forced]=np.rint((hp[forced]+ap[forced])/2).astype(int)
        ap[forced]=hp[forced]
    rounds = np.zeros(count, int)
    ot_samples = state["overtime"]["samples"]
    tied = hp == ap
    for _ in range(20):
        if not tied.any():
            break
        samples = ot_samples[rng.integers(0, len(ot_samples), tied.sum())]
        hp[tied] += np.maximum(0, np.rint(m["home_points"][0]*5/48+samples[:, 0])).astype(int)
        ap[tied] += np.maximum(0, np.rint(m["away_points"][0]*5/48+samples[:, 1])).astype(int)
        rounds[tied] += 1
        tied = hp == ap
    if tied.any():
        raise RuntimeError("Overtime simulation failed to resolve; no silent coin flip")
    return dict(home_scores=hp, away_scores=ap, home_win_probability=float((hp>ap).mean()),
                analytic_probability=float(pred["probability"][0]), ot_rate=float((rounds>0).mean()),
                mean_home_points=float(hp.mean()), mean_away_points=float(ap.mean()),
                margin_interval80=np.quantile(hp-ap, [.1,.9]).tolist(),
                mean_possessions=float(pace.mean()), count=count, seed=seed, uncertainty=uncertainty,
                warning="HIGH coverage/extrapolation: no calibrated numeric epistemic widening available" if uncertainty=="HIGH" else None)


def empirical_diagnostics(frame):
    rows = []
    for block, group in frame.groupby(era(frame.season_end_year)):
        hp, ap = group.home_points.to_numpy(), group.away_points.to_numpy()
        home_res = group.home_regulation_points-(group.home_ortg+group.away_drtg-group.league_ortg)*group.pace/100
        away_res = group.away_regulation_points-(group.away_ortg+group.home_drtg-group.league_ortg)*group.pace/100
        rows.append(dict(block=int(block), start=group.season.min(), end=group.season.max(), n=len(group),
            home_margin=float((hp-ap).mean()), home_win_rate=float(group.home_win.mean()),
            home_offensive_effect=float(home_res.mean()), home_defensive_effect=float(-away_res.mean()),
            team_score_sd=float(np.std(np.r_[hp,ap],ddof=1)), total_sd=float(np.std(hp+ap,ddof=1)),
            margin_sd=float(np.std(hp-ap,ddof=1)), pace_sd=float(group.pace.std()),
            home_residual_sd=float(home_res.std()), away_residual_sd=float(away_res.std())))
    return rows


def gates(m):
    return dict(score_mae=m["score_mae"]<=11., margin_mae=m["margin_mae"]<=13.5,
                total_mae=m["total_mae"]<=17.5, brier=m["brier"]<=.23, log_loss=m["log_loss"]<=.66,
                calibration=m["ece"]<=.04, coverage80=.75<=m["margin_coverage80"]<=.85,
                coverage95=.92<=m["margin_coverage95"]<=.98)


def equations(model):
    if model["kind"]=="A_formula":
        coefficient, intercept = model["coefficient"], model["intercept"]
        pace = dict(slope=1., intercept=0.)
    else:
        scale, reg = model["estimator"].steps[0][1], model["estimator"].steps[1][1]
        coefficient = reg.coef_/scale.scale_
        intercept = reg.intercept_-coefficient@scale.mean_
        ps, pr = model["pace_model"].steps[0][1], model["pace_model"].steps[1][1]
        pace = dict(slope=float(pr.coef_[0]/ps.scale_[0]), intercept=float(pr.intercept_-pr.coef_[0]/ps.scale_[0]*ps.mean_[0]))
    return dict(efficiency_intercept=float(intercept), own_offense=float(coefficient[0]),
                opponent_defense=float(coefficient[1]), home_minus_away_effect_per100_by_era=coefficient[2:].tolist(),
                pace=pace, era_blocks=["through1989", "1990-1999", "2000-2009", "2010 onward; TRAIN fit through2014 only"])


def read_development():
    data = pd.read_csv(OUTPUT, dtype={"game_id":str})
    return data[data.eligible & data.home_ortg.notna() & ~data.neutral_site & (data.season_end_year<=2019)].copy()


def develop():
    if LEDGER.exists():
        raise RuntimeError("Final test consumed: no further fitting/selection")
    protected = protected_hashes()
    frame = read_development()
    train, validation = frame[frame.season_end_year<=2014], frame[frame.season_end_year>=2015]
    states, results, diagnostics, oof_folds = {}, {}, {}, []
    for kind in KINDS:
        print("Training chronological game calibration: " + kind, flush=True)
        oof = expanding_oof(train, kind)
        cov, ot, model = covariance_model(oof), overtime_model(oof), fit_mean(train, kind)
        diagnostics[kind] = dict(equations=equations(model), covariance=cov, overtime={k:v for k,v in ot.items() if k!="samples"},
                                 empirical_variance_by_era=empirical_diagnostics(train))
        for end, group in oof.groupby("base_train_end"):
            oof_folds.append(dict(candidate=kind, train_end=int(end), evaluate_start=int(group.season_end_year.min()),
                evaluate_end=int(group.season_end_year.max()), n=len(group),
                pace_residual_sd=float(group.pace_residual.std()), home_efficiency_residual_sd=float(group.home_eff_residual.std()),
                away_efficiency_residual_sd=float(group.away_eff_residual.std())))
        for mode in ("pooled", "era"):
            name = kind+"_"+mode
            state = dict(mean_model=model, covariance=cov, overtime={**ot,"tie_calibration":fit_tie_mass(oof,cov,mode)}, variance_mode=mode)
            states[name] = state
            pred = projection(validation, state)
            results[name] = metrics(validation, pred)
    best_margin = min(r["margin_mae"] for r in results.values())
    competitive = [k for k in states if results[k]["margin_mae"]<=best_margin+.5]
    best = min(competitive, key=lambda k: results[k]["log_loss"])
    selected = next(k for k in states if k in competitive and results[k]["log_loss"]<=results[best]["log_loss"]+.002
                    and results[k]["score_mae"]<=results[best]["score_mae"]+.2)
    state = states[selected]
    selected_oof = expanding_oof(train, state["mean_model"]["kind"])
    # A contextual historical bridge: inference only, not another strength-model fit.
    bridge = frozen_engine_bridge(validation, state)
    timings = []
    for count in (1000, 10000, 50000):
        duration = []
        for _ in range(3):
            start = time.perf_counter()
            simulate(validation.iloc[:1], state, count=count)
            duration.append(time.perf_counter()-start)
        timings.append(dict(count=count, seconds_median=float(np.median(duration)),
                            worst_case_win_probability_mc_se=float(.5/np.sqrt(count))))
    summary = dict(status="frozen_before_final_game_test", selected=selected,
        split=dict(train=f"{train.season.min()} through2013-14 usable pregame strengths; prior years are source/warmup only", validation="2014-15 through2018-19", test="2019-20 through2024-25"),
        counts=dict(train=len(train), validation=len(validation), oof=len(selected_oof)),
        selected_overtime={k:v for k,v in state["overtime"].items() if k!="samples"},
        validation_candidates=results, training_diagnostics=diagnostics, forward_residual_folds=oof_folds,
        validation_selected_gates=gates(results[selected]), frozen_engine_validation_bridge=bridge,
        validation_empirical=empirical_diagnostics(validation), runtime_benchmarks=timings,
        simulation_count_default=10000, simulation_count_minimum=1000,
        protected_hashes=protected, current_team_engine_retrained=False,
        uncertainty="Keep point means unchanged. HIGH is an explicit unsupported-domain warning, not a calibrated probability interval. No arbitrary variance multiplier.",
        limitations=["Game calibration primarily validated with prior-date box-score strength estimates, not timestamped player rotations",
                     "Frozen team-engine bridge uses retrospectively known season membership; not a fully prospective roster test",
                     "2010-14 learned home effect and latest training-era covariance extrapolated to newer eras without refitting",
                     "Only identified Orlando neutral games excluded; other neutral sites may remain",
                     "Possession estimator changes around1996-97; early box scores have missingness",
                     "1979-80/1980-81 and much of1981-82 have zero-coded unavailable FGA; retained scores cannot supply valid possessions",
                     "Pinned source has five missing2024-25 games plus four invalid tied/zero-score records quarantined across all years",
                     "Gaussian latent residuals approximate tails; overtime bootstrap is not possession-by-possession"])
    if protected != protected_hashes():
        raise RuntimeError("Existing research artifact changed")
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(state, ARTIFACT)
    write_json(SUMMARY, summary)
    protocol = dict(status="frozen", selected=selected, dataset_sha256=fingerprint(OUTPUT), quality_sha256=fingerprint(QUALITY),
        source_manifest_sha256=fingerprint(MANIFEST), artifact_sha256=fingerprint(ARTIFACT), summary_sha256=fingerprint(SUMMARY),
        source_hashes={p.name:fingerprint(p) for p in [Path(__file__), Path(__file__).with_name("build_game_logs.py")]},
        protected_hashes=protected, final_test="2019-20 through2024-25; eligible non-neutral regular-season games; one selected model evaluation",
        thresholds=dict(score_mae_max=11., margin_mae_max=13.5, total_mae_max=17.5, brier_max=.23, log_loss_max=.66,
                        ece_max=.04, coverage80=[.75,.85], coverage95=[.92,.98], season_abs_margin_bias_max=3.),
        selection="Validation log loss among candidates within0.5 margin MAE; prefer simpler within0.002 log loss and0.2 score MAE",
        final_verdict_rule="A requires metric gates, season stability, and fully prospective existing-engine bridge. Without the latter, at most B; never tune on test.")
    write_json(PROTOCOL, protocol)
    report()
    print(json.dumps(clean(dict(selected=selected, validation=results[selected], bridge=bridge)),indent=2))


def frozen_engine_bridge(frame, state):
    import improve_projection as old
    import defensive_continuity as defense
    base = pd.read_csv(OUT / "defensive_continuity_v1_features.csv")
    base = base[base.season_end_year.between(2015,2019)].copy()
    frozen = joblib.load(old.BUNDLE)
    defensive = joblib.load(ROOT / "artifacts/simulation/defensive_continuity_v1.joblib")
    strength = base[["season_end_year", "team_id"]].copy()
    for target in ("ortg", "pace"):
        strength[target] = old.previous.predict(base, target, frozen["models"][target][frozen["selected_models"][target]])
    strength["drtg"] = defense.predict(base, defensive["model"])
    games = frame.copy()
    for side in ("home", "away"):
        renamed = strength.rename(columns={"team_id":side+"_team_id", **{k:side+"_engine_"+k for k in ("ortg","drtg","pace")}})
        games = games.merge(renamed, on=["season_end_year",side+"_team_id"], how="left", validate="many_to_one")
        for k in ("ortg","drtg","pace"):
            games[side+"_"+k] = games[side+"_engine_"+k]
    games = games.dropna(subset=["home_ortg","away_ortg"])
    return dict(n=len(games), metrics=metrics(games,projection(games,state)),
                identity_gamma=1., team_model_retrained=False, prospective=False,
                warning="Retrospective season membership; uncertainty/variance trained on different pregame-strength estimator; diagnostic only")


def final_stage():
    if LEDGER.exists():
        raise RuntimeError("Final game test already consumed; use --report")
    protocol = json.loads(PROTOCOL.read_text())
    checks = [fingerprint(OUTPUT)==protocol["dataset_sha256"], fingerprint(QUALITY)==protocol["quality_sha256"],
              fingerprint(MANIFEST)==protocol["source_manifest_sha256"], fingerprint(ARTIFACT)==protocol["artifact_sha256"],
              fingerprint(SUMMARY)==protocol["summary_sha256"], protected_hashes()==protocol["protected_hashes"]]
    checks += [fingerprint(Path(__file__).with_name(name))==value for name,value in protocol["source_hashes"].items()]
    if not all(checks):
        raise RuntimeError("Frozen game protocol/input mismatch")
    with LEDGER.open("x",encoding="utf-8") as stream:
        json.dump(dict(status="started", evaluation_count=1, protocol_sha256=fingerprint(PROTOCOL)),stream,indent=2)
    frame = pd.read_csv(OUTPUT,dtype={"game_id":str})
    frame = frame[frame.eligible & frame.home_ortg.notna() & ~frame.neutral_site & frame.season_end_year.between(2020,2025)].copy()
    state = joblib.load(ARTIFACT)
    pred = projection(frame,state)
    result = metrics(frame,pred)
    season_metrics = {str(year):metrics(frame.loc[mask],projection(frame.loc[mask],state))
                      for year in sorted(frame.season_end_year.unique()) for mask in [frame.season_end_year==year]}
    mean_margin = pred["means"]["home_points"]-pred["means"]["away_points"]
    mean_total = pred["means"]["home_points"]+pred["means"]["away_points"]
    groups = {"even_abs_expected_margin_le3":abs(mean_margin)<=3, "home_underdogs":mean_margin<0,
              "strong_mismatch_abs_expected_margin_ge10":abs(mean_margin)>=10,
              "high_pace_ge100":pred["means"]["pace"]>=100, "low_pace_le95":pred["means"]["pace"]<=95}
    subgroup_metrics = {name:metrics(frame.loc[mask],projection(frame.loc[mask],state)) for name,mask in groups.items() if mask.any()}
    # The same locked test evaluation includes sampled probabilities for every game.
    probabilities, ot_rates, mc_home, mc_away = [], [], [], []
    start = time.perf_counter()
    for index in range(len(frame)):
        if index%1000==0:
            print(f"Frozen MC evaluation: {index}/{len(frame)} games",flush=True)
        game = frame.iloc[index:index+1]
        simulation = simulate(game,state,seed=SEED+int(game.game_id.iloc[0]))
        probabilities.append(simulation["home_win_probability"])
        ot_rates.append(simulation["ot_rate"])
        mc_home.append(simulation["mean_home_points"])
        mc_away.append(simulation["mean_away_points"])
    mc_seconds = time.perf_counter()-start
    mc = metrics(frame,{**pred, "home":np.array(mc_home), "away":np.array(mc_away)},np.array(probabilities))
    examples = {}
    for name,index in [("close",np.argmin(abs(mean_margin))), ("strong_favorite",np.argmax(abs(mean_margin))),
                       ("high_scoring",np.argmax(mean_total)), ("low_scoring",np.argmin(mean_total))]:
        row = frame.iloc[index:index+1]
        sim = simulate(row,state,seed=SEED+int(row.game_id.iloc[0]))
        examples[name] = dict(game=row[["game_id","date","season","home_team_id","away_team_id","home_points","away_points"]].iloc[0].to_dict(),
                             simulation={k:v for k,v in sim.items() if k not in {"home_scores","away_scores"}})
    checks = gates(result)
    checks["monte_carlo_metrics"] = all(gates(mc).values())
    checks["season_stability"] = all(abs(v["margin_bias"])<=3. and v["ece"]<=.07 for v in season_metrics.values())
    checks["prospective_existing_engine_bridge"] = False
    verdict = "A. READY TO BUILD PRODUCTION SINGLE-GAME SIMULATOR" if all(checks.values()) else "B. GAME-LEVEL CALIBRATION NEEDS TARGETED FIX"
    if protected_hashes()!=protocol["protected_hashes"]:
        raise RuntimeError("Original research artifacts changed")
    summary = json.loads(SUMMARY.read_text())
    summary.update(status="complete_no_retuning", final_test=result, final_test_monte_carlo=mc,
        final_test_seasons=season_metrics, final_test_subgroups=subgroup_metrics, final_test_empirical=empirical_diagnostics(frame),
        test_monte_carlo_seconds=mc_seconds, observed_test_ot_rate=float((frame.overtime_periods>0).mean()),
        simulated_test_ot_rate=float(np.mean(ot_rates)), examples=examples, acceptance_checks=checks, verdict=verdict,
        old_team_test_untouched=True, protected_hashes_match=True,
        exact_next_step="Validate a timestamped pregame roster/strength bridge and its residual variance; resolve any failed era gates before production. No defensive retuning or UI yet.")
    write_json(SUMMARY,summary)
    output=frame[["game_id","season","date","home_team_id","away_team_id","home_points","away_points"]].copy()
    output["expected_home"],output["expected_away"] = pred["home"],pred["away"]
    output["analytic_home_win_probability"],output["monte_carlo_home_win_probability"] = pred["probability"], probabilities
    output.to_csv(OUT/"game_calibration_predictions.csv",index=False)
    write_json(LEDGER,dict(status="complete",evaluation_count=1,protocol_sha256=fingerprint(PROTOCOL),summary_sha256=fingerprint(SUMMARY),
                           previous_team_model_test_evaluations=0, new_game_test_evaluations=1))
    report()
    print(json.dumps(clean(dict(verdict=verdict,test=result,monte_carlo=mc,checks=checks)),indent=2))


def table(rows):
    if not rows:
        return "No supported rows."
    columns=list(dict.fromkeys(k for row in rows for k in row))
    def cell(value):
        if isinstance(value,float):return f"{value:.5f}"
        if isinstance(value,(dict,list)):return json.dumps(clean(value))
        return str(value)
    return "\n".join(["| "+" | ".join(columns)+" |","| "+" | ".join(["---"]*len(columns))+" |"]+
                      ["| "+" | ".join(cell(row.get(c,"")) for c in columns)+" |" for row in rows])


def plot_calibration(summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(6,5))
    for label,metric in [("Validation",summary["validation_candidates"][summary["selected"]]),("Frozen game test",summary.get("final_test"))]:
        if metric:
            bins=metric["calibration_bins"]
            ax.plot([b["predicted"] for b in bins],[b["actual"] for b in bins],marker="o",label=label)
    ax.plot([0,1],[0,1],"--",color="gray")
    ax.set(xlabel="Predicted home win probability",ylabel="Observed home win rate",xlim=(0,1),ylim=(0,1))
    ax.legend();fig.tight_layout()
    directory=REPORT.parent/"figures";directory.mkdir(exist_ok=True)
    fig.savefig(directory/"game_win_probability_calibration.png",dpi=150);plt.close(fig)


def report():
    s=json.loads(SUMMARY.read_text())
    q=json.loads(QUALITY.read_text())
    source=json.loads(MANIFEST.read_text())
    columns=pd.read_csv(OUTPUT,nrows=0).columns.tolist()
    lines=["# Game-Log Calibration for V1 Single-Game Simulation","",
        "Research only. No production API/UI, playoffs, gamecast, season simulator, or team-model retraining.","",
        "## Reproduction and Provenance","",
        "`python scripts/simulation/acquire_game_logs.py` downloads only Games.csv, TeamStatistics.csv and TeamStatisticsExtended.csv. "
        "`--offline` verifies cached hashes without network. `python scripts/simulation/build_game_logs.py` rebuilds canonical games offline. "
        "`python scripts/simulation/game_log_calibration.py --develop` fits TRAIN and selects on validation; tests then `--final` once. "
        "`--report` only renders saved results. The exclusive ledger blocks further final evaluation or development.","",
        f"Source: [{source['source']}]({source['source']}/versions/{source['version']}), Eoin A Moore, NBA.com upstream per author. "
        f"Version {source['version']}; {source['license']}; updated {source['source_updated']}. Anonymous public Kaggle file endpoints, "
        f"no token/login, HTML scraping, or restriction bypass. Cold pipeline requests: {source['initial_pipeline_requests']}; warm/offline:0. "
        "Exact URLs/bytes/SHA256 in `data/simulation/game_log_source_manifest.json`. Raw downloads and large canonical/prediction CSVs ignored.","",
        f"Coverage: {q['start']} through {q['end']}, {q['games']} canonical regular-season games. "
        f"Quarantined invalid records: {len(q['quarantined_games'])}; nonmatching canonical team-season schedule counts: {len(q['team_season_count_mismatches'])}. "
        "2025-26 excluded prospectively because the pinned June snapshot is not assumed complete. Cup games count only when the official ID is002; "
        "finals/exhibitions/playoffs excluded. Full per-season counts and missingness in `game_log_quality.json`.","",
        "## Canonical Schema and Identity","", "`"+", ".join(columns)+"`","",
        "Game IDs are source NBA IDs padded to10 digits, not synthesized IDs. Team/franchise IDs use the EXISTING historical identity functions, "
        "including separate Charlotte lineages, Nets, Seattle/OKC, Kings, Clippers and displaced Hornets. Game-index home/away IDs define joins; "
        "inconsistent box-row home flags are not trusted. Scores checked against corresponding team boxes; duplicate keys fail.","",
        "Actual possessions: mean of reported team possessions where present; else mean(FGA+0.44*FTA-ORB+TO). "
        "These fallback values are ESTIMATES, not reconstructed official counts. Source method is retained per side. Pace=possessions*48/game_minutes. "
        "Game minutes=numMinutes/5. Regulation points=sum(Q1..Q4), or final score only for confirmed48-minute games. "
        "Unknown/invalid periods remain missing/ineligible. OT possessions allocated by elapsed time, an explicit approximation. "
        "Outcome game ORtg columns are NEVER features; pregame home/away ORtg/DRtg/pace are separate fields. "
        "Missingness can restrict eligibility rather than fabricate historical fields.","",
        "## Leakage and Time Splits","", "`"+json.dumps(s['split'])+"`", "",
        "Pregame strength is regulation points per100 estimated regulation possessions, and pace. Each date's games are predicted BEFORE any "
        "results on that date update the accumulators. Fixed20-game previous-season franchise prior; expansion teams use prior league context. "
        "League environment uses only prior dates and a500-team-game prior-season prior. Every latest-information date is strictly earlier than game date. "
        "No full target-season team ratings, realized game pace, outcome, postgame Elo, or future roster outcomes enter model features. "
        "Actual pace is a TRAIN label/denominator for efficiency calibration only, never substituted for predicted pace at inference.","",
        f"Counts: `{json.dumps(s['counts'])}`. Six expanding OOF blocks estimate variance from genuine earlier-fit predictions, not fitted residuals. "
        "OOF warmup through1984-85. Point model refit on all TRAIN only. Calibration candidates fixed before the final game test. "
        "Previously recorded TEAM-model test is protected by hashes and not rerun; the new GAME test evaluates a different locked research layer.","",
        table(s['forward_residual_folds']),"",
        "## Home-Court and Variance Findings","",
        "TRAIN empirical blocks (final-score SD; adjusted effects use observed regulation pace for descriptive residual analysis, not prediction; "
        "effects are associations, not causal travel/home isolation):","",table(s['training_diagnostics']['A_formula']['empirical_variance_by_era']),"",
        "Validation empirical blocks:","",table(s['validation_empirical']),"",
        "Home effect is learned as a signed half-effect for each side, by broad TRAIN era, not hardcoded+3. "
        "The2010+ coefficients are estimated through2013-14 ONLY; no later home effect is silently refit. "
        "Era covariance is shrunk toward pooled OOF covariance using500 pseudo-games. Unseen recent years reuse the latest TRAIN block; "
        "validation chooses pooled versus era variance, not individual test-season coefficients.","",
        "## Expected Score and Candidates","",
        "A: efficiency=league+own_offense_above_league+opponent_defense_above_league+TRAIN common offset+signed home effect. "
        "Pace=mean(team paces). B: ridge30 learns the two efficiency weights, common offset, four signed home effects and a pace-deviation slope. "
        "Both predict efficiency before multiplying by expected possessions/100; they do NOT blindly average ORtg and opponent DRtg. "
        "No nonlinear family justified: retain the small4-candidate mean/variance comparison.","",
        table([dict(candidate=k,**v['equations']) for k,v in s['training_diagnostics'].items()]),"",
        table([dict(candidate=k,**{a:b for a,b in v.items() if a!='calibration_bins'}) for k,v in s['validation_candidates'].items()]),"",
        f"Selected `{s['selected']}`. Selection: lowest validation log loss among candidates within0.5 best margin MAE; "
        "prefer simpler within0.002 log loss and0.2 score MAE. All actual coefficients/variance arrays are in the summary and ignored artifact.","",
        "## Win Probability and Monte Carlo","",
        "Draw jointly Gaussian residuals for pace, home efficiency, away efficiency using genuine TRAIN-OOF covariance. "
        "This retains shared pace and score correlation; independent score noise is not assumed. Compute score as "
        "(mean_pace+pace_noise)*(mean_efficiency+efficiency_noise)/100 minus the covariance-product mean term. "
        "Round to nonnegative integer scores. Analytic margin variance includes the Gaussian product's second-order covariance. "
        "Analytic probability combines Gaussian regulation win mass, the fitted close-game tie atom and a Gaussian OT-margin approximation; "
        "analytic final means approximate expected overtime additions. "
        "Monte Carlo explicitly resolves ties; analytic and MC probabilities are evaluated separately, not fitted to test outcomes.","",
        "Default10000 draws: worst-case win-rate MC standard error0.5 percentage points. Practical minimum1000:1.58 points, "
        "not the default. Seed is explicit, local numpy Generator; game-test seed=20261015+NBA game ID. Same input/seed exactly reproducible. "
        "Counts benchmarked, not assumed:","",table(s['runtime_benchmarks']),"",
        "Tied rounded regulation scores get explicit five-minute overtime scoring increments. Joint paired OT residuals are bootstrapped "
        "from genuine TRAIN-OOF games with exactly one observed OT, centered before adding expected5/48 regulation scoring. "
        "TRAIN-OOF diagnostics showed Gaussian rounding underestimates regulation ties. One empirical Bernoulli parameter moves a fraction "
        "of1-3 point rounded margins onto a tie, approximately preserving total score; it is not uniform scoring noise or an extra strength adjustment. "
        "This simple endgame atom changes near-zero regulation margins slightly and is an explicit modeling approximation. "
        "Repeat OT until untied; after20 periods fail loudly, never silently return a tie or resolve with a coin flip. This is not a possession engine.","",
        "Selected TRAIN-only overtime fit: `"+json.dumps(s['selected_overtime'])+"`","",
        "## Existing Engine and Uncertainty","",
        "Frozen existing ORtg/pace artifacts and approved DRtg(identity gamma1) were loaded for an inference-only2014-15 through2018-19 "
        "bridge. No defensive tuning. Results below are diagnostic, not fully prospective: retrospectively known full-season roster membership "
        "does not establish what a pregame roster would have been. Primary probability/variance results use strictly prior-date box-score strengths.","",
        "`"+json.dumps(s['frozen_engine_validation_bridge'])+"`","",
        s['uncertainty'],"",
        "Statistical game-outcome intervals do not quantify unknown synthetic-roster interactions. Keep HIGH flags, warn and restrict claims; "
        "do not arbitrarily shift means or apply an unvalidated2x variance. Numerical epistemic propagation remains a targeted requirement "
        "for a prospective engine bridge, not hidden behind attractive winner accuracy.","",
        "## Limitations","",*['- '+x for x in s['limitations']],"",
        "## Final Game Test and Decision","",
        "Thresholds fixed before test: score MAE<=11; margin MAE<=13.5; total MAE<=17.5; Brier<=.23; log loss<=.66; "
        "ECE<=.04; regulation-margin80% coverage in[.75,.85],95% coverage in[.92,.98]; each season absolute margin bias<=3 and ECE<=.07. "
        "A additionally requires a fully prospective existing-engine bridge. These are research criteria, not the old defensive gates.",""
    ]
    if 'final_test' in s:
        lines += [table([dict(method='analytic',**{k:v for k,v in s['final_test'].items() if k!='calibration_bins'}),
                         dict(method='MC10000',**{k:v for k,v in s['final_test_monte_carlo'].items() if k!='calibration_bins'})]),"",
                  table([dict(season=k,**{a:b for a,b in v.items() if a!='calibration_bins'}) for k,v in s['final_test_seasons'].items()]),"",
                  table([dict(group=k,**{a:b for a,b in v.items() if a!='calibration_bins'}) for k,v in s['final_test_subgroups'].items()]),"",
                  "Elite/poor matchup proxy is abs(expected margin)>=10; it is not defined using final standings/outcomes.","",
                  table(s['final_test_empirical']),"",
                  f"Observed test OT rate={s['observed_test_ot_rate']:.5f}; MC rate={s['simulated_test_ot_rate']:.5f}. "
                  f"Entire test MC time={s['test_monte_carlo_seconds']:.2f}s. No retuning.","",
                  table(s['final_test']['calibration_bins']),"",
                  "![Win probability calibration](figures/game_win_probability_calibration.png)","",
                  "Representative games chosen by predicted matchup context, not whether the prediction was correct:","",
                  table([dict(example=k,**v['game'],**v['simulation']) for k,v in s['examples'].items()]),"",
                  "Acceptance: `"+json.dumps(s['acceptance_checks'])+"`","", "**"+s['verdict']+"**","",
                  "Existing engine bridge is not fully prospective; the game layer's numerical uncertainty is not automatically validated "
                  "for highly extrapolated synthetic rosters. A targeted integration/calibration fix, not further defensive tuning, is required.","",
                  s['exact_next_step'],"", "Original team research and historical inputs unchanged by hash. Final GAME test consumed once; "
                  "old TEAM test not reevaluated. No production API/UI changes, no merge or push.",""]
    else:
        lines += ["Final game test pending; parameters and protocol frozen.",""]
    REPORT.write_text("\n".join(lines),encoding="utf-8")
    plot_calibration(s)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    for name in ("develop","final","report"):
        group.add_argument("--"+name,action="store_true")
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        if args.develop:develop()
        elif args.final:final_stage()
        else:report()
