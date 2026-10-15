"""Final, training/validation-only game calibration; sealed2026 evaluated once.

Modes: --develop, --final, --report. Never reruns the old2020-2025 benchmark.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import gammaln
from threadpoolctl import threadpool_limits

import game_log_calibration as old
from acquire_game_logs import ROOT, fingerprint
from final_game_data import ACQUISITION, ROSTERS, CACHE, read_holdout_outcomes
from final_game_engine import OUTPUT as STRENGTHS, build_strengths, transport_games

OUT=ROOT/"data/simulation"
SUMMARY=OUT/"game_final_summary.json"
PROTOCOL=OUT/"game_final_protocol.json"
LEDGER=OUT/"game_final_holdout_run.json"
BUNDLE=ROOT/"artifacts/simulation/game_final_v1.joblib"
REPORT=ROOT/"reports/simulation/GAME_LOG_CALIBRATION.md"
HOME_MODELS=["trailing5","exponential5","era_shrunk"]
SAMPLERS=["A_pooled_gaussian","B_era_gaussian","C_recency_gaussian","D_bootstrap8","E_student_t"]
LEVELS=[.5,.8,.9,.95]
RESID=["z_pace","z_home","z_away"]
SEED=20261016


def protected_hashes():
    paths=list((ROOT/"data/historical/processed").glob("*.csv"))
    paths += [p for p in OUT.glob("*") if p.is_file() and not p.name.startswith("game_final_")]
    paths += [p for p in (ROOT/"scripts/simulation").glob("*.py") if p.name not in
              {"final_game_data.py","final_game_engine.py",Path(__file__).name,"test_final_game_calibration.py"}]
    paths += [p for p in (ROOT/"artifacts/simulation").glob("*.joblib") if p!=BUNDLE]
    paths += [ROOT/"reports/simulation/SIMULATION_ENGINE_DESIGN.md"]
    return {str(p.relative_to(ROOT)).replace("\\","/"):fingerprint(p) for p in paths}


def load_development():
    rows=[]
    for frame in pd.read_csv(old.OUTPUT,dtype={"game_id":str},chunksize=10000):
        frame=frame[(frame.season_end_year<=2019)&frame.eligible&frame.home_ortg.notna()&~frame.neutral_site]
        if len(frame):rows.append(frame)
    return pd.concat(rows,ignore_index=True)


def assert_history(frame,year):
    if len(frame)==0 or frame.season_end_year.max()>=year or frame.season_end_year.max()>2019:
        raise ValueError("Calibration must use earlier seasons through2019 only")


def base_means(frame):
    p=(frame.home_pace.to_numpy()+frame.away_pace.to_numpy())/2
    h=frame.home_ortg.to_numpy()+frame.away_drtg.to_numpy()-frame.league_ortg.to_numpy()
    a=frame.away_ortg.to_numpy()+frame.home_drtg.to_numpy()-frame.league_ortg.to_numpy()
    return p,h,a


def home_parameters(history,year,kind):
    prior=history[(history.season_end_year<year)&(history.season_end_year<=2019)]
    assert_history(prior,year)
    p,h,a=base_means(prior)
    home=prior.home_regulation_points.to_numpy()/p*100-h
    away=prior.away_regulation_points.to_numpy()/p*100-a
    common=float(np.mean((home+away)/2))
    delta=home-away
    latest=int(prior.season_end_year.max())
    if kind=="trailing5":
        weights=(prior.season_end_year.to_numpy()>=latest-4).astype(float)
        effect=float(np.average(delta,weights=weights))
    elif kind=="exponential5":
        weights=np.exp2(-(latest-prior.season_end_year.to_numpy())/5)
        effect=float(np.average(delta,weights=weights))
    elif kind=="era_shrunk":
        recent=prior.season_end_year.to_numpy()>=latest//10*10
        weight=float(recent.sum()/(recent.sum()+2000))
        effect=weight*float(delta[recent].mean())+(1-weight)*float(delta.mean())
    else:raise ValueError("Unknown home model")
    return dict(kind=kind,common_offset=common,home_margin_per100=effect,fitting_max_year=latest,
                target_year=int(year),n=len(prior))


def means(frame,home):
    p,h,a=base_means(frame)
    h=h+home["common_offset"]+home["home_margin_per100"]/2
    a=a+home["common_offset"]-home["home_margin_per100"]/2
    return dict(pace=p,home_eff=h,away_eff=a,home=p*h/100,away=p*a/100)


def residual_frame(frame,home_kind,home_history=None):
    if frame.season_end_year.max()>2019:raise ValueError("Holdout outcomes cannot enter calibration residuals")
    home_history=frame if home_history is None else home_history
    rows=[]
    for year,held in frame.groupby("season_end_year"):
        if year<1986:continue
        home=home_parameters(home_history,int(year),home_kind)
        m=means(held,home)
        row=held.copy()
        for k,v in m.items():row["expected_"+k]=v
        row["z_pace"]=(held.pace.to_numpy()-m["pace"])/held.league_pace.to_numpy()
        row["z_home"]=(held.home_regulation_points.to_numpy()/held.pace.to_numpy()*100-m["home_eff"])/held.league_ortg.to_numpy()
        row["z_away"]=(held.away_regulation_points.to_numpy()/held.pace.to_numpy()*100-m["away_eff"])/held.league_ortg.to_numpy()
        row["calibration_train_end"]=home["fitting_max_year"]
        rows.append(row)
    result=pd.concat(rows,ignore_index=True)
    if not (result.calibration_train_end<result.season_end_year).all():raise ValueError("Nonforward residual")
    return result


def weighted_covariance(x,w):
    w=np.asarray(w,float);w=w/w.sum()
    center=np.average(x,axis=0,weights=w)
    z=x-center
    covariance=(z*w[:,None]).T@z/(1-np.square(w).sum())
    if np.linalg.eigvalsh(covariance).min()<=0:raise ValueError("Unsupported residual covariance")
    return covariance,z,w


def fit_df(x,covariance,w):
    distance=np.einsum("ni,ij,nj->n",x,np.linalg.inv(covariance),x)
    def objective(df):
        likelihood=gammaln((df+3)/2)-gammaln(df/2)-1.5*np.log((df-2)*np.pi)-.5*np.linalg.slogdet(covariance)[1]-(df+3)/2*np.log1p(distance/(df-2))
        return -float(w@likelihood)
    fitted=minimize_scalar(objective,bounds=(5.,80.),method="bounded")
    return float(fitted.x)


def fit_sampler(history,year,kind):
    prior=history[(history.season_end_year<year)&(history.season_end_year<=2019)]
    assert_history(prior,year)
    latest=int(prior.season_end_year.max())
    if kind=="D_bootstrap8":prior=prior[prior.season_end_year>=latest-7]
    if len(prior)<500:raise ValueError("Insufficient residual support")
    x=prior[RESID].to_numpy()
    w=np.exp2(-(latest-prior.season_end_year.to_numpy())/5) if kind in {"C_recency_gaussian","E_student_t"} else np.ones(len(prior))
    covariance,centered,w=weighted_covariance(x,w)
    if kind=="B_era_gaussian":
        recent=prior.season_end_year.to_numpy()>=latest//10*10
        if recent.sum()>=100:
            recent_cov,_,_=weighted_covariance(x[recent],np.ones(recent.sum()))
            weight=recent.sum()/(recent.sum()+1000)
            covariance=weight*recent_cov+(1-weight)*covariance
    model=dict(kind=kind,covariance=covariance,cholesky=np.linalg.cholesky(covariance),
               fitting_max_year=latest,n=len(prior),effective_n=float(1/np.square(w).sum()),
               normalization="pace residual / prior-date league pace; efficiency residuals / prior-date league ORtg")
    if kind=="D_bootstrap8":model["samples"]=centered
    if kind=="E_student_t":model["df"]=fit_df(centered,covariance,w)
    return model


def student_evidence(residuals):
    train=residuals[residuals.season_end_year<=2010]
    held=residuals[residuals.season_end_year.between(2011,2014)]
    model=fit_sampler(train,2011,"E_student_t")
    x=held[RESID].to_numpy();c=model["covariance"]
    d=np.einsum("ni,ij,nj->n",x,np.linalg.inv(c),x)
    df=model["df"]
    t=gammaln((df+3)/2)-gammaln(df/2)-1.5*np.log((df-2)*np.pi)-.5*np.linalg.slogdet(c)[1]-(df+3)/2*np.log1p(d/(df-2))
    gaussian=-1.5*np.log(2*np.pi)-.5*np.linalg.slogdet(c)[1]-.5*d
    improvement=float(np.mean(t-gaussian))
    return dict(df=df,train_end=2010,evaluate="2011-2014 TRAIN block",log_likelihood_gain=improvement,
                justified=bool(df<60 and improvement>.005),rule="Shared-scale multivariate t only if TRAIN held-block likelihood gain>0.005 and df<60")


def draw_residuals(model,shape,rng):
    if model["kind"]=="D_bootstrap8":return model["samples"][rng.integers(0,len(model["samples"]),size=shape)]
    values=rng.standard_normal((*shape,3))@model["cholesky"].T
    if model["kind"]=="E_student_t":
        values*=np.sqrt((model["df"]-2)/rng.chisquare(model["df"],size=(*shape,1)))
    return values


def regulation(frame,m,model,count,rng):
    z=draw_residuals(model,(len(frame),count),rng)
    lp=frame.league_pace.to_numpy()[:,None]
    lo=frame.league_ortg.to_numpy()[:,None]
    p=np.maximum(1.,m["pace"][:,None]+z[:,:,0]*lp)
    h=p*(m["home_eff"][:,None]+z[:,:,1]*lo)/100-model["covariance"][0,1]*lp*lo/100
    a=p*(m["away_eff"][:,None]+z[:,:,2]*lo)/100-model["covariance"][0,2]*lp*lo/100
    return np.maximum(0,np.rint(h)).astype(int),np.maximum(0,np.rint(a)).astype(int),h-a,p.mean(axis=1)


def fit_ot(history,year,sampler,kind):
    prior=history[(history.season_end_year<year)&(history.season_end_year<=2019)]
    assert_history(prior,year)
    latest=int(prior.season_end_year.max())
    if kind=="trailing5":prior=prior[prior.season_end_year>=latest-4]
    w=np.exp2(-(latest-prior.season_end_year.to_numpy())/5) if kind=="exponential5" else np.ones(len(prior))
    w=w/w.sum()
    observed=float(w@(prior.overtime_periods.to_numpy()>0))
    rng=np.random.default_rng(1847)
    sample=prior.iloc[rng.choice(len(prior),size=200,replace=True,p=w)]
    m={k:sample["expected_"+k].to_numpy() for k in ["pace","home_eff","away_eff","home","away"]}
    h,a,_,_=regulation(sample,m,sampler,1000,rng)
    base=float(np.mean(h==a));near=float(np.mean((abs(h-a)>=1)&(abs(h-a)<=3)))
    extra=max(0.,(observed-base)/near) if near>0 else 0.
    thinning=max(0.,1-observed/base) if base>observed else 0.
    if extra>1:raise ValueError("Close-game tie correction unsupported")
    ot=prior[prior.overtime_periods==1]
    if len(ot)<50:raise ValueError("Insufficient observed one-OT games")
    scale=(ot.expected_pace*ot.league_ortg/100*5/48).to_numpy()
    residuals=np.column_stack([(ot[side+"_points"]-ot[side+"_regulation_points"]-ot["expected_"+side]*5/48).to_numpy()/scale for side in ["home","away"]])
    residuals-=residuals.mean(axis=0)
    return dict(kind=kind,samples=residuals,extra=extra,thinning=thinning,observed_rate=observed,
                natural_tie_rate=base,n=len(prior),one_ot_games=len(ot),fitting_max_year=latest)


def simulate(frame,home,sampler,ot,count=10000,seed=None,uncertainty="STANDARD"):
    if count<1000:raise ValueError("At least1000 simulations required")
    if uncertainty not in {"STANDARD","HIGH"}:raise ValueError("Unknown uncertainty label")
    rng=np.random.default_rng(seed)
    m=means(frame,home)
    results=[]
    for start in range(0,len(frame),64):
        piece=frame.iloc[start:start+64]
        local={k:v[start:start+64] for k,v in m.items()}
        hp,ap,raw_margin,pace=regulation(piece,local,sampler,count,rng)
        close=(abs(hp-ap)>=1)&(abs(hp-ap)<=3)
        pin=close&(rng.random(hp.shape)<ot["extra"])
        hp[pin]=np.rint((hp[pin]+ap[pin])/2).astype(int);ap[pin]=hp[pin]
        thin=(hp==ap)&(raw_margin!=0)&(rng.random(hp.shape)<ot["thinning"])
        hp[thin&(raw_margin>0)]+=1;ap[thin&(raw_margin<0)]+=1
        reg_margin=hp-ap
        tied=hp==ap
        ot_rate=tied.mean(axis=1)
        for _ in range(20):
            if not tied.any():break
            sample=ot["samples"][rng.integers(0,len(ot["samples"]),size=int(tied.sum()))]
            scale=np.repeat((local["pace"]*piece.league_ortg.to_numpy()/100*5/48)[:,None],count,axis=1)[tied]
            hmean=np.repeat((local["home"]*5/48)[:,None],count,axis=1)[tied]
            amean=np.repeat((local["away"]*5/48)[:,None],count,axis=1)[tied]
            hp[tied]+=np.maximum(0,np.rint(hmean+sample[:,0]*scale)).astype(int)
            ap[tied]+=np.maximum(0,np.rint(amean+sample[:,1]*scale)).astype(int)
            tied=hp==ap
        if tied.any():raise RuntimeError("Overtime failed to resolve, never silently return final ties")
        row=pd.DataFrame(dict(expected_home=hp.mean(axis=1),expected_away=ap.mean(axis=1),probability=(hp>ap).mean(axis=1),
                              simulated_ot=ot_rate,expected_possessions=pace,final_ties=0),index=piece.index)
        high=np.full(len(piece),uncertainty=="HIGH")
        for side in ["home","away"]:
            if side+"_engine_uncertainty" in piece:
                high|=piece[side+"_engine_uncertainty"].eq("HIGH").to_numpy()
        row["uncertainty_label"]=np.where(high,"HIGH","STANDARD")
        row["uncertainty_warning"]=np.where(high,"Unsupported custom/forecast domain; outcome intervals do not exhaust epistemic uncertainty","")
        for level in LEVELS:
            low,high=(1-level)/2,(1+level)/2
            row[f"lo{int(level*100)}"],row[f"hi{int(level*100)}"]=np.quantile(hp-ap,[low,high],axis=1)
            row[f"reg_lo{int(level*100)}"],row[f"reg_hi{int(level*100)}"]=np.quantile(reg_margin,[low,high],axis=1)
        results.append(row)
    return pd.concat(results).reindex(frame.index)


def metrics(frame,pred):
    y=frame.home_win.to_numpy();p=np.clip(pred.probability.to_numpy(),1e-9,1-1e-9)
    h,a=pred.expected_home.to_numpy(),pred.expected_away.to_numpy()
    bins=old.calibration_bins(y,p)
    result=dict(n=len(frame),score_mae=float(np.mean(np.r_[abs(h-frame.home_points),abs(a-frame.away_points)])),
        margin_mae=float(np.mean(abs(h-a-frame.margin))),total_mae=float(np.mean(abs(h+a-frame.home_points-frame.away_points))),
        brier=float(np.mean((p-y)**2)),log_loss=float(np.mean(-y*np.log(p)-(1-y)*np.log(1-p))),
        ece=sum(b["n"]*abs(b["predicted"]-b["actual"]) for b in bins)/len(frame),accuracy=float(np.mean((p>=.5)==y)),
        observed_ot=float((frame.overtime_periods>0).mean()),simulated_ot=float(pred.simulated_ot.mean()),
        observed_home_margin=float(frame.margin.mean()),expected_home_margin=float(np.mean(h-a)),
        home_margin_bias=float(np.mean(h-a-frame.margin)),calibration_bins=bins)
    for level in LEVELS:
        key=str(int(level*100))
        result["coverage"+key]=float(((frame.margin.to_numpy()>=pred["lo"+key])&(frame.margin.to_numpy()<=pred["hi"+key])).mean())
        result["regulation_coverage"+key]=float(((frame.regulation_margin.to_numpy()>=pred["reg_lo"+key])&(frame.regulation_margin.to_numpy()<=pred["reg_hi"+key])).mean())
    return result


def evaluate_years(frame,history,home_history,home_kind,sampler_kind,ot_kind,count=5000):
    predictions=[];parameters=[]
    for year,held in frame.groupby("season_end_year"):
        home=home_parameters(home_history,int(year),home_kind)
        sampler=fit_sampler(history,int(year),sampler_kind)
        ot=fit_ot(history,int(year),sampler,ot_kind)
        predictions.append(simulate(held,home,sampler,ot,count,SEED+int(year)))
        parameters.append(dict(year=int(year),home=home,sampler={k:v for k,v in sampler.items() if k not in {"samples","cholesky","covariance"}},
                               overtime={k:v for k,v in ot.items() if k!="samples"}))
    pred=pd.concat(predictions).reindex(frame.index)
    by_year={str(y):metrics(g,pred.loc[g.index]) for y,g in frame.groupby("season_end_year")}
    return metrics(frame,pred),by_year,pred,parameters


def develop():
    if LEDGER.exists():raise RuntimeError("2026 holdout already consumed; no further development")
    if not ACQUISITION.exists() or not STRENGTHS.exists():raise RuntimeError("Run blind acquisition and frozen roster bridge first")
    protected=protected_hashes()
    frame=load_development()
    validation=frame[frame.season_end_year.between(2015,2019)]
    home_scores={}
    for kind in HOME_MODELS:
        errors=[]
        parameters=[]
        for year,held in validation.groupby("season_end_year"):
            h=home_parameters(frame,int(year),kind);m=means(held,h)
            error=m["home"]-m["away"]-held.regulation_margin.to_numpy()
            errors.append(dict(year=int(year),margin_mae=float(np.abs(error).mean()),bias=float(error.mean())))
            parameters.append(h)
        home_scores[kind]=dict(mean_abs_season_bias=float(np.mean([abs(e["bias"]) for e in errors])),
                               margin_mae=float(np.mean([e["margin_mae"] for e in errors])),seasons=errors,parameters=parameters)
    best=min(HOME_MODELS,key=lambda k:home_scores[k]["mean_abs_season_bias"]+.1*home_scores[k]["margin_mae"])
    home_choice=next(k for k in HOME_MODELS if home_scores[k]["mean_abs_season_bias"]+.1*home_scores[k]["margin_mae"]<=
                     home_scores[best]["mean_abs_season_bias"]+.1*home_scores[best]["margin_mae"]+.05)
    residuals=residual_frame(frame,home_choice)
    home_stability=[]
    for block,group in residuals.groupby(residuals.season_end_year//10*10):
        p,bh,ba=base_means(group)
        h=group.expected_home_eff.to_numpy()-bh
        a=group.expected_away_eff.to_numpy()-ba
        home_stability.append(dict(block=int(block),n=len(group),expected_home_scoring_effect=float(np.mean(p*h/100)),
            expected_away_scoring_effect=float(np.mean(p*a/100)),expected_home_margin_effect=float(np.mean(p*(h-a)/100)),
            observed_home_margin=float(group.margin.mean()),mean_predicted_margin=float((group.expected_home-group.expected_away).mean())))
    evidence=student_evidence(residuals)
    kinds=SAMPLERS if evidence["justified"] else SAMPLERS[:-1]
    candidates={}
    for kind in kinds:
        print("Prospective validation variance: "+kind,flush=True)
        metric,seasons,_,params=evaluate_years(validation,residuals,frame,home_choice,kind,"trailing5")
        candidates[kind]=dict(metrics=metric,seasons=seasons,parameters=params)
    def interval_priority(name):
        rows=list(candidates[name]["seasons"].values())
        return (sum(max(0.,.92-r["coverage95"]) for r in rows),
                float(np.mean([abs(r["coverage95"]-.95) for r in rows])),
                float(np.mean([abs(r["coverage"+str(int(level*100))]-level) for r in rows for level in LEVELS])),
                max(r["log_loss"] for r in rows),SAMPLERS.index(name))
    selected=min(kinds,key=interval_priority)
    overtime={}
    for kind in ["trailing5","exponential5"]:
        metric,seasons,pred,params=evaluate_years(validation,residuals,frame,home_choice,selected,kind)
        overtime[kind]=dict(metrics=metric,seasons=seasons,parameters=params,
                           mean_abs_season_ot_error=float(np.mean([abs(r["observed_ot"]-r["simulated_ot"]) for r in seasons.values()])))
    ot_choice=min(overtime,key=lambda k:overtime[k]["mean_abs_season_ot_error"])
    strength=pd.read_csv(STRENGTHS)
    engine_frame=transport_games(frame[frame.season_end_year.between(2010,2019)],strength)
    engine_residuals=residual_frame(engine_frame,home_choice,frame)
    engine_history=engine_residuals[engine_residuals.season_end_year>=2015]
    engine_validation=engine_frame[engine_frame.season_end_year.between(2018,2019)]
    em,es,_,ep=evaluate_years(engine_validation,engine_history,frame,home_choice,selected,ot_choice)
    # Display unadapted accumulated error too, not just the engine-specific residual fit.
    um,us,_,_=evaluate_years(engine_validation,residuals,frame,home_choice,selected,ot_choice)
    h=home_parameters(frame,2026,home_choice)
    rolling_sampler=fit_sampler(residuals,2026,selected)
    engine_sampler=fit_sampler(engine_history,2026,selected)
    rolling_ot=fit_ot(residuals,2026,rolling_sampler,ot_choice)
    engine_ot=fit_ot(engine_history,2026,engine_sampler,ot_choice)
    bundle=dict(home=h,home_choice=home_choice,sampler_choice=selected,ot_choice=ot_choice,
                rolling_sampler=rolling_sampler,engine_sampler=engine_sampler,rolling_ot=rolling_ot,engine_ot=engine_ot)
    bench=[]
    matchup=validation.iloc[:1]
    for count in [1000,5000,10000,25000]:
        durations=[];probabilities=[];points=[]
        for seed in range(5):
            start=time.perf_counter();s=simulate(matchup,h,rolling_sampler,rolling_ot,count,SEED+seed)
            durations.append(time.perf_counter()-start);probabilities.append(float(s.probability.iloc[0]));points.append(float(s.expected_home.iloc[0]))
        bench.append(dict(count=count,seconds_median=float(np.median(durations)),
                          worst_case_probability_mc_se=.5/np.sqrt(count),win_probability_seed_sd=float(np.std(probabilities,ddof=1)),
                          home_mean_seed_sd=float(np.std(points,ddof=1))))
    prefix=REPORT.read_text(encoding="utf-8").split("\n## Modern Variance Calibration")[0]
    summary=dict(status="frozen_before2026",holdout_outcomes_accessed=False,source=json.loads(ACQUISITION.read_text()),
        splits=dict(train="usable1982-83 through2013-14; forward residuals1985-86 onward",validation="2014-15 through2018-19, prequential previous-season updates",
                    old_test="2019-20 through2024-25 preserved and excluded from every calibration fit",new_holdout="2025-26,1230 regular-season games"),
        home_candidates=home_scores,selected_home=home_choice,student_evidence=evidence,variance_candidates=candidates,selected_sampler=selected,
        home_stability=home_stability,
        ot_candidates=overtime,selected_ot=ot_choice,
        end_to_end_validation=dict(period="2017-18 through2018-19",calibration="2014-15 through seasons strictly before each evaluated season",
            metrics=em,seasons=es,parameters=ep,unadapted_rolling_residual_metrics=um,
            engine_specific_residuals="Covariance/empirical noise learned from actual roster-engine errors in earlier validation seasons; not an arbitrary HIGH-flag multiplier",
            team_engine_retrained=False,opening_roster_proxy=True),
        final_parameters=dict(home=h,rolling_sampler={k:v for k,v in rolling_sampler.items() if k not in {"samples","cholesky"}},
                              engine_sampler={k:v for k,v in engine_sampler.items() if k not in {"samples","cholesky"}},
                              rolling_ot={k:v for k,v in rolling_ot.items() if k!="samples"},engine_ot={k:v for k,v in engine_ot.items() if k!="samples"}),
        benchmarks=bench,default_simulations=10000,minimum_simulations=1000,protected_hashes=protected,
        original_report_prefix_sha256=__import__("hashlib").sha256(prefix.encode()).hexdigest(),
        limitations=["First-game identity listings are an opening-roster proxy, potentially affected by who was listed/available; not a certified preseason transaction roster",
                     "All calibration ends2018-19; no refresh using previously evaluated2020-2025 outcomes, leaving a six-season calibration gap",
                     "2026 is untouched at game-level calibration only; upstream team research previously saw its aggregate evaluation",
                     "Neutral venues other than known Orlando restart not comprehensively identified",
                     "Historical possession estimates and time-proportional OT possessions are approximate",
                     "Outcome intervals model game noise and observed engine forecast errors, not every custom-team extrapolation uncertainty"])
    if protected_hashes()!=protected:raise RuntimeError("Protected original research changed")
    joblib.dump(bundle,BUNDLE)
    old.write_json(SUMMARY,summary)
    sources=[Path(__file__),Path(__file__).with_name("final_game_data.py"),Path(__file__).with_name("final_game_engine.py")]
    protocol=dict(status="frozen_before2026",source_hashes={p.name:fingerprint(p) for p in sources},
        input_hashes={str(p.relative_to(ROOT)).replace("\\","/"):fingerprint(p) for p in [ACQUISITION,ROSTERS,STRENGTHS,old.OUTPUT]},
        acquisition_raw_hashes=summary["source"]["raw_hashes"],protected_hashes=protected,summary_sha256=fingerprint(SUMMARY),artifact_sha256=fingerprint(BUNDLE),
        calibration_max_year=2019,new_holdout_year=2026,holdout_policy="One selected rolling model plus separate selected roster-engine track in a single locked evaluation; no retuning",
        interval_target95=[.92,.98],
        decision_rules=dict(A="Both modes satisfy95%coverage92-98%,ECE<=.04,logloss<=.66,Brier<=.23,scoreMAE<=11,marginMAE<=13.5,totalMAE<=17.5,OTgap<=.015,homebias<=2,certified preseason rosters",
                            B="Both modes satisfy95%coverage92-98%,ECE<=.06,logloss<=.69,Brier<=.245,scoreMAE<=12,marginMAE<=14.5,totalMAE<=19,OTgap<=.025,homebias<=3; roster-engine Brier improves home-only baseline by>=.005; documented proxy limitations",
                            C="Otherwise; no additional optimization pass"),
        counts=[1000,5000,10000,25000],random_seed_optional=True,original_report_prefix_sha256=summary["original_report_prefix_sha256"])
    old.write_json(PROTOCOL,protocol)
    render_report()
    print(json.dumps(old.clean(dict(selected_home=home_choice,selected_sampler=selected,selected_ot=ot_choice,
                                    validation=candidates[selected]["metrics"],end_to_end=em,student=evidence)),indent=2))


def decision(rolling,engine,baseline):
    def acceptable(m,strict):
        return bool(.92<=m["coverage95"]<=.98 and m["ece"]<=(.04 if strict else .06) and
                    m["log_loss"]<=(.66 if strict else .69) and m["brier"]<=(.23 if strict else .245) and
                    m["score_mae"]<=(11 if strict else 12) and m["margin_mae"]<=(13.5 if strict else 14.5) and
                    m["total_mae"]<=(17.5 if strict else 19) and abs(m["observed_ot"]-m["simulated_ot"])<=(.015 if strict else .025) and
                    abs(m["home_margin_bias"])<=(2 if strict else 3))
    checks=dict(rolling_acceptable=acceptable(rolling,False),roster_engine_acceptable=acceptable(engine,False),
                roster_engine_useful=engine["brier"]<=baseline-.005,certified_preseason_rosters=False,
                unchanged_defensive_model=True,model_choices_frozen_before_new_holdout=True)
    if acceptable(rolling,True) and acceptable(engine,True) and checks["certified_preseason_rosters"]:
        verdict="A. READY TO BUILD PRODUCTION SINGLE-GAME SIMULATOR"
    elif all(checks[k] for k in ["rolling_acceptable","roster_engine_acceptable","roster_engine_useful"]):
        verdict="B. ACCEPTABLE FOR UNCERTAINTY-AWARE V1 WITH DOCUMENTED LIMITATIONS"
    else:verdict="C. GAME SIMULATOR NOT RELIABLE ENOUGH"
    return verdict,checks


def final_stage():
    if LEDGER.exists():raise RuntimeError("2026 evaluation already consumed; use --report only")
    p=json.loads(PROTOCOL.read_text())
    checks=[protected_hashes()==p["protected_hashes"],fingerprint(SUMMARY)==p["summary_sha256"],fingerprint(BUNDLE)==p["artifact_sha256"]]
    checks += [fingerprint(Path(__file__).with_name(name))==value for name,value in p["source_hashes"].items()]
    checks += [fingerprint(ROOT/path)==value for path,value in {**p["input_hashes"],**p["acquisition_raw_hashes"]}.items()]
    if not all(checks):raise RuntimeError("Frozen inputs, research, source or parameter hashes changed")
    with LEDGER.open("x",encoding="utf-8") as stream:
        json.dump(dict(status="started",evaluation_count=1,protocol_sha256=fingerprint(PROTOCOL)),stream,indent=2)
    held=read_holdout_outcomes(protocol_verified=True)
    bundle=joblib.load(BUNDLE)
    engine_games=transport_games(held,pd.read_csv(STRENGTHS))
    result={};predictions={}
    for mode,frame in [("rolling",held),("roster_engine",engine_games)]:
        key="engine" if mode=="roster_engine" else "rolling"
        start=time.perf_counter()
        pred=simulate(frame,bundle["home"],bundle[key+"_sampler"],bundle[key+"_ot"],10000,SEED+2026)
        result[mode]=dict(metrics=metrics(frame,pred),seconds=time.perf_counter()-start)
        predictions[mode]=pd.concat([frame[["game_id","date","home_team_id","away_team_id","home_points","away_points"]],pred],axis=1)
        predictions[mode].to_csv(CACHE/(mode+"_2026_predictions.csv"),index=False)
    validation=load_development();validation=validation[validation.season_end_year>=2015]
    home_rate=float(validation.home_win.mean())
    baseline=float(np.mean((home_rate-held.home_win.to_numpy())**2))
    verdict,checks=decision(result["rolling"]["metrics"],result["roster_engine"]["metrics"],baseline)
    s=json.loads(SUMMARY.read_text())
    s.update(status="complete_no_retuning",new_holdout=result,home_only_baseline_brier=baseline,acceptance_checks=checks,verdict=verdict,
        original_test_untouched=True,original_team_test_not_repeated=True,protected_hashes_match=protected_hashes()==p["protected_hashes"],
        exact_next_step="Freeze this final V1 decision. For A/B, production single-game implementation may proceed in a separate task with uncertainty labels and documented roster assumptions; for C, reconsider the approach, not another tuning loop.")
    if not s["protected_hashes_match"]:raise RuntimeError("Original evaluation changed")
    old.write_json(SUMMARY,s)
    old.write_json(LEDGER,dict(status="complete",evaluation_count=1,protocol_sha256=fingerprint(PROTOCOL),summary_sha256=fingerprint(SUMMARY),
                              holdout_year=2026,old_game_test_evaluations=0,old_team_test_evaluations=0,tracks=["rolling","roster_engine"]))
    render_report()
    print(json.dumps(old.clean(dict(verdict=verdict,results=result,checks=checks)),indent=2))


def compact(metric):
    return {k:v for k,v in metric.items() if k!="calibration_bins"}


def render_report():
    s=json.loads(SUMMARY.read_text())
    table=old.table
    prefix=REPORT.read_text(encoding="utf-8").split("\n## Modern Variance Calibration")[0]
    if __import__("hashlib").sha256(prefix.encode()).hexdigest()!=s["original_report_prefix_sha256"]:
        raise RuntimeError("Earlier report results changed")
    lines=["\n## Modern Variance Calibration","",
        "Final pass. Earlier2020-2025 results above are preserved, NOT rerun or used for fitting/selection. "
        "Train/validation only through2018-19;2025-26 is new at the GAME layer, not virgin to all upstream season-level research.","",
        "Commands: `python scripts/simulation/final_game_data.py --acquire`; `python scripts/simulation/final_game_engine.py`; "
        "`python scripts/simulation/finalize_game_calibration.py --develop`; tests; `--final` ONCE; `--report` only reads saved results. "
        "An exclusive ledger plus source/input/artifact hashes protects the holdout. No defensive team model retraining or UI/API changes.","",
        "`"+json.dumps(s["splits"])+"`","",
        "All five residual families use joint pace/home-efficiency/away-efficiency vectors from strictly earlier-season mean predictions. "
        "Residuals are normalized by pregame league pace/ORtg, so the same dimensionless noise scales into the target era's scoring environment. "
        "A pooled Gaussian; B latest available decade covariance shrunk with1000 pseudo-games toward pooled; C exponential recency half-life5 seasons; "
        "D empirical paired8-season residual bootstrap; E one shared-scale multivariate t, df fitted in[5,80], if TRAIN held-block likelihood justifies it. "
        "No independent score noise, arbitrary +/- points, or validation-chosen variance multipliers.","",
        "Student-t evidence: `"+json.dumps(s["student_evidence"])+"`","",
        "Every validation season N uses only residuals from seasons<N, including already completed earlier validation seasons. "
        "Selection prioritizes summed95% undercoverage below92%, then absolute95% calibration error, all50/80/90/95 coverage error, "
        "worst-season log loss, interpretability. The frozen source implements this exact ordering; MAE is not the primary selector.","",
        table([dict(candidate=k,**compact(v["metrics"])) for k,v in s["variance_candidates"].items()]),"",
        "Selected **"+s["selected_sampler"]+"**. Interval coverage below is measured on SAMPLED FINAL margins, not a Gaussian proxy; "
        "regulation coverage is also reported separately. Inclusive integer intervals can conservatively exceed their nominal levels.","",
        table([dict(candidate=k,season=y,**compact(v)) for k,r in s["variance_candidates"].items() for y,v in r["seasons"].items()]),"",
        "## Prospective Home Court Model","",
        "Transparent efficiency remains league+own offensive deviation+opponent defensive deviation+common offset+signed home half-effect. "
        "Expected pace is the mean of the two projected paces. Common offset and home effect for season N use only completed seasons<N. "
        "Compare trailing5, exponential half-life5, and current available decade shrunk toward the long run with2000 pseudo-games. "
        "Choose mean absolute seasonal margin bias plus0.1*margin MAE; prefer the first simpler candidate within0.05. No hardcoded+3.","",
        table([dict(candidate=k,mean_abs_season_bias=v["mean_abs_season_bias"],margin_mae=v["margin_mae"]) for k,v in s["home_candidates"].items()]),"",
        table([dict(candidate=k,**p) for k,v in s["home_candidates"].items() for p in v["parameters"]]),"",
        "Selected **"+s["selected_home"]+"**. For target2026 the parameters are frozen from history ending2019: `"+
        json.dumps(s["final_parameters"]["home"])+"`. Home scoring effect per100 is common_offset+H/2, away scoring effect is common_offset-H/2; "
        "home margin effect is H. This six-season refresh gap is explicit, not silently filled with old test outcomes.","",
        "Prospective home-effect stability (points; expected effects are relative to the no-home league-relative baseline):","",
        table(s["home_stability"]),"",
        "## Overtime Calibration","",
        "The prior constant tie atom pooled older, higher OT rates, while newer margins spread out. The new prospective trailing5/EW5 rate "
        "uses seasons before N only. Estimate the sampler's natural rounded ties and1-3 point mass on200 historical matchups x1000 draws. "
        "Fit one tie atom, or thin surplus rounded ties using the sign of the underlying unrounded regulation margin. It models endgame "
        "discreteness; it is not arbitrary uniform score noise. Remaining regulation ties receive empirical paired five-minute OT increments, "
        "scaled to target scoring environment. Repeat until untied; fail after20 periods rather than return ties or a winner coin flip.","",
        table([dict(candidate=k,mean_abs_season_ot_error=v["mean_abs_season_ot_error"],**compact(v["metrics"])) for k,v in s["ot_candidates"].items()]),"",
        "Selected **"+s["selected_ot"]+"**. Rolling/engine OT parameters: `"+json.dumps({k:v for k,v in s["final_parameters"].items() if k.endswith("_ot")})+"`","",
        "## End-to-End Roster Engine Evaluation","",
        "The official NBA public live-box endpoints returned403; no retry or bypass. Public Kaggle version515 PlayerStatistics.csv was cached "
        "and streamed with ONLY game ID, team ID, NBA person ID and names selected. All listed identities in each team's first regular-season game "
        "form the opening-roster approximation. No target-season minutes, starts, scoring, or late-season roster membership determines features. "
        "DNP identities are retained when the publisher lists them. This proxy can still reflect appearance/availability selection; it is not a "
        "certified preseason transaction roster and cannot establish exact injury availability.","",
        "Names resolve to existing canonical player IDs using PRIOR-season profiles only; unknown/ambiguous entrants receive the existing frozen "
        "pooled priors. Use the fixed lagged profile builder, fixed240-minute rotation/usage, prior franchise continuity and original ORtg/pace/DRtg "
        "artifacts. DRtg gamma remains1. No target-season realized team ratings enter. Forecasts are transported additively from their source "
        "league anchor to the pregame target league anchor; offense, defense and pace deviations are preserved.","",
        "Engine-specific residuals are learned from actual roster-engine errors in2014-15 onward, not from HIGH labels. "
        "Separate forward validation2017-18/2018-19 uses only earlier seasons' engine residuals. This empirical input-mode covariance adaptation "
        "is the only numerical forecast-error propagation; means and team strengths are unchanged.","",
        table([dict(mode="engine-specific residuals",**compact(s["end_to_end_validation"]["metrics"])),
               dict(mode="unadapted rolling residuals",**compact(s["end_to_end_validation"]["unadapted_rolling_residual_metrics"]))]),"",
        table([dict(season=y,**compact(v)) for y,v in s["end_to_end_validation"]["seasons"].items()]),"",
        "## 2025-26 Untouched Game-Level Holdout","",
        "Source: [public pinned NBA dataset]("+s["source"]["source"]+"). Blind index verification:1230 unique regular-season IDs,30 teams x82 games, "
        "2460 required team-box keys. NBA Cup games counted only under regular-season IDs. Source game index defines home/away; extra raw box keys "
        "are ignored rather than trusting row flags. Acquisition loaded no score/winner/performance columns. Source SHA256 and identity rosters "
        "were fixed before model selection. Prior team research saw2026 season aggregates; this holdout is untouched specifically at game level.","",
        "The complete2026 game outcomes are read only AFTER frozen protocol verification and exclusive ledger creation. All1230 games must reconcile "
        "against team boxes with usable regulation/duration/possession data; otherwise stop, not substitute a pseudo-holdout.","",
        "95% final-margin coverage target fixed at92-98%. Other A/B/C criteria, including ECE/log loss/Brier/home/OT usefulness, are recorded "
        "before evaluation in `game_final_protocol.json`; no criteria change after seeing2026.","",
        "## Final V1 Simulation Protocol","",
        "1. Select target-era league ORtg/pace known before the matchup.\n"
        "2. Transport roster strength deviations into that environment.\n"
        "3. Apply the frozen prospective home effect and transparent score formula.\n"
        "4. Select calibrated rolling or roster-engine joint residual distribution.\n"
        "5. Jointly draw pace and both efficiencies; center the covariance-product mean term.\n"
        "6. Produce nonnegative integer regulation scores and the calibrated discrete tie mass.\n"
        "7. Resolve every remaining tie with explicit empirical overtime.\n"
        "8. Repeat, returning final-score expectations, win probability,50/80/90/95 intervals and diagnostics.","",
        table(s["benchmarks"]),"",
        "Default10000, practical minimum1000. Worst-case probability MC SE at10000 is0.5 percentage points. "
        "`seed=None` gives random runs; an explicit seed with the same inputs gives identical draws. HIGH extrapolation labels do not change "
        "mean strengths or trigger an invented variance multiplier. Engine-mode residual noise is empirically fitted independently of HIGH. "
        "Outcome intervals still do not exhaust unknown synthetic-roster interaction uncertainty.","",
        *["- "+v for v in s["limitations"]],""
    ]
    if "new_holdout" in s:
        lines += ["### Frozen2026 Results","",table([dict(mode=k,**compact(v["metrics"])) for k,v in s["new_holdout"].items()]),"",
                  "Calibration bins:","",table([dict(mode=k,**b) for k,v in s["new_holdout"].items() for b in v["metrics"]["calibration_bins"]]),"",
                  "Acceptance: `"+json.dumps(s["acceptance_checks"])+"`","","**"+s["verdict"]+"**","",s["exact_next_step"],"",
                  "One new2026 evaluation with two predefined tracks; no old game/team benchmark rerun, no retuning, no defense changes, no UI/API changes. "
                  "All prior results above remain preserved. No merge or push.",""]
    else:lines += ["New holdout pending; all choices/parameters frozen from TRAIN/VALIDATION only.",""]
    if "new_holdout" in s:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(11,4.5))
        for mode,row in s["new_holdout"].items():
            bins=row["metrics"]["calibration_bins"]
            axes[0].plot([b["predicted"] for b in bins],[b["actual"] for b in bins],marker="o",label=mode)
            axes[1].plot(LEVELS,[row["metrics"]["coverage"+str(int(n*100))] for n in LEVELS],marker="o",label=mode)
        for ax in axes:
            ax.plot([0,1],[0,1],"--",color="gray");ax.legend();ax.set_xlim(0,1);ax.set_ylim(0,1)
        axes[0].set(xlabel="Predicted home win probability",ylabel="Observed home win rate")
        axes[1].set(xlabel="Nominal final-margin coverage",ylabel="Observed coverage")
        fig.tight_layout()
        directory=REPORT.parent/"figures";directory.mkdir(exist_ok=True)
        fig.savefig(directory/"game_final_calibration.png",dpi=150);plt.close(fig)
        lines += ["![Final2026 calibration](figures/game_final_calibration.png)",""]
    REPORT.write_text(prefix+"\n".join(lines),encoding="utf-8")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    for mode in ["develop","final","report"]:group.add_argument("--"+mode,action="store_true")
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        if args.develop:develop()
        elif args.final:final_stage()
        else:render_report()
