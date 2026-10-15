"""Inference-only opening-game roster bridge into the frozen HoopsLab engine."""
from __future__ import annotations

import re
import unicodedata

import joblib
import numpy as np
import pandas as pd

import improve_projection as engine
import defensive_continuity as defense
from acquire_game_logs import ROOT
from final_game_data import ROSTERS

OUTPUT=ROOT/"data/simulation/game_final_roster_strengths.csv"


def name_key(name):
    value=unicodedata.normalize("NFKD",str(name)).encode("ascii","ignore").decode().lower()
    return re.sub(r"[^a-z0-9]","",value)


def load_context():
    base=ROOT/"data/historical/processed"
    players=pd.read_csv(base/"player_seasons.csv")
    players=players[players.season_end_year<=2025].copy()
    splits=pd.read_csv(base/"player_team_splits.csv")
    splits=splits[splits.season_end_year<=2025]
    teams=pd.read_csv(base/"team_seasons.csv")
    teams=teams[teams.season_end_year<=2025]
    leagues=pd.read_csv(base/"league_seasons.csv")
    leagues=leagues[leagues.season_end_year<=2025].set_index("season_end_year")
    supplemental=pd.read_csv(ROOT/"data/analysis/simulation_defensive_source_fields.csv")
    supplemental=supplemental[supplemental.season_end_year<=2025]
    profiles=engine.previous.add_derived_player_features(players).merge(supplemental,
        on=["player_id","season_end_year","team"],how="left",validate="one_to_one")
    profiles=profiles[profiles.source_minutes.notna() & ((profiles.source_minutes-profiles.minutes).abs()<=1.)]
    fields=engine.previous.PROFILE_FIELDS+["team_pace","league_pace"]
    training=profiles[(profiles.season_end_year<=2013)&(profiles.minutes>=100)]
    defaults=training[fields].median()
    roles={role:g[fields].median().fillna(defaults) for role,g in training.groupby("position")}
    histories={pid:g.sort_values("season_end_year") for pid,g in profiles.groupby("player_id")}
    roster=engine.previous.roster_rows(players,splits)
    env=teams[["season_end_year","team_id","franchise_id","def_rtg"]].copy()
    env["relative_defense"]=env.def_rtg-env.season_end_year.map(leagues.league_points_per_75)/75*100
    joined=roster.merge(env[["season_end_year","team_id","relative_defense"]],on=["season_end_year","team_id"],validate="many_to_one")
    joined["weighted"]=joined.relative_defense*joined.minutes
    summed=joined.groupby(["season_end_year","player_id"])[["weighted","minutes"]].sum()
    environments=summed.weighted.div(summed.minutes).to_dict()
    membership=roster.groupby(["season_end_year","player_id"]).franchise_id.agg(set).to_dict()
    origins=roster.sort_values("minutes",ascending=False).drop_duplicates(["season_end_year","player_id"]).set_index(["season_end_year","player_id"]).franchise_id.to_dict()
    schedules=teams.assign(games=teams.wins+teams.losses).groupby("season_end_year").games.max().to_dict()
    return dict(histories=histories,defaults=defaults,roles=roles,environments=environments,schedules=schedules,
                membership=membership,origins=origins,league=leagues,players=players,
                franchise_context=env.set_index(["season_end_year","franchise_id"]).relative_defense.to_dict(),name_maps={})


def match_players(members,context,year):
    # Identity-only current roster: candidates are chosen only from prior profiles.
    if year not in context["name_maps"]:
        directory=context["players"][context["players"].season_end_year<year]
        context["name_maps"][year]=directory.groupby(directory.player_name.map(name_key)).player_id.agg(lambda x:sorted(set(x))).to_dict()
    by_name=context["name_maps"][year]
    rows=[]
    for r in members.itertuples():
        candidates=by_name.get(name_key(r.player_name),[])
        pid=candidates[0] if len(candidates)==1 else f"nba:unknown:{r.nba_player_id}"
        rows.append(dict(player_id=pid,player_name=r.player_name,matched_prior_identity=len(candidates)==1))
    return pd.DataFrame(rows)


def project_roster(members,year,team_id,franchise_id,team_name,context,state,defensive):
    matched=match_players(members,context,year)
    prior=engine.profile_inputs(matched,year,context)
    known=prior.loc[prior.known==1,"season_end_year"]
    if len(known) and known.max()>=year:raise ValueError("Target-season player profile leaked")
    minutes=engine.projected_rotation(prior,state["rotation"],state["rookie_priors"],state["rotation_model"])
    case=dict(year=year,team_id=team_id,team_name=team_name,season=f"{year-1}-{str(year)[-2:]}",prior=prior,
              context=context["league"].loc[year-1],actual={})
    features=engine.aggregate_case(case,minutes,state["usage"],state["role_usage_means"])
    returning=[franchise_id in context["membership"].get((year-1,pid),set()) for pid in prior.player_id]
    features.update(defense.continuity(minutes,returning,context["franchise_context"].get((year-1,franchise_id),0.)))
    weights=minutes/240.
    contexts=prior.prior_defensive_environment.to_numpy()
    mean_context=float(weights@contexts)
    origin=[context["origins"].get((int(r.season_end_year),r.player_id),"unknown:"+r.player_id) if r.known else "unknown:"+r.player_id for r in prior.itertuples()]
    origin_weights=pd.Series(weights).groupby(origin).sum()
    features.update(defensive_missing_share=float(weights@(1-prior.known)),
                    source_context_sd=float(np.sqrt(weights@np.square(contexts-mean_context))),
                    source_diversity=float(1-np.square(origin_weights).sum()))
    frame=pd.DataFrame([features])
    result=dict(season_end_year=year,team_id=team_id,franchise_id=franchise_id,team_name=team_name,
        source_anchor_ortg=features["anchor_ortg"],source_anchor_pace=features["anchor_pace"],
        roster_players=len(prior),unknown_prior_players=int((prior.known==0).sum()),
        latest_player_profile_year=int(known.max()) if len(known) else None,
        minutes_sum=float(minutes.sum()),continuity=features["returning_minutes_share"],
        opening_information_date=members.information_date.iloc[0])
    for target in ["ortg","pace"]:
        result[target]=float(engine.previous.predict(frame,target,state["models"][target][state["selected_models"][target]])[0])
    result["drtg"]=float(defense.predict(frame,defensive["model"])[0])
    result["uncertainty"]=defense.uncertainty(features,defensive["domain"],defensive["uncertainty_thresholds"])["category"]
    return result


def build_strengths():
    context=load_context()
    state=joblib.load(engine.BUNDLE)
    defensive=joblib.load(ROOT/"artifacts/simulation/defensive_continuity_v1.joblib")
    members=pd.read_csv(ROSTERS)
    result=[]
    for (year,tid),group in members.groupby(["season_end_year","team_id"]):
        if len(result)%60==0:print(f"Frozen lagged roster inference: {len(result)}/{members.groupby(['season_end_year','team_id']).ngroups}",flush=True)
        r=group.iloc[0]
        result.append(project_roster(group,int(year),tid,r.franchise_id,r.team_name,context,state,defensive))
    output=pd.DataFrame(result)
    output.to_csv(OUTPUT,index=False,float_format="%.10f")
    return output


def transport_games(games,strengths):
    output=games.copy()
    for side in ["home","away"]:
        keep=["season_end_year","team_id","ortg","drtg","pace","source_anchor_ortg","source_anchor_pace","uncertainty"]
        team=strengths[keep].rename(columns={k:side+"_engine_"+k for k in keep if k!="season_end_year"})
        output=output.merge(team,left_on=["season_end_year",side+"_team_id"],right_on=["season_end_year",side+"_engine_team_id"],validate="many_to_one")
        for k in ["ortg","drtg"]:
            output[side+"_"+k]=output.league_ortg+output[side+"_engine_"+k]-output[side+"_engine_source_anchor_ortg"]
        output[side+"_pace"]=output.league_pace+output[side+"_engine_pace"]-output[side+"_engine_source_anchor_pace"]
    return output


if __name__=="__main__":
    build_strengths()
