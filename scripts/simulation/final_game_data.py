"""Blind 2026 acquisition and opening-game identity-only roster cache.

The --acquire stage never selects score/stat columns. Outcome parsing is exposed
only to the final evaluation after its protocol has been frozen.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd

from acquire_game_logs import RAW, ROOT, VERSION, DATASET, API, fingerprint, acquire
from build_game_logs import identity, season_end, add_pregame_strength

CACHE = ROOT / "data/simulation/raw/game_final"
ACQUISITION = ROOT / "data/simulation/game_final_acquisition.json"
ROSTERS = ROOT / "data/simulation/game_final_opening_rosters.csv"
PLAYER_RAW = CACHE / "PlayerStatistics.csv"


def blind_index():
    columns=["gameId","gameDateTimeEst","hometeamId","awayteamId","hometeamCity","hometeamName","awayteamCity","awayteamName"]
    index=pd.read_csv(RAW/"Games.csv",usecols=columns)
    index["game_id"]=index.gameId.astype(str).str.zfill(10)
    index["season_end_year"]=index.game_id.map(season_end)
    return index[index.game_id.str.startswith("002")].copy()


def acquire_blind():
    acquire(offline=True)
    CACHE.mkdir(parents=True,exist_ok=True)
    index=blind_index()
    held=index[index.season_end_year==2026]
    counts=pd.concat([held.hometeamId,held.awayteamId]).value_counts()
    boxes=pd.read_csv(RAW/"TeamStatistics.csv",usecols=["gameId","teamId"])
    required={(r.gameId,tid) for r in held.itertuples() for tid in (r.hometeamId,r.awayteamId)}
    if len(held)!=1230 or held.gameId.nunique()!=1230 or len(counts)!=30 or not (counts==82).all():
        raise RuntimeError("STOP: complete 2025-26 regular-season schedule not established")
    if not required<=set(zip(boxes.gameId,boxes.teamId)):
        raise RuntimeError("STOP: required 2025-26 team box keys unavailable")
    if not PLAYER_RAW.exists():
        url=API+"download/"+DATASET+f"/PlayerStatistics.csv?datasetVersionNumber={VERSION}"
        print("Downloading public bulk player table; extracting identity columns only",flush=True)
        temp=PLAYER_RAW.with_suffix(".partial")
        try:
            with urlopen(url,timeout=120) as response,temp.open("wb") as stream:
                if "html" in response.headers.get("Content-Type","").lower():
                    raise RuntimeError("Public roster download returned HTML; no restriction bypass")
                size=0
                while chunk:=response.read(1024*1024):
                    size+=len(chunk)
                    if size>600_000_000:raise RuntimeError("Unexpected oversized player table")
                    stream.write(chunk)
            temp.replace(PLAYER_RAW)
        finally:
            if temp.exists():temp.unlink()
    selected=index[index.season_end_year.isin(list(range(2010,2020))+[2026])].sort_values(["gameDateTimeEst","gameId"])
    first={}
    for row in selected.itertuples():
        for side in ["home","away"]:
            tid=getattr(row,side+"teamId")
            key=(row.season_end_year,tid)
            if key not in first:
                team=identity(getattr(row,side+"teamCity"),getattr(row,side+"teamName"),row.season_end_year)
                first[key]=dict(gameId=row.gameId,season_end_year=row.season_end_year,source_team_id=tid,
                                team_id=team["team_id"],franchise_id=team["franchise_id"],team_name=team["display_name"],
                                information_date=row.gameDateTimeEst[:10])
    by_key={(v["gameId"],v["source_team_id"]):v for v in first.values()}
    records=[]
    fields=["gameId","personId","firstName","lastName","playerteamId"]
    for chunk in pd.read_csv(PLAYER_RAW,usecols=fields,chunksize=100000):
        subset=chunk[chunk.gameId.isin({k[0] for k in by_key})]
        for row in subset.itertuples():
            key=(row.gameId,row.playerteamId)
            if key in by_key:
                name=" ".join(str(v) for v in [row.firstName,row.lastName] if pd.notna(v)).strip()
                records.append(dict(**by_key[key],nba_player_id=int(row.personId),player_name=name))
    roster=pd.DataFrame(records).drop_duplicates(["season_end_year","team_id","nba_player_id"])
    sizes=roster.groupby(["season_end_year","team_id"]).size()
    if len(sizes)!=len(first) or sizes.min()<5:
        raise RuntimeError("Opening-game roster coverage insufficient; do not silently use full-season membership")
    roster.to_csv(ROSTERS,index=False)
    source_hashes={str((RAW/f).relative_to(ROOT)).replace("\\","/"):fingerprint(RAW/f)
                   for f in ["Games.csv","TeamStatistics.csv","TeamStatisticsExtended.csv"]}
    source_hashes[str(PLAYER_RAW.relative_to(ROOT)).replace("\\","/")]=fingerprint(PLAYER_RAW)
    record=dict(status="blind_index_and_rosters_verified",source="https://www.kaggle.com/datasets/"+DATASET+f"/versions/{VERSION}",
        version=VERSION,holdout_games=1230,teams=30,games_per_team=82,required_box_keys=2460,
        dates=[held.gameDateTimeEst.min(),held.gameDateTimeEst.max()],raw_hashes=source_hashes,
        roster_sha256=fingerprint(ROSTERS),roster_team_seasons=len(sizes),roster_rows=len(roster),
        official_roster_attempt="NBA public live boxscore endpoints returned403; not retried/bypassed",
        outcome_columns_accessed_during_acquisition=False,
        roster_method="All identity rows listed in each team's first regular-season game, including DNP rows when source lists them; no minutes, starts, outcomes or target-season performance selected",
        roster_limitation="First-game box roster is a reproducible opening-roster approximation, not a certified preseason transaction roster; appearance/availability selection remains possible",
        holdout_scope="2025-26 regular-season game outcomes untouched by this calibration layer; earlier team research saw aggregate season evaluation",
        old_test_policy="2020-2025 excluded from all calibration fitting/model selection; only prior-date league/input context may use already completed games")
    ACQUISITION.write_text(json.dumps(record,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:record[k] for k in ["status","holdout_games","teams","games_per_team","roster_team_seasons","roster_rows"]},indent=2))


def read_holdout_outcomes(protocol_verified=False):
    if not protocol_verified:raise RuntimeError("2026 outcomes sealed until protocol verification")
    index=pd.read_csv(RAW/"Games.csv",low_memory=False)
    index=index[index.gameId.astype(str).str.zfill(10).str.startswith("00225")].copy()
    boxes=pd.read_csv(RAW/"TeamStatistics.csv",low_memory=False)
    boxes=boxes[boxes.gameId.isin(index.gameId)].set_index(["gameId","teamId"])
    ext=pd.read_csv(RAW/"TeamStatisticsExtended.csv",usecols=["gameId","teamId","possessions"])
    ext=ext[ext.gameId.isin(index.gameId)].set_index(["gameId","teamId"])
    rows=[]
    for game in index.to_dict("records"):
        gid=game["gameId"]
        row=dict(game_id=str(gid).zfill(10),season="2025-26",season_end_year=2026,
                 date=game["gameDateTimeEst"][:10],neutral_site=False,split="new_holdout",eligible=True)
        for side,other in [("home","away"),("away","home")]:
            tid=game[side+"teamId"];box=boxes.loc[(gid,tid)]
            mapped=identity(game[side+"teamCity"],game[side+"teamName"],2026)
            row[side+"_team_id"],row[side+"_franchise_id"]=mapped["team_id"],mapped["franchise_id"]
            points=game[side+"Score"]
            if points!=box.teamScore or points<=0:raise RuntimeError("2026 score reconciliation failed")
            minutes=float(box.numMinutes)/5
            if not (48<=minutes<=78 and abs((minutes-48)/5-round((minutes-48)/5))<.02):
                raise RuntimeError("2026 game duration unavailable or invalid")
            quarters=[box[f"q{i}Points"] for i in range(1,5)]
            if minutes==48:
                reg=points
            elif all(pd.notna(v) and v>=0 for v in quarters) and 0<sum(quarters)<=points:
                reg=sum(quarters)
            elif pd.notna(box.otAllPoints) and 0<=box.otAllPoints<points:
                reg=points-box.otAllPoints
            else:raise RuntimeError("2026 OT regulation points unavailable; no fabricated fallback")
            reported=float(ext.loc[(gid,tid),"possessions"]) if (gid,tid) in ext.index else np.nan
            possessions=reported if np.isfinite(reported) and reported>0 else box.fieldGoalsAttempted+.44*box.freeThrowsAttempted-box.reboundsOffensive+box.turnovers
            row[side+"_points"],row[side+"_regulation_points"],row[side+"_minutes"],row[side+"_possessions"]=points,reg,minutes,possessions
        if row["home_minutes"]!=row["away_minutes"]:raise RuntimeError("2026 duration mismatch")
        row["overtime_periods"]=int(round((row["home_minutes"]-48)/5))
        row["pace"]=(row["home_possessions"]+row["away_possessions"])/2*48/row["home_minutes"]
        row["margin"]=row["home_points"]-row["away_points"]
        row["home_win"]=int(row["margin"]>0)
        row["regulation_margin"]=row["home_regulation_points"]-row["away_regulation_points"]
        if row["overtime_periods"]>0 and row["regulation_margin"]!=0:
            raise RuntimeError("2026 OT regulation scores do not tie")
        row["eligible"]=bool(np.isfinite(row["regulation_margin"]) and row["margin"]!=0 and 60<row["pace"]<140)
        rows.append(row)
    frame=pd.DataFrame(rows)
    history=pd.read_csv(ROOT/"data/simulation/game_logs_processed.csv",dtype={"game_id":str})
    # Past seasons and earlier target dates supply INPUTS, never recalibration targets.
    combined=add_pregame_strength(pd.concat([history.drop(columns=[c for c in history if c.endswith(("_ortg","_drtg","_prior_games","_last_information_date")) or c in ["league_pace","league_prior_season_end_year","home_pace","away_pace"]]),frame],ignore_index=True))
    held=combined[combined.season_end_year==2026].copy()
    if len(held)!=1230 or not held.eligible.all():
        raise RuntimeError("STOP: complete reliable2026 outcome/period/possession coverage not established")
    return held


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acquire",action="store_true",required=True)
    parser.parse_args();acquire_blind()
