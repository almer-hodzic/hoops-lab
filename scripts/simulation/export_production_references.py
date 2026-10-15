"""Inference-only frozen reference cases using original fitted research objects."""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

import defensive_continuity as defense
import finalize_game_calibration as game
import improve_projection as team
from export_production_simulator import OUTPUT, ROOT, clean, fingerprint


def roster(name, ids, season=None):
    rows = [dict(playerId="br:"+value[0], season=value[1]) if isinstance(value, tuple)
            else dict(playerId="br:"+value, season=season) for value in ids]
    slots = ["PG", "SG", "SF", "PF", "C"]
    return dict(name=name, starters=[dict(row, slot=slots[i]) for i, row in enumerate(rows[:5])], bench=rows[5:])


def requests():
    warriors = roster("2015-16 Warriors", ["curryst01", "thompkl01", "barneha02", "greendr01", "bogutan01",
        "iguodan01", "livinsh01", "speigma01", "barbole01", "ezelife01", "clarkia01", "rushbr01"], "2015-16")
    cavaliers = roster("2015-16 Cavaliers", ["irvinky01", "smithjr01", "jamesle01", "loveke01", "thomptr01",
        "dellama01", "shumpim01", "mozgoti01"], "2015-16")
    mixed = roster("Mixed-era named rotation", [("curryst01", "2015-16"), ("jordami01", "1990-91"),
        ("jamesle01", "2012-13"), ("duncati01", "2002-03"), ("onealsh01", "1999-00"),
        ("ginobma01", "2006-07"), ("pippesc01", "1995-96"), ("olajuha01", "1993-94")])
    ironman = dict(mixed, name="All-time ironman", bench=[])
    return [dict(case=label, request=dict(teamA=a, teamB=b, targetSeason=season, venue=venue,
                simulationCount=10000, seed=12345, ironmanMode=iron)) for label, a, b, season, venue, iron in [
        ("A_historical_full_rosters", warriors, cavaliers, "2016-17", "teamA_home", False),
        ("B_mixed_era", mixed, warriors, "1990-91", "teamB_home", False),
        ("C_neutral", cavaliers, warriors, "2016-17", "neutral", False),
        ("D_home", cavaliers, warriors, "2016-17", "teamA_home", False),
        ("E_seeded", warriors, cavaliers, "1979-80", "neutral", False),
        ("F_ironman_high", ironman, ironman, "2016-17", "neutral", True)] ]


def reference_team(roster_input, season, ironman, context, strength, defensive):
    players, defaults, roles, histories, schedules, teams, league = context
    rows, origins, environments, missing = [], [], [], []
    entries = roster_input["starters"] + roster_input["bench"]
    for entry in entries:
        raw = players[(players.player_id == entry["playerId"]) & (players.season == entry["season"])].iloc[0]
        # The approved lagged adapter supplies history MPG/availability and null priors.
        data = dict(histories=histories, defaults=defaults, roles=roles, schedules=schedules, environments={})
        member = pd.DataFrame([dict(player_id=raw.player_id, player_name=raw.player_name)])
        profile = team.profile_inputs(member, int(raw.season_end_year)+1, data).iloc[0].copy()
        profile["projected_age"] = raw.age  # Selected historical version, as in approved custom_lineup.
        profile["prior_defensive_environment"] = 0.
        rows.append(profile.to_dict())
        source_team = teams[(teams.team_id == raw.team_id) & (teams.season_end_year == raw.season_end_year)]
        env = float(source_team.def_rtg.iloc[0] - league.loc[int(raw.season_end_year)].league_points_per_75/75*100)
        environments.append(env)
        origins.append((int(raw.season_end_year), raw.franchise_id))
        missing.append(1-sum(pd.notna(raw[field]) for field in defense.SOURCE_FIELDS)/len(defense.SOURCE_FIELDS))
    prior = pd.DataFrame(rows)
    prior["prior_defensive_environment"] = environments
    prior["prior_rank"] = prior.prior_mpg.rank(method="first", ascending=False)
    prior["roster_depth"] = len(prior)
    slots = [e["slot"] for e in roster_input["starters"]] + ["BENCH"]*len(roster_input["bench"])
    minutes = team.projected_rotation(prior, strength["rotation"], strength["rookie_priors"], strength["rotation_model"],
                                     mode="ironman" if ironman else "full", slots=slots)
    target = league[league.season == season].iloc[0]
    features = team.aggregate_case(dict(prior=prior, context=target, year=int(target.name), season=season,
        team_id="custom", team_name=roster_input["name"], actual={}), minutes, strength["usage"], strength["role_usage_means"])
    features.update(defense.continuity(minutes, np.ones(len(prior), bool), 999., synthetic=True))
    weights = minutes/240.
    mean = float(weights @ environments)
    grouped = pd.Series(weights).groupby(pd.Series(origins).map(str)).sum()
    features.update(defensive_missing_share=float(weights @ missing),
                    source_context_sd=float(np.sqrt(weights @ np.square(np.asarray(environments)-mean))),
                    source_diversity=float(1-np.square(grouped).sum()))
    frame = pd.DataFrame([features])
    values = {key: float(team.previous.predict(frame, key, strength["models"][key][strength["selected_models"][key]])[0])
              for key in ["ortg", "pace"]}
    values["drtg"] = float(defense.predict(frame, defensive["model"])[0])
    return dict(strengths=values, minutes=minutes, features={k: v for k, v in features.items() if isinstance(v, (int, float))},
                uncertainty=defense.uncertainty(features, defensive["domain"], defensive["uncertainty_thresholds"])["category"])


def export():
    strength = joblib.load(team.BUNDLE)
    defensive = joblib.load(ROOT / "artifacts/simulation/defensive_continuity_v1.joblib")
    final = joblib.load(game.BUNDLE)
    base = ROOT / "data/historical/processed"
    players = team.previous.add_derived_player_features(pd.read_csv(base / "player_seasons.csv"))
    players = players.merge(pd.read_csv(ROOT / "data/analysis/simulation_defensive_source_fields.csv"),
                            on=["player_id", "season_end_year", "team"], validate="one_to_one")
    players = players[players.source_minutes.notna() & ((players.source_minutes-players.minutes).abs() <= 1.)]
    fields = team.previous.PROFILE_FIELDS + ["team_pace", "league_pace"]
    training = players[(players.season_end_year <= 2013) & (players.minutes >= 100)]
    defaults = training[fields].median()
    roles = {role: rows[fields].median().fillna(defaults) for role, rows in training.groupby("position")}
    histories = {pid: rows.sort_values("season_end_year") for pid, rows in players.groupby("player_id")}
    teams = pd.read_csv(base / "team_seasons.csv")
    schedules = teams.assign(games=teams.wins+teams.losses).groupby("season_end_year").games.max().to_dict()
    league = pd.read_csv(base / "league_seasons.csv").set_index("season_end_year")
    context = players, defaults, roles, histories, schedules, teams, league
    cases = requests()
    for case in cases:
        req = case["request"]
        for key in ["teamA", "teamB"]:
            case[key] = reference_team(req[key], req["targetSeason"], req["ironmanMode"], context, strength, defensive)
        a, b = case["teamA"]["strengths"], case["teamB"]["strengths"]
        h, v = (b, a) if req["venue"] == "teamB_home" else (a, b)
        l = league[league.season == req["targetSeason"]].iloc[0]
        frame = pd.DataFrame([dict(league_ortg=l.league_points_per_75/75*100, league_pace=l.league_pace,
            home_pace=h["pace"], away_pace=v["pace"], home_ortg=h["ortg"], home_drtg=h["drtg"],
            away_ortg=v["ortg"], away_drtg=v["drtg"])])
        home = final["home"] if req["venue"] != "neutral" else dict(common_offset=0., home_margin_per100=0.)
        case["expected"] = {key: float(value[0]) for key, value in game.means(frame, home).items()}
        case["simulation"] = game.simulate(frame, home, final["engine_sampler"], final["engine_ot"], 10000, 12345).to_dict("records")[0]
    result = dict(provenance="Original fitted research objects and research profile/rotation/aggregation/prediction/simulate functions; no training or holdout evaluation",
                  researchCommit="a749144", tolerance=1e-9, configSha256=fingerprint(OUTPUT / "config.json"), cases=cases)
    (OUTPUT / "reference_cases.json").write_text(json.dumps(clean(result), indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print("Exported six original-research reference cases (no fits, no benchmark evaluations)")


if __name__ == "__main__":
    export()
