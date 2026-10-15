"""Focused validation-only improvements, followed by a single locked final evaluation.

Run --validate (repeatable before freezing), tests, then --final exactly once.
--report only renders stored results; it cannot fit or evaluate a model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from threadpoolctl import threadpool_limits

import validate_deployability as previous
from simulation_engine_spike import ROOT, OUT_DIR, REPORT_DIR, CUSTOM_TEAM, metrics

PREFIX = OUT_DIR / "targeted_projection"
BUNDLE = ROOT / "artifacts" / "simulation" / "targeted_projection_frozen.joblib"
VALIDATION = Path(str(PREFIX) + "_validation.json")
PROTOCOL = Path(str(PREFIX) + "_protocol.json")
FINAL = Path(str(PREFIX) + "_final.json")
LEDGER = Path(str(PREFIX) + "_final_run.json")
ROTATION_FIELDS = ["prior_mpg", "history_mpg", "prior_availability", "projected_age",
                   "prior_start_share", "prior_rank", "roster_depth", "known", "returner"]
TRANSFER_FIELDS = ["usage_pct", "true_shooting_pct", "relative_ts", "assist_pct", "turnover_pct",
                   "three_point_attempt_rate", "minutes_per_game", "defensive_box_plus_minus",
                   "defensive_win_shares", "steal_pct", "block_pct", "defensive_rebound_pct",
                   "steals_per_75", "blocks_per_75", "defensive_rebounds_per_75", "age", "defensive_environment"]
DEFENSE_FIELDS = ["w_defensive_box_plus_minus", "w_steal_pct", "w_block_pct", "w_defensive_rebound_pct"]
DOMAIN_FIELDS = previous.DEPLOYABLE_FEATURE_SET + ["w_age", "center_minutes_share", "prior_defensive_environment"]
SOURCE_FILES = [ROOT / "data/historical/processed" / name for name in
                ["player_seasons.csv", "player_team_splits.csv", "team_seasons.csv", "league_seasons.csv"]]
SOURCE_FILES += [OUT_DIR / "lagged_roster_features.csv", Path(__file__),
                 ROOT / "scripts/simulation/validate_deployability.py",
                 ROOT / "scripts/historical/export_cached_defensive_profiles.py"]
BASELINES = {"validation": {"ortg": 2.200, "drtg": 2.275, "pace": 1.519},
             "test": {"ortg": 2.450, "drtg": 2.392, "pace": 1.688}}


def write_json(path, value):
    path.write_text(json.dumps(previous.json_clean(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def input_fingerprints():
    paths = SOURCE_FILES + sorted((ROOT / "data/historical/raw/basketball-reference").glob("NBA_*_advanced.html"))
    return {str(path.relative_to(ROOT)): fingerprint(path) for path in paths}


def frozen_keys(test=False):
    # Read membership only: no old final-test targets, predictions or errors enter selection.
    rows = pd.read_csv(OUT_DIR / "lagged_roster_features.csv",
                       usecols=["mode", "season_end_year", "team_id"])
    rows = rows[(rows["mode"] == "deployable") & ((rows.season_end_year >= 2020) if test else (rows.season_end_year <= 2019))]
    return set(zip(rows.season_end_year.astype(int), rows.team_id))


def prepare():
    _, roster, teams, leagues, profiles, histories, defaults, roles, quarantine = previous.prepare_inputs()
    source = pd.read_csv(ROOT / "data/analysis/simulation_defensive_source_fields.csv")
    roster = roster.merge(source, on=["player_id", "season_end_year", "team"], how="left", validate="many_to_one")
    valid = roster.source_minutes.notna() & ((roster.source_minutes - roster.minutes).abs() <= 1.)
    roster["identity_verified"] = valid
    for field in previous.DEFENSIVE_FIELDS.values():
        roster.loc[~valid, field] = np.nan
    team_env = teams[["season_end_year", "team_id", "def_rtg"]].merge(
        leagues.reset_index()[["season_end_year", "league_points_per_75"]], on="season_end_year")
    team_env["def_environment"] = team_env.def_rtg - team_env.league_points_per_75 / 75 * 100
    with_env = roster.merge(team_env[["season_end_year", "team_id", "def_environment"]],
                            on=["season_end_year", "team_id"], how="left", validate="many_to_one")
    with_env["env_weight"] = with_env.minutes.where(with_env.def_environment.notna(), 0.)
    with_env["weighted_environment"] = with_env.def_environment.fillna(0.) * with_env.env_weight
    grouped = with_env.groupby(["season_end_year", "player_id"])
    totals = grouped[["env_weight", "weighted_environment"]].sum()
    env = totals.weighted_environment.div(totals.env_weight.replace(0., np.nan)).fillna(0.).to_dict()
    membership = grouped.team_id.agg(set).to_dict()
    schedules = teams.assign(games=teams.wins + teams.losses).groupby("season_end_year").games.max().to_dict()
    return dict(roster=roster, teams=teams, leagues=leagues, profiles=profiles,
                histories=histories, defaults=defaults, roles=roles, quarantine=quarantine,
                environments=env, membership=membership, schedules=schedules)


def profile_inputs(group, year, data):
    """No current performance/starts/minutes is read here; only player identities."""
    prior = previous.prior_profiles(group[["player_id", "player_name"]], year,
                                    data["histories"], data["defaults"], data["roles"], 3)
    extras = []
    for row in prior.itertuples():
        known = bool(row.known)
        history = data["histories"].get(row.player_id)
        if history is not None:
            years = history.season_end_year.to_numpy()
            history = history.iloc[np.searchsorted(years, year - 3):np.searchsorted(years, year)]
        else:
            history = pd.DataFrame()
        if known and len(history):
            history = history[history.minutes > 0].iloc[::-1]
            weights = np.power(.5, np.arange(len(history)))
            history_mpg = float(np.average(history.minutes_per_game, weights=weights))
            games = float(history.iloc[0].games)
            availability = min(games / data["schedules"].get(int(row.season_end_year), 82.), 1.)
            age = float(row.age) + float(row.profile_lag) if pd.notna(row.age) else 26.
            starts = float(row.games_started / row.games) if pd.notna(row.games_started) and row.games > 0 else .5
            raw = history.iloc[0]
            defensive_env = data["environments"].get((int(raw.season_end_year), row.player_id), 0.)
        else:
            history_mpg, availability, age, starts, defensive_env = 8., .5, 26., .25, 0.
        extras.append(dict(history_mpg=history_mpg, prior_availability=availability,
                           projected_age=age, prior_start_share=starts,
                           prior_defensive_environment=defensive_env))
    prior = pd.concat([prior, pd.DataFrame(extras)], axis=1)
    prior["prior_rank"] = prior.prior_mpg.rank(method="first", ascending=False).to_numpy()
    prior["roster_depth"] = len(prior)
    return prior


def build_cases(data, keys):
    groups = {(int(y), team): group.drop_duplicates("player_id").reset_index(drop=True)
              for (y, team), group in data["roster"].groupby(["season_end_year", "team_id"])}
    teams = data["teams"].set_index(["season_end_year", "team_id"])
    cases = []
    for year, team_id in sorted(keys):
        if len(cases) % 250 == 0:
            print(f"  Prior-only profiles: {len(cases)}/{len(keys)} team-seasons", flush=True)
        group = groups[(year, team_id)]
        team = teams.loc[(year, team_id)]
        prior = profile_inputs(group, year, data)
        actual_minutes = group.minutes.to_numpy(float) / group.minutes.sum() * 240.
        labels = group.copy()
        labels["actual_equivalent_mpg"] = actual_minutes
        labels["actual_appearance_mpg"] = group.minutes / group.games
        labels["actual_starter"] = group.games_started / group.games >= .5
        labels["starter_observed"] = group.games_started.notna()
        labels["diagnostic_role"] = np.where(pd.Series(actual_minutes).rank(method="first", ascending=False) <= 5,
                                            "starter", np.where(pd.Series(actual_minutes).rank(method="first", ascending=False) <= 8,
                                                                "primary_bench", "deep_bench"))
        cases.append(dict(year=year, team_id=team_id, season=team.season, team_name=team.team_name,
                          prior=prior, labels=labels, context=data["leagues"].loc[year - 1],
                          actual={"ortg": float(team.off_rtg), "drtg": float(team.def_rtg), "pace": float(team.pace)}))
    return cases


def bounded_minutes(preference, floor, cap):
    preference, floor, cap = map(lambda x: np.asarray(x, float), (preference, floor, cap))
    if not np.isfinite(preference).all() or floor.sum() > 240 or cap.sum() < 240:
        raise ValueError("Infeasible named-player rotation")
    low, high = float(np.min(floor - preference) - 240), float(np.max(cap - preference) + 240)
    for _ in range(80):
        shift = (low + high) / 2
        if np.clip(preference + shift, floor, cap).sum() < 240:
            low = shift
        else:
            high = shift
    result = np.clip(preference + (low + high) / 2, floor, cap)
    assert np.isclose(result.sum(), 240.) and (result <= cap + 1e-8).all()
    return result


def rookie_priors(training):
    if any(case["year"] > 2014 for case in training):
        raise ValueError("Entrant priors may use training outcomes only")
    rows = []
    for case in training:
        labels = case["labels"]
        missing = (case["prior"].known.to_numpy() == 0) & labels.identity_verified.to_numpy()
        selected = labels.loc[missing, ["diagnostic_role", "actual_equivalent_mpg", "actual_appearance_mpg", "usage_pct"]]
        rows.extend(selected.rename(columns={"diagnostic_role": "role", "actual_equivalent_mpg": "minutes",
                                              "actual_appearance_mpg": "appearance_mpg", "usage_pct": "usage"}).to_dict("records"))
    frame = pd.DataFrame(rows)
    roles = {role: dict(n=len(group), equivalent_mpg=float(group.minutes.mean()),
                       appearance_mpg=float(group.appearance_mpg.mean())) for role, group in frame.groupby("role")}
    return dict(roles=roles, unknown_role_mixture_mpg=float(frame.minutes.mean()), n=len(frame))


def role_inputs(prior, priors):
    result = prior[ROTATION_FIELDS].copy()
    unknown = result.known == 0
    result.loc[unknown, ["prior_mpg", "history_mpg"]] = priors["unknown_role_mixture_mpg"]
    # Ranks are recomputed after assigning an honest unknown-role minutes prior.
    result["prior_rank"] = result.prior_mpg.rank(method="first", ascending=False)
    return result


def projected_rotation(prior, config, priors, model=None, mode="historical", slots=None):
    n = len(prior)
    if mode == "ironman":
        if n != 5:
            raise ValueError("Ironman requires exactly five named players")
        return np.full(5, 48.)
    if mode == "full" and not 8 <= n <= 12:
        raise ValueError("Normal full-game mode requires 8-12 named players")
    if n < 8:
        raise ValueError("Historical/full rotation needs at least eight players")
    if config == "old":
        return previous.rotation(prior.prior_mpg)
    inputs = role_inputs(prior, priors)
    if config == "ridge_minutes":
        preference = model.predict(inputs[ROTATION_FIELDS])
    else:
        preference = (.6 * inputs.prior_mpg + .4 * inputs.history_mpg) * (.6 + .4 * inputs.prior_availability)
    rank = np.argsort(-inputs.prior_mpg.to_numpy(), kind="stable")
    starters = rank[:5] if slots is None else np.flatnonzero(np.isin(slots, ["PG", "SG", "SF", "PF", "C"]))
    if slots is not None:
        if len(starters) != 5 or set(np.asarray(slots)[starters]) != {"PG", "SG", "SF", "PF", "C"}:
            raise ValueError("Full roster slots must name exactly five starters")
        for i in np.flatnonzero(prior.known.to_numpy() == 0):
            role = "starter" if i in starters else "deep_bench" if slots[i] == "deep_bench" else "primary_bench"
            preference[i] = priors["roles"][role]["equivalent_mpg"]
    floor, cap = np.full(n, .5), np.full(n, 28.)
    floor[starters], cap[starters] = (20. if slots is not None else 4.), 38.
    if mode == "full":
        floor[np.setdiff1d(np.arange(n), starters)] = 4.
    return bounded_minutes(preference, floor, cap)


def errors(actual, estimated):
    delta = np.asarray(estimated) - np.asarray(actual)
    return dict(n=len(delta), mae=float(np.mean(np.abs(delta))), rmse=float(np.sqrt(np.mean(delta ** 2))), bias=float(np.mean(delta)))


def rotation_diagnostics(cases, minutes, data):
    player_rows, teams = [], []
    for case, projected in zip(cases, minutes):
        p, actual = case["prior"], case["labels"]
        actual_mpg = actual.actual_equivalent_mpg.to_numpy()
        labels = actual.to_dict("records")
        for i, row in enumerate(p.itertuples()):
            label = labels[i]
            prev_teams = data["membership"].get((case["year"] - 1, row.player_id), set())
            category = "no_prior" if not row.known else "absence_return" if row.profile_lag > 1 else "returning" if case["team_id"] in prev_teams else "transferred"
            player_rows.append(dict(season=case["season"], team=case["team_name"], player=row.player_name,
                                    category=category, starter=bool(label["actual_starter"]),
                                    starter_observed=bool(label["starter_observed"]),
                                    prior_games_fraction=float(row.prior_availability),
                                    actual_mpg=float(label["actual_appearance_mpg"]),
                                    actual_equivalent_mpg=float(actual_mpg[i]), projected_mpg=float(projected[i])))
        teams.append(dict(top5_share_error=abs(np.sort(projected)[-5:].sum() - np.sort(actual_mpg)[-5:].sum()) / 240.,
                          top8_share_error=abs(np.sort(projected)[-8:].sum() - np.sort(actual_mpg)[-8:].sum()) / 240.,
                          hhi_error=abs(np.square(projected / 240.).sum() - np.square(actual_mpg / 240.).sum())))
    frame = pd.DataFrame(player_rows)
    results = dict(equivalent_mpg=errors(frame.actual_equivalent_mpg, frame.projected_mpg),
                   appearance_mpg=errors(frame.actual_mpg, frame.projected_mpg),
                   starter=errors(frame[frame.starter & frame.starter_observed].actual_equivalent_mpg,
                                  frame[frame.starter & frame.starter_observed].projected_mpg),
                   bench=errors(frame[~frame.starter & frame.starter_observed].actual_equivalent_mpg,
                                frame[~frame.starter & frame.starter_observed].projected_mpg),
                   starter_appearance=errors(frame[frame.starter & frame.starter_observed].actual_mpg,
                                             frame[frame.starter & frame.starter_observed].projected_mpg),
                   bench_appearance=errors(frame[~frame.starter & frame.starter_observed].actual_mpg,
                                           frame[~frame.starter & frame.starter_observed].projected_mpg),
                   by_prior_availability={name: errors(g.actual_equivalent_mpg, g.projected_mpg) for name, g in frame.groupby("category")},
                   by_prior_availability_appearance={name: errors(g.actual_mpg, g.projected_mpg) for name, g in frame.groupby("category")},
                   **pd.DataFrame(teams).mean().to_dict())
    bins = np.select([frame.category == "no_prior", frame.prior_games_fraction < .5, frame.prior_games_fraction < .8],
                     ["no_prior", "<50%", "50-80%"], default=">=80%")
    results["by_prior_games_played"] = {name: dict(equivalent=errors(g.actual_equivalent_mpg, g.projected_mpg),
                                                 appearance=errors(g.actual_mpg, g.projected_mpg)) for name, g in frame.groupby(bins)}
    results["largest_errors"] = frame.assign(error=(frame.projected_mpg - frame.actual_equivalent_mpg).abs()).nlargest(12, "error").to_dict("records")
    return results, frame


def usage(prior, minutes, config, role_means):
    alpha, shrink = config["alpha"], config["shrink"]
    role = np.where(prior.prior_rank <= 5, "starter", np.where(prior.prior_rank <= 8, "primary_bench", "deep_bench"))
    preference = (1 - shrink) * prior.usage_pct.to_numpy() + shrink * np.array([role_means[r] for r in role])
    effective, share, _ = previous.normalize_usage(preference, minutes, exponent=alpha)
    return effective, share


def usage_diagnostics(cases, rotations, config, role_means):
    rows, teams = [], []
    for case, minutes in zip(cases, rotations):
        pred, share = usage(case["prior"], minutes, config, role_means)
        raw = case["labels"].usage_pct.to_numpy(float)
        valid = np.isfinite(raw) & case["labels"].identity_verified.to_numpy()
        if not valid.all():
            continue
        target, target_share, _ = previous.normalize_usage(raw, minutes, exponent=1.)
        rows.extend(dict(raw=float(raw[i]), target=float(target[i]), predicted=float(pred[i])) for i in range(len(raw)))
        teams.append(dict(top1_error=abs(pred.max() - target.max()),
                          top3_share_error=abs(np.sort(share)[-3:].sum() - np.sort(target_share)[-3:].sum()),
                          concentration_error=abs(np.square(share).sum() - np.square(target_share).sum())))
    frame = pd.DataFrame(rows)
    return dict(player_usage=errors(frame.raw, frame.predicted),
                conserved_reference_usage=errors(frame.target, frame.predicted), **pd.DataFrame(teams).mean().to_dict())


def aggregate_case(case, minutes, usage_config, role_means):
    previous.USAGE_EXPONENT = 1.
    prior = case["prior"]
    features, _, _ = previous.aggregate(prior, minutes, case["context"])
    effective, share = usage(prior, minutes, usage_config, role_means)
    for field, name in [("relative_ts", "relative_ts"), ("three_point_attempt_rate", "3par"), ("free_throw_rate", "ftr")]:
        features["usage_weighted_" + name] = float(np.dot(share, prior[field]))
    features["usage_concentration"] = float(np.square(share).sum())
    features["usage_compression"] = float(np.dot(minutes / 240., np.maximum(prior.usage_pct - effective, 0.)))
    features["w_age"] = float(np.dot(minutes / 240., prior.projected_age))
    features["prior_defensive_environment"] = float(np.dot(minutes / 240., prior.prior_defensive_environment))
    positions = prior.position.fillna("").astype(str)
    features["center_minutes_share"] = float(np.dot(minutes / 240., positions.str.contains("C").astype(float)))
    return dict(season=case["season"], season_end_year=case["year"], team_id=case["team_id"],
                team_name=case["team_name"], **features,
                **{"actual_" + name: value for name, value in case["actual"].items()})


def transfer_pairs(cases, data):
    raw_lookup = {(row["player_id"], int(row["season_end_year"])): row for row in data["profiles"].to_dict("records")}
    rows = []
    for case in cases:
        labels = case["labels"].to_dict("records")
        for i, prior in enumerate(case["prior"].itertuples()):
            target = labels[i]
            if not prior.known or not target["identity_verified"]:
                continue
            raw = raw_lookup[(prior.player_id, int(prior.season_end_year))]
            prev_teams = data["membership"].get((case["year"] - 1, prior.player_id), set())
            transfer = case["team_id"] not in prev_teams
            row = dict(season_end_year=case["year"], season=case["season"], team=case["team_name"],
                       player=prior.player_name, transfer=transfer, lag=int(prior.profile_lag),
                       prior_position=str(raw["position"]), target_position=str(target["position"]),
                       prior_minutes=float(raw["minutes"]), target_minutes=float(target["minutes"]),
                       prior_games=float(raw["games"]), target_games=float(target["games"]),
                       prior_fga=float(raw["field_goal_attempts"]), target_fga=float(target["field_goal_attempts"]),
                       star_transfer=bool(transfer and prior.usage_pct >= 27 and prior.prior_mpg >= 28),
                       ordinary_transfer=bool(transfer and prior.prior_mpg >= 10 and not (prior.usage_pct >= 27 and prior.prior_mpg >= 28)))
            for field in TRANSFER_FIELDS:
                row["prior_" + field] = raw.get(field, np.nan)
                row["target_" + field] = target.get(field, np.nan)
            row["prior_defensive_environment"] = prior.prior_defensive_environment
            row["target_defensive_environment"] = data["environments"].get((case["year"], prior.player_id), np.nan)
            row["large_role_change"] = bool(transfer and (abs(target["usage_pct"] - raw["usage_pct"]) >= 5 or abs(target["minutes_per_game"] - raw["minutes_per_game"]) >= 8))
            rows.append(row)
    return pd.DataFrame(rows)


def stability(frame, fields=TRANSFER_FIELDS):
    result = {}
    for field in fields:
        pair = frame[["prior_" + field, "target_" + field]].dropna()
        n = len(pair)
        result[field] = dict(n=n, correlation=float(pair.iloc[:, 0].corr(pair.iloc[:, 1])) if n > 2 else None,
                             mean_change=float((pair.iloc[:, 1] - pair.iloc[:, 0]).mean()) if n else None,
                             mae_change=float((pair.iloc[:, 1] - pair.iloc[:, 0]).abs().mean()) if n else None)
    return result


def transfer_analysis(pairs):
    validation = pairs.query("2015 <= season_end_year <= 2019")
    result = {name: stability(validation[mask]) for name, mask in {
        "all_transfers": validation.transfer, "star_transfers": validation.star_transfer,
        "ordinary_transfers": validation.ordinary_transfer, "large_role_changes": validation.large_role_change,
        "transfers_500_minutes_both_seasons": validation.transfer & (validation.prior_minutes >= 500) & (validation.target_minutes >= 500) & (validation.lag == 1),
        "returning": ~validation.transfer}.items()}
    result["defensive_era_stability"] = {}
    for label, group in pairs[pairs.transfer].groupby(pd.cut(pairs[pairs.transfer].season_end_year, [1979, 1989, 1999, 2009, 2014, 2019]), observed=True):
        result["defensive_era_stability"][str(label)] = stability(group, list(previous.DEFENSIVE_FIELDS.values()))
    result["position_stability"] = {"validation_transfer_exact_position_match": float(
        (validation.loc[validation.transfer, "prior_position"] == validation.loc[validation.transfer, "target_position"]).mean())}
    return result


def star_analysis(cases, rotations, data):
    rows = []
    for case, minutes in zip(cases, rotations):
        prior, target = case["prior"], case["labels"]
        valid = prior.known.to_numpy(bool) & target.identity_verified.to_numpy()
        high = (prior.usage_pct >= 27).to_numpy() & (minutes >= 20) & valid
        if not valid.any():
            continue
        weights = minutes[valid] / minutes[valid].sum()
        row = dict(season=case["season"], season_end_year=case["year"], team=case["team_name"], high_usage_players=int(high.sum()))
        for field in ["usage_pct", "true_shooting_pct", "relative_ts", "assist_pct", "turnover_pct"]:
            values = target[field].to_numpy(float)[valid] - prior[field].to_numpy(float)[valid]
            row["change_" + field] = float(np.average(values, weights=weights))
        rows.append(row)
    frame = pd.DataFrame(rows)
    val = frame.query("2015 <= season_end_year <= 2019")
    cohorts = []
    for name, group in val.groupby(np.where(val.high_usage_players >= 2, "2+ high-usage players", "0-1 high-usage players")):
        cohorts.append(dict(cohort=name, n=len(group), **group.filter(like="change_").mean().to_dict()))
    famous = frame[frame.team.isin(["Boston Celtics", "Miami Heat", "Golden State Warriors"]) &
                   frame.season.isin(["2007-08", "2010-11", "2016-17"])]
    # Exact pairs, not the Cartesian set of famous teams and seasons.
    famous = famous[((famous.team == "Boston Celtics") & (famous.season == "2007-08")) |
                    ((famous.team == "Miami Heat") & (famous.season == "2010-11")) |
                    ((famous.team == "Golden State Warriors") & (famous.season == "2016-17"))]
    return dict(validation_cohorts=cohorts, named_cases=famous.to_dict("records"),
                automatically_identified_validation_cases=val[val.high_usage_players >= 2].sort_values("change_usage_pct").head(10).to_dict("records"),
                conclusion="Descriptive associations only; selection, role changes and coaching confound effects. No causal star-stacking efficiency penalty is imposed.")


def defensive_tolerances(validation):
    anchor = metrics(validation.actual_drtg, validation.anchor_drtg)["mae"]
    return dict(overall_mae_max=.95 * anchor, anchor_validation_mae=anchor,
                elite_poor_mae_max=5., worst_bucket_abs_bias_max=3., abs_error_p90_max=5.5,
                subera_mae_over_anchor_max=.15, box_heavy_abs_bias_max=2.,
                rationale="Require >=5% validation MAE improvement on league anchor, <=5 extreme-bucket MAE, <=3 bucket bias, <=5.5 p90 error, <=0.15 subera regression, <=2 box-heavy bias. Practical bounds declared before candidate evaluation; no final-test values used.")


def defensive_diagnostics(frame, predictions, box_threshold):
    delta = predictions - frame.actual_drtg.to_numpy()
    relative = frame.actual_drtg - frame.groupby("season_end_year").actual_drtg.transform("mean")
    buckets = pd.cut(relative, [-np.inf, -3, -1, 1, 3, np.inf], labels=["elite", "above_average", "average", "below_average", "poor"])
    working = frame.assign(error=delta, bucket=buckets)
    result = dict(overall=metrics(frame.actual_drtg, predictions), abs_error_p90=float(np.quantile(abs(delta), .9)), buckets={})
    for label, group in working.groupby("bucket", observed=True):
        result["buckets"][str(label)] = dict(n=len(group), mae=float(group.error.abs().mean()), bias=float(group.error.mean()))
    result["suberas"] = {}
    for label, group in working.groupby(np.where(working.season_end_year <= 2016, "2015-2016", "2017-2019")):
        result["suberas"][label] = dict(mae=float(group.error.abs().mean()), anchor_mae=float((group.anchor_drtg - group.actual_drtg).abs().mean()))
    box = working[working.w_defensive_box_plus_minus >= box_threshold]
    result["box_heavy"] = dict(n=len(box), bias=float(box.error.mean()) if len(box) else 0., threshold=box_threshold)
    return result


def accepted_defense(diagnostics, limits):
    buckets = diagnostics["buckets"]
    checks = dict(overall=diagnostics["overall"]["mae"] <= limits["overall_mae_max"],
                  extremes=all(buckets[b]["mae"] <= limits["elite_poor_mae_max"] for b in ["elite", "poor"] if b in buckets),
                  bucket_bias=all(abs(b["bias"]) <= limits["worst_bucket_abs_bias_max"] for b in buckets.values()),
                  p90=diagnostics["abs_error_p90"] <= limits["abs_error_p90_max"],
                  suberas=all(g["mae"] <= g["anchor_mae"] + limits["subera_mae_over_anchor_max"] for g in diagnostics["suberas"].values()),
                  box_heavy=abs(diagnostics["box_heavy"]["bias"]) <= limits["box_heavy_abs_bias_max"])
    return dict(accepted=all(checks.values()), checks=checks)


def fit_team_models(train, val, transfer_training):
    if train.season_end_year.max() > 2014 or val.season_end_year.min() < 2015 or val.season_end_year.max() > 2019:
        raise ValueError("Model selection cannot access the frozen final test")
    if len(transfer_training) and transfer_training.season_end_year.max() > 2014:
        raise ValueError("Defensive stability filtering is training-only")
    stable = [field for field in ["steal_pct", "block_pct", "defensive_rebound_pct"]
              if stability(transfer_training, [field])[field]["correlation"] >= .35]
    candidates = {
        "ortg": {"formula": previous.FORMULA_FIELDS["ortg"], "ridge_compact": previous.FORMULA_FIELDS["ortg"]},
        "drtg": {"previous_formula": DEFENSE_FIELDS,
                 "stable_rates": ["w_" + f for f in stable] + ["w_age", "center_minutes_share"],
                 "prior_environment": DEFENSE_FIELDS + ["prior_defensive_environment", "w_age", "rotation_hhi"]},
        "pace": {"compact": previous.FORMULA_FIELDS["pace"], "previous_ridge": previous.DEPLOYABLE_FEATURE_SET},
    }
    limits = defensive_tolerances(val)
    box_threshold = float(train.w_defensive_box_plus_minus.quantile(.9))
    scores, fitted, selected, defensive = {}, {}, {}, {}
    for target, configs in candidates.items():
        scores[target], fitted[target] = {}, {}
        for name, columns in {"league_anchor": [], **configs}.items():
            if columns:
                regression = LinearRegression() if name in {"formula", "previous_formula"} else Ridge(alpha=30.)
                model = previous.pipeline(regression)
                model.fit(train[columns], train["actual_" + target] - train["anchor_" + target])
            else:
                model = None
            candidate = model, columns
            fitted[target][name] = candidate
            prediction = previous.predict(val, target, candidate)
            scores[target][name] = metrics(val["actual_" + target], prediction)
            if target == "drtg":
                diagnostics = defensive_diagnostics(val, prediction, box_threshold)
                defensive[name] = dict(diagnostics=diagnostics, **accepted_defense(diagnostics, limits))
        eligible = [name for name in configs if target != "drtg" or defensive[name]["accepted"]]
        pool = eligible if eligible else list(configs)
        best = min(pool, key=lambda name: scores[target][name]["mae"])
        # Prefer the smaller equation when its MAE is within 0.05 points of best.
        near = [name for name in pool if scores[target][name]["mae"] <= scores[target][best]["mae"] + .05]
        selected[target] = min(near, key=lambda name: (len(configs[name]), list(configs).index(name)))
    return fitted, selected, scores, dict(tolerances=limits, candidates=defensive,
                                         stable_transfer_fields=stable, box_threshold=box_threshold)


def domain_reference(train):
    return {field: dict(min=float(train[field].min()), max=float(train[field].max()),
                        p01=float(train[field].quantile(.01)), p99=float(train[field].quantile(.99)),
                        scale=max(float(train[field].quantile(.75) - train[field].quantile(.25)),
                                  float(train[field].max() - train[field].min()) * .01, 1e-6)) for field in DOMAIN_FIELDS}


def extrapolation(features, reference):
    rows = []
    for field, bounds in reference.items():
        value = float(features[field])
        tolerance = 1e-10 * max(1., abs(bounds["min"]), abs(bounds["max"]))
        excess = max(bounds["min"] - value, value - bounds["max"], 0.)
        exceed = excess / bounds["scale"] if excess > tolerance else 0.
        status = "outside" if exceed > 0 else "near_edge" if value < bounds["p01"] or value > bounds["p99"] else "inside"
        rows.append(dict(feature=field, value=value, status=status, exceed_iqr_units=exceed, **bounds))
    distances = np.array([row["exceed_iqr_units"] for row in rows])
    outside = float(np.mean(distances > 0))
    maximum, rms = float(distances.max()), float(np.sqrt(np.square(distances).mean()))
    risk = "high" if maximum > 1 or outside > .15 else "elevated" if outside > 0 else "within_marginal_support"
    return dict(risk=risk, rms_outside_domain_distance=rms, maximum_exceed_iqr_units=maximum,
                fraction_outside=outside, features=rows,
                interpretation="Feature coverage/uncertainty only, not quality or probability; correlated features and unobserved lineup interactions limit interpretation.")


def custom_lineup(data, state):
    rows = []
    for slot, name, season in CUSTOM_TEAM:
        row = data["profiles"][(data["profiles"].player_name == name) & (data["profiles"].season == season)].iloc[0].copy()
        row["known"], row["returner"], row["profile_lag"] = 1., 0., 1.
        row["projected_age"], row["prior_defensive_environment"] = row.age, data["environments"].get((int(row.season_end_year), row.player_id), 0.)
        row["prior_rank"] = 0.
        rows.append(row)
    prior = pd.DataFrame(rows).reset_index(drop=True)
    prior["prior_rank"] = prior.minutes_per_game.rank(method="first", ascending=False)
    case = dict(prior=prior, year=2017, season="2016-17", team_id="custom", team_name="All-time ironman",
                context=data["leagues"].loc[2016], actual={"ortg": np.nan, "drtg": np.nan, "pace": np.nan})
    minutes = projected_rotation(prior, state["rotation"], state["rookie_priors"], mode="ironman")
    features = aggregate_case(case, minutes, state["usage"], state["role_usage_means"])
    effective, share = usage(prior, minutes, state["usage"], state["role_usage_means"])
    values = {target: float(previous.predict(pd.DataFrame([features]), target, state["models"][target][state["selected_models"][target]])[0]) for target in previous.TARGETS}
    return dict(semantics="Five named players, 48 minutes each; ironman hypothetical only; not a game simulation",
                target_era="2016-17 preseason using prior league anchors, no fatigue/bench model",
                players=[dict(player=row.player_name, season=row.season, minutes=48., usage_pct=float(effective[i]),
                              possession_share=float(share[i])) for i, row in prior.iterrows()],
                projections=values, extrapolation=extrapolation(features, state["domain"]),
                defensive_uncertainty="High: box/context estimates plus a cross-era five-star roster outside ordinary-team support; validation residual bounds are not calibrated ironman coverage.")


def defensive_feature_audit(data, pairs):
    profiles = data["profiles"].query("season_end_year <= 2019")
    fields = list(previous.DEFENSIVE_FIELDS.values()) + ["steals_per_75", "blocks_per_75", "defensive_rebounds_per_75", "age", "position"]
    source = {"defensive_box_plus_minus": "Cached Basketball-Reference advanced DBPM; source-estimated box/context impact",
              "defensive_win_shares": "Cached advanced DWS; cumulative, strongly team/minutes dependent; excluded",
              "steal_pct": "Cached advanced STL%; source-computed player box rate",
              "block_pct": "Cached advanced BLK%; source-computed player box rate",
              "defensive_rebound_pct": "Cached advanced DRB%; source-computed player opportunity rate"}
    rows = []
    stable_training = stability(pairs.query("season_end_year <= 2014 and transfer"))
    for field in fields:
        rows.append(dict(feature=field, provenance=source.get(field, "Canonical totals/advanced; per75 uses prior minutes and prior team pace" if "per_75" in field else "Canonical source age/position"),
                         context="team-dependent estimate" if field in {"defensive_box_plus_minus", "defensive_win_shares"} else "player attribute/rate with role and opponent context",
                         training_transfer_correlation=stable_training.get(field, {}).get("correlation"),
                         availability_by_era={str(era): float(group[field].notna().mean()) for era, group in profiles.groupby((profiles.season_end_year // 10) * 10)}))
    rows += [dict(feature="rotation_hhi", provenance="Proposed named-player minutes", context="roster structure; no realized target inputs",
                  transfer_stability="Not an individual metric; evaluated through team validation, not player correlation"),
             dict(feature="prior_defensive_environment", provenance="Prior team DRtg minus prior league anchor, weighted by prior player team minutes", context="intentionally context-heavy; lagged only",
                  training_transfer_correlation=stable_training["defensive_environment"]["correlation"])]
    return rows


def validation_stage():
    if LEDGER.exists() or FINAL.exists():
        raise RuntimeError("Final evaluation already consumed; this protocol cannot be retuned")
    print("Building training/validation cases only (target years <=2019)...", flush=True)
    data = prepare()
    cases = build_cases(data, frozen_keys())
    train = [case for case in cases if case["year"] <= 2014]
    val = [case for case in cases if 2015 <= case["year"] <= 2019]
    priors = rookie_priors(train)
    print("Fitting training-only minutes model and comparing validation rotations...", flush=True)
    inputs = pd.concat([role_inputs(case["prior"], priors) for case in train])
    target_minutes = np.concatenate([case["labels"].actual_equivalent_mpg for case in train])
    rotation_model = previous.pipeline(Ridge(alpha=100.)).fit(inputs, target_minutes)
    configs = ["old", "balanced_prior", "ridge_minutes"]
    rotation_scores, rotations = {}, {}
    for config in configs:
        rotations[config] = [projected_rotation(case["prior"], config, priors, rotation_model) for case in val]
        rotation_scores[config], _ = rotation_diagnostics(val, rotations[config], data)
    # Select on the conserved target (240 total), not on per-appearance MPG which does not sum to 240.
    rotation_choice = min(configs[1:], key=lambda name: rotation_scores[name]["equivalent_mpg"]["mae"])
    if rotation_scores["balanced_prior"]["equivalent_mpg"]["mae"] <= rotation_scores[rotation_choice]["equivalent_mpg"]["mae"] + .10:
        rotation_choice = "balanced_prior"
    if rotation_scores[rotation_choice]["equivalent_mpg"]["mae"] >= rotation_scores["old"]["equivalent_mpg"]["mae"]:
        rotation_choice = "old"
    role_rows = pd.concat([case["labels"] for case in train])
    print("Comparing validation usage allocations...", flush=True)
    role_usage = role_rows.groupby("diagnostic_role").usage_pct.mean().to_dict()
    usage_configs = [dict(name="alpha1", alpha=1., shrink=0.), dict(name="alpha075", alpha=.75, shrink=0.),
                     dict(name="role_shrink10", alpha=1., shrink=.1)]
    usage_scores = {config["name"]: usage_diagnostics(val, rotations[rotation_choice], config, role_usage) for config in usage_configs}
    best = min(usage_configs, key=lambda config: usage_scores[config["name"]]["player_usage"]["mae"])
    usage_choice = best if usage_scores["alpha1"]["player_usage"]["mae"] - usage_scores[best["name"]]["player_usage"]["mae"] >= .05 else usage_configs[0]
    all_rotations = [projected_rotation(case["prior"], rotation_choice, priors, rotation_model) for case in cases]
    features = pd.DataFrame([aggregate_case(case, minutes, usage_choice, role_usage) for case, minutes in zip(cases, all_rotations)])
    pairs = transfer_pairs(cases, data)
    print("Auditing transfer stability and fitting training-only team models...", flush=True)
    train_features = features[features.season_end_year <= 2014]
    val_features = features.query("2015 <= season_end_year <= 2019")
    models, selected, scores, defensive = fit_team_models(train_features, val_features, pairs.query("season_end_year <= 2014 and transfer"))
    state = dict(rotation=rotation_choice, rotation_model=rotation_model, rookie_priors=priors,
                 usage=usage_choice, role_usage_means=role_usage, models=models, selected_models=selected,
                 domain=domain_reference(train_features))
    custom = custom_lineup(data, state)
    defense_ok = defensive["candidates"][selected["drtg"]]["accepted"]
    readiness = dict(defensive_tolerances_pass=defense_ok,
                     rotation_not_worse=rotation_scores[rotation_choice]["equivalent_mpg"]["mae"] <= rotation_scores["old"]["equivalent_mpg"]["mae"],
                     usage_not_worse=usage_scores[usage_choice["name"]]["player_usage"]["mae"] <= usage_scores["alpha1"]["player_usage"]["mae"] + .05,
                     offense_within_10pct_prior_validation=scores["ortg"][selected["ortg"]]["mae"] <= BASELINES["validation"]["ortg"] * 1.1,
                     pace_within_10pct_prior_validation=scores["pace"][selected["pace"]]["mae"] <= BASELINES["validation"]["pace"] * 1.1)
    # Foundation readiness is decided before inspecting the final test.
    verdict = "A. READY FOR GAME-LOG CALIBRATION" if all(readiness.values()) else "B. NEEDS ONE MORE TARGETED MODEL FIX"
    result = dict(population=dict(train=len(train), validation=len(val), final_test_keys=len(frozen_keys(True))),
                  rotation_before_after=rotation_scores, selected_rotation=rotation_choice, rookie_priors=priors,
                  usage_comparison=usage_scores, selected_usage=usage_choice,
                  transfer_stability=transfer_analysis(pairs), star_combinations=star_analysis(cases, all_rotations, data),
                  defensive_feature_audit=defensive_feature_audit(data, pairs), defense_acceptance=defensive,
                  validation_candidates=scores, selected_models=selected,
                  model_equations={target: previous.equation_parameters(models[target][selected[target]]) for target in selected},
                  selected_validation={target: scores[target][selected[target]] for target in selected},
                  readiness_checks=readiness,
                  custom_lineup=custom, verdict_before_test=verdict, baseline_mae=BASELINES,
                  quarantine=data["quarantine"])
    OUT_DIR.mkdir(exist_ok=True)
    BUNDLE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(state, BUNDLE)
    write_json(VALIDATION, result)
    features.to_csv(Path(str(PREFIX) + "_validation_features.csv"), index=False)
    pairs.query("2015 <= season_end_year <= 2019").to_csv(Path(str(PREFIX) + "_validation_transfers.csv"), index=False)
    _, player_errors = rotation_diagnostics(val, rotations[rotation_choice], data)
    player_errors.to_csv(Path(str(PREFIX) + "_validation_rotation.csv"), index=False)
    protocol = dict(status="frozen_before_final_test", inputs=input_fingerprints(), bundle_sha256=fingerprint(BUNDLE),
                    validation_sha256=fingerprint(VALIDATION), final_test_keys_sha256=hashlib.sha256(json.dumps(sorted(frozen_keys(True))).encode()).hexdigest(),
                    rotation=rotation_choice, usage=usage_choice, selected_models=selected,
                    thresholds=defensive["tolerances"], verdict_before_test=verdict,
                    final_test_policy="2019-20 through 2025-26, unchanged 210 teams; one selected-model evaluation; no retuning")
    write_json(PROTOCOL, protocol)
    render_report()
    print(json.dumps(previous.json_clean({"rotation": rotation_choice, "usage": usage_choice,
                     "validation": result["selected_validation"], "defense_accepted": defense_ok, "verdict": verdict}), indent=2))


def final_stage():
    if LEDGER.exists() or FINAL.exists():
        raise RuntimeError("Final test already consumed; refusing another evaluation")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if input_fingerprints() != protocol["inputs"] or fingerprint(BUNDLE) != protocol["bundle_sha256"] or fingerprint(VALIDATION) != protocol["validation_sha256"]:
        raise RuntimeError("Frozen inputs/model/code/results changed; final evaluation refused")
    keys = frozen_keys(True)
    if len(keys) != 210 or hashlib.sha256(json.dumps(sorted(keys)).encode()).hexdigest() != protocol["final_test_keys_sha256"]:
        raise RuntimeError("Final population changed")
    state = joblib.load(BUNDLE)
    data = prepare()
    cases = build_cases(data, keys)
    features = pd.DataFrame([aggregate_case(case, projected_rotation(case["prior"], state["rotation"], state["rookie_priors"], state["rotation_model"]),
                                            state["usage"], state["role_usage_means"]) for case in cases])
    if len(features) != 210 or not np.isfinite(features[list(set(sum([state["models"][t][state["selected_models"][t]][1] for t in previous.TARGETS], [])))].to_numpy()).all():
        raise RuntimeError("Final feature preflight failed, no evaluation consumed")
    # Exclusive ledger is created BEFORE any predictions or test metrics; retries cannot silently retest.
    with LEDGER.open("x", encoding="utf-8") as stream:
        json.dump(dict(status="evaluation_started", protocol_sha256=fingerprint(PROTOCOL), evaluation_count=1), stream, indent=2)
    output = features[["season", "team_name", "actual_ortg", "actual_drtg", "actual_pace"]].copy()
    scores = {}
    for target in previous.TARGETS:
        prediction = previous.predict(features, target, state["models"][target][state["selected_models"][target]])
        output["predicted_" + target] = prediction
        scores[target] = metrics(features["actual_" + target], prediction)
    output.to_csv(Path(str(PREFIX) + "_final_predictions.csv"), index=False)
    lakers = output[(output.season == "2019-20") & (output.team_name == "Los Angeles Lakers")]
    result = dict(evaluation_count=1, population=len(output), metrics=scores, prior_test_mae=BASELINES["test"],
                  lakers_2019_20=lakers.to_dict("records"), verdict=protocol["verdict_before_test"],
                  protocol_sha256=fingerprint(PROTOCOL), retuning_after_test=False)
    write_json(FINAL, result)
    write_json(LEDGER, dict(status="complete", evaluation_count=1, protocol_sha256=fingerprint(PROTOCOL), final_sha256=fingerprint(FINAL)))
    render_report()
    print(json.dumps(result, indent=2))


def render_report():
    if not VALIDATION.exists():
        raise RuntimeError("Run validation first")
    val = json.loads(VALIDATION.read_text(encoding="utf-8"))
    final = json.loads(FINAL.read_text(encoding="utf-8")) if FINAL.exists() else None
    table = previous.table
    sections = ["\n## Rotation Validation", "",
                "Focused improvement protocol: `python scripts/simulation/improve_projection.py --validate`, "
                "then tests, then `--final` once. `--report` only renders stored results. "
                "Training targets end 2013-14; validation is 2014-15 through 2018-19. "
                "Final 2019-20 through 2025-26 retains all 210 original team keys. "
                "The frozen protocol fingerprints code, canonical data, prior membership, model bundle and validation results; "
                "an exclusive ledger prevents a second final evaluation. No test outcomes inform selection/tolerances.", "",
                "Detailed artifacts: `data/simulation/targeted_projection_validation.json`, "
                "`targeted_projection_protocol.json`, validation rotation/transfer/features CSVs, and final JSON/predictions CSV. "
                "Frozen fitted models are cached under ignored `artifacts/simulation/`.", "",
                "MPG requires two denominators: actual per-appearance minutes/games does not sum to 240 because "
                "injuries, partial seasons, trades and player absences overlap. Primary rotation error uses "
                "team-equivalent MPG = 240*player season minutes/sum(team player minutes), the allocation the model "
                 "must conserve. Per-appearance errors are separately reported; they are not interchangeable. "
                "Target starts label starter/bench diagnostics only; they never enter rotation inputs.", ""]
    rows = []
    for name, scores in val["rotation_before_after"].items():
        rows.append(dict(model=name, equivalent_mae=scores["equivalent_mpg"]["mae"], equivalent_rmse=scores["equivalent_mpg"]["rmse"],
                         appearance_mae=scores["appearance_mpg"]["mae"], appearance_rmse=scores["appearance_mpg"]["rmse"],
                         starter_mae=scores["starter"]["mae"], bench_mae=scores["bench"]["mae"],
                         starter_appearance_mae=scores["starter_appearance"]["mae"], bench_appearance_mae=scores["bench_appearance"]["mae"],
                         top5_share_error=scores["top5_share_error"], top8_share_error=scores["top8_share_error"], hhi_error=scores["hhi_error"]))
    sections += [table(rows), "", "Selected rotation: `" + val["selected_rotation"] + "`.", "",
                 "The balanced prior uses 0.6*last MPG + 0.4*three-year MPG (history weights1,0.5,0.25), "
                 "multiplied by 0.6+0.4*prior availability. The learned candidate is a fixed Ridge(alpha100) "
                 "predicting team-equivalent minutes from prior MPG/history, games availability, projected age, prior "
                 "starts fraction, prior rank, roster depth, known-profile and absence flags. Training outcomes are "
                 "labels only. No selected rotation uses target MPG/starts. Select lower validation equivalent-MPG MAE, "
                 "preferring the transparent balanced rule within0.10 MPG. This validation-only practical-equivalence "
                 "choice avoids selecting a nine-feature minutes regression for a tiny MAE gain while its "
                 "top-five share/concentration errors are worse. "
                 "retain old rotation if neither candidate improves. Historical caps38 for prior-ranked first five, "
                 "28 for others; floors4/0.5 preserve depth. A scalar shift with clipping projects onto exactly240 "
                 "minutes. Unlike the old hard24 starter floor, it permits prior stars to lose roles.", "",
                 "Unknown entrants receive a training-only mixture minutes prior, not fabricated NBA skill observations. "
                 "When legitimate Team Builder slots are supplied, training-calibrated starter/primary/deep-bench "
                 "minutes priors are available. Historical new-entrant roles are unknown; target starts/rank are never "
                 "used to assign them. Statistical profile medians remain explicitly uncertain population imputations.", "",
                 "Training entrant priors: `" + json.dumps(val["rookie_priors"]) + "`", ""]
    sections.append(table([dict(model=name, group=group, **score) for name in ["old", val["selected_rotation"]]
                           for group, score in val["rotation_before_after"][name]["by_prior_availability"].items()]))
    sections += ["", "Per-appearance MPG errors by the same prior-history categories:", "",
                 table([dict(model=name, group=group, **score) for name in ["old", val["selected_rotation"]]
                        for group, score in val["rotation_before_after"][name]["by_prior_availability_appearance"].items()]), "",
                 "Prior games-played coverage (fraction of prior league schedule):", "",
                 table([dict(model=name, availability=group, denominator=kind, **score)
                        for name in ["old", val["selected_rotation"]] for group, kinds in val["rotation_before_after"][name]["by_prior_games_played"].items()
                        for kind, score in kinds.items()])]
    sections += ["", "Largest errors after selection:", "", table(val["rotation_before_after"][val["selected_rotation"]]["largest_errors"]), "",
                 "Structural limits: actual season-long rosters contain more players than an active game roster; "
                 "membership is known in hindsight, but signing/trade dates and injury availability are not modeled. "
                 "A single240-minute allocation cannot reproduce per-appearance roles or predict unknown rookie talent. "
                 "Prior teammate continuity and prior team membership classify returning/transfer cases; multi-team "
                 "prior profiles do not identify exact offseason transfer dates.", "",
                 "Product rule: normal full-game mode requires8-12 named players, exactly240 minutes; "
                 "provided starter slots receive floors20/caps38 and bench floors4/caps28. Five-player mode is "
                 "explicit ironman only,48 each, high uncertainty. Six/seven-player full-game rosters are rejected. "
                 "Historical research can retain larger season-long rosters; no anonymous players are invented. "
                 "Team Builder UI is unchanged.", "", "## Player Transfer Stability", "",
                 "Correlations and mean absolute/change statistics below use validation transfers only, with "
                 "verified source identities and sourced values. Star transfers: prior USG>=27 and MPG>=28; "
                 "ordinary transfers: MPG>=10 excluding stars. Large role changes (diagnostic labels only): "
                 "absolute USG change>=5 or appearance MPG change>=8. Subsets overlap and player/team rows "
                 "are not independent. Absence returns are identified separately in rotation diagnostics.", ""]
    sections += ["Stability samples include low-minute stints: correlations for all transfers are noisy, "
                 "not estimates of intrinsic skill persistence. Inspect the separately reported star/ordinary "
                 "subsets and per-player raw attempts/minutes before assigning strong causal weights. "
                 "Changes in TS and 3PAr use fractions; usage/AST/TOV and defensive percentages use percentage points.", ""]
    sections.append(table([dict(subset=subset, field=field, **score) for subset, fields in val["transfer_stability"].items()
                           if subset not in {"defensive_era_stability", "position_stability"} for field, score in fields.items()]))
    sections += ["", "Position-role stability: `" + json.dumps(val["transfer_stability"]["position_stability"]) + "`", ""]
    sections += ["", "Defensive transfer stability by historical era (training and validation only):", "",
                 table([dict(era=era, field=field, **score) for era, fields in val["transfer_stability"]["defensive_era_stability"].items() for field, score in fields.items()]),
                 "", "## Usage Normalization Validation", "",
                 "e_i=m_i/48; q_i=clip((1-shrink)*prior_USG_i+shrink*training_role_mean,1,60)^alpha; "
                 "v_i=100*q_i/sum(e_j*q_j); s_i=e_i*v_i/100. Thus sum(s_i)=1. "
                 "Compare alpha1, alpha0.75 and10% shrinkage to training-only rank-role means. Keep alpha1 unless "
                 "validation raw-USG MAE improves by at least0.05 percentage points. Target rank/starts never "
                 "determine predictor role; prior-ranked role is used. Raw-USG errors compare actual observations; "
                 "distribution/concentration errors use actual USG normalized under the same projected minutes "
                 "to avoid a denominator artifact. This latter reference is diagnostic, not a predictor.", ""]
    sections.append(table([dict(candidate=name, raw_usage_mae=score["player_usage"]["mae"],
                               conserved_usage_mae=score["conserved_reference_usage"]["mae"], top1=score["top1_error"],
                               top3_share=score["top3_share_error"], concentration=score["concentration_error"]) for name, score in val["usage_comparison"].items()]))
    sections += ["", "Selected usage: `" + json.dumps(val["selected_usage"]) + "`", "",
                 "Star combinations (prior USG>=27, projected MPG>=20):", "", table(val["star_combinations"]["validation_cohorts"]), "",
                 table(val["star_combinations"]["named_cases"]), "",
                 "Automatically identified validation cases:", "", table(val["star_combinations"]["automatically_identified_validation_cases"]), "",
                 val["star_combinations"]["conclusion"], "",
                 "## Defensive Acceptance Tolerances", "",
                 "Defensive provenance and availability (source availability is not imputed availability):", ""]
    sections.append(table([dict(**{k: v for k, v in row.items() if k != "availability_by_era"},
                                availability=json.dumps(row.get("availability_by_era", {}))) for row in val["defensive_feature_audit"]]))
    sections += ["", "DBPM/DWS incorporate team/box context and are not causal individual defense. DWS is not used. "
                 "Stable-rates candidate retains only rate features whose training-transfer correlation>=0.35. "
                 "The compact prior-environment candidate deliberately includes only lagged team environment; "
                 "none uses target DRtg/pace. Prior age/position and projected concentration are available without "
                 "target performance. All models predict residual DRtg relative to prior league scoring.", "",
                 "Limits fixed before candidate evaluation: `" + json.dumps(val["defense_acceptance"]["tolerances"]) + "`", "",
                 "Buckets use actual DRtg minus same-season league mean for diagnostics: elite<-3, above-average "
                 "[-3,-1), average[-1,1], below-average(1,3], poor>3. Positive prediction bias means underrating "
                 "defense; negative bias means overrating it. The training90th-percentile prior DBPM defines "
                 "box-heavy teams. Quality buckets are never predictors.", ""]
    rows = []
    for name, candidate in val["defense_acceptance"]["candidates"].items():
        rows.append(dict(candidate=name, accepted=candidate["accepted"], checks=json.dumps(candidate["checks"])))
        sections += ["Candidate `" + name + "` buckets:", "", table([dict(bucket=bucket, **score) for bucket, score in candidate["diagnostics"]["buckets"].items()]), "",
                     "Suberas: `" + json.dumps(candidate["diagnostics"]["suberas"]) + "`; box-heavy bias: `" + json.dumps(candidate["diagnostics"]["box_heavy"]) + "`", ""]
    sections.append(table(rows))
    sections += ["", "Validation model comparisons (smaller equation preferred within0.05 MAE):", "",
                 table([dict(target=target, candidate=name, **score) for target, candidates in val["validation_candidates"].items() for name, score in candidates.items()]), "",
                 "Selected models: `" + json.dumps(val["selected_models"]) + "`", "",
                 "Exact selected equation coefficients, intercepts and training imputation medians are exported "
                 "under `model_equations` in the validation JSON. The frozen artifact also retains the minutes "
                 "regression and all preprocessing fitted on training rows only.", "",
                 "## Synthetic Extrapolation Detection", "",
                 "Each important deployable feature is inside its training1st-99th percentile interval, near the "
                 "edge inside min/max, or outside observed min/max. Distance is excess beyond min/max divided "
                 "by training IQR (minimum scale max(1% range,1e-6)). Report RMS distance, largest excess and "
                 "outside fraction. High uncertainty if any excess>1 IQR or >15% features outside; elevated if "
                 "any outside. This is a domain-coverage indicator, never a quality score or calibrated confidence. "
                 "Marginal ranges cannot establish joint lineup support; correlated features may duplicate evidence.", "",
                 table(val["custom_lineup"]["extrapolation"]["features"]), "",
                 "Custom ironman: " + val["custom_lineup"]["semantics"], "", table(val["custom_lineup"]["players"]), "",
                 "Target era: " + val["custom_lineup"]["target_era"], "",
                 "Projections: `" + json.dumps(val["custom_lineup"]["projections"]) + "`", "",
                 "Extrapolation: `" + json.dumps({k: v for k, v in val["custom_lineup"]["extrapolation"].items() if k != "features"}) + "`", "",
                 val["custom_lineup"]["defensive_uncertainty"], "", "### Final Frozen Evaluation", ""]
    if final:
        sections += ["Exactly one final evaluation completed after protocol freeze. No retuning followed. "
                     "Prior-test baseline MAE is comparison-only, never an acceptance threshold.", "",
                     table([dict(target=target, previous_mae=BASELINES["test"][target], **score) for target, score in final["metrics"].items()]), "",
                     "2019-20 Lakers (held until final evaluation):", "", table(final["lakers_2019_20"]), ""]
    else:
        sections += ["Pending. No final-test performance has been evaluated in this protocol.", ""]
    sections += ["Verdict fixed before final evaluation: **" + val["verdict_before_test"] + "**.", "",
                 "Readiness checks: `" + json.dumps(val["readiness_checks"]) + "`. In addition to defensive "
                 "tolerances, require rotation/usage not to regress and offense/pace validation MAE within10% "
                 "of the prior validation baseline. These practical allowances are fixed before final testing.", "",
                 "Remaining limitations: roster-conditional season aggregates, unknown entrant skill/availability, "
                 "defensive context and extreme cross-era extrapolation. A permits game-log noise calibration "
                 "only with these constraints; it does not validate ironman predictions. If defensive acceptance "
                 "fails, the next step is targeted calibration of extreme defensive buckets and prior context, "
                 "using validation only and a new genuinely unseen holdout before further model testing. "
                 "No historical game logs acquired; no UI/application changes.", ""]
    report = REPORT_DIR / "SIMULATION_ENGINE_DESIGN.md"
    original = report.read_text(encoding="utf-8").split("\n## Rotation Validation")[0]
    report.write_text(original + "\n".join(sections), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate", action="store_true")
    mode.add_argument("--final", action="store_true")
    mode.add_argument("--report", action="store_true")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        if args.validate:
            validation_stage()
        elif args.final:
            final_stage()
        else:
            render_report()
