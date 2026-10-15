"""Validation-only defensive continuity audit. Never evaluates the frozen final test."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from threadpoolctl import threadpool_limits

import improve_projection as prior
from simulation_engine_spike import ROOT, OUT_DIR, REPORT_DIR, metrics

SUMMARY = OUT_DIR / "defensive_continuity_v1_summary.json"
INDIVIDUAL = prior.DEFENSE_FIELDS + ["w_age", "center_minutes_share", "rotation_hhi"]
CONTINUITY_COLUMNS = INDIVIDUAL + ["retained_defensive_context", "returning_minutes_share"]
PER75_COLUMNS = ["w_defensive_box_plus_minus", "w_steals_per_75", "w_blocks_per_75", "w_defensive_rebounds_per_75",
                 "w_age", "center_minutes_share", "rotation_hhi", "retained_defensive_context", "returning_minutes_share"]
SOURCE_FIELDS = ["defensive_box_plus_minus", "steals_per_75", "blocks_per_75", "defensive_rebounds_per_75"]
PROTECTED = [prior.BUNDLE, prior.VALIDATION, prior.PROTOCOL, prior.FINAL, prior.LEDGER,
             Path(str(prior.PREFIX) + "_final_predictions.csv"),
             Path(str(prior.PREFIX) + "_validation_features.csv"),
             Path(str(prior.PREFIX) + "_validation_rotation.csv")]
OFFICIAL_TEST = dict(ortg=dict(mae=2.362, rmse=2.964, r2=.394),
                     drtg=dict(mae=2.316, rmse=2.779, r2=.322),
                     pace=dict(mae=1.668, rmse=2.052, r2=-.062))


def hashes():
    # Hash bytes only: frozen individual test outcomes/errors are never parsed.
    return {str(path.relative_to(ROOT)): prior.fingerprint(path) for path in PROTECTED}


def require_development(frame):
    if frame.season_end_year.min() < 1981 or frame.season_end_year.max() > 2019:
        raise ValueError("Only 1980-81 through 2018-19 development targets are allowed")


def continuity(minutes, returning, previous_environment, synthetic=False):
    minutes, returning = np.asarray(minutes, float), np.asarray(returning, bool)
    if not np.isclose(minutes.sum(), 240.) or len(minutes) != len(returning):
        raise ValueError("Continuity requires the unchanged 240-minute named-player rotation")
    if synthetic:
        returning = np.zeros(len(minutes), bool)
    order = np.argsort(-minutes, kind="stable")[:5]
    share = float(np.dot(minutes / 240., returning))
    return dict(returning_minutes_share=share,
                returning_top5_share=float(np.dot(minutes[order], returning[order]) / minutes[order].sum()),
                returning_rotation_players=int(np.sum(returning & (minutes >= 8.))),
                transfer_or_unknown_minutes_share=1. - share,
                retained_defensive_context=share * float(previous_environment) if share else 0.)


def read_development(name):
    frame = pd.read_csv(ROOT / "data/historical/processed" / name)
    return frame[frame.season_end_year <= 2019].copy()


def build_continuity(base, state, old_summary):
    """Reproduce rotation from prior identities/history; current minutes only define membership."""
    players = read_development("player_seasons.csv")
    splits = read_development("player_team_splits.csv")
    teams, leagues = read_development("team_seasons.csv"), read_development("league_seasons.csv")
    removed = {(r["player_id"], r["season"], r["team"]) for r in old_summary["quarantine"]}
    valid = np.array([(r.player_id, r.season, r.team) not in removed for r in players.itertuples()])
    profiles = players[valid]
    histories = {pid: group.sort_values("season_end_year").to_dict("records") for pid, group in profiles.groupby("player_id")}
    columns = ["player_id", "player_name", "season_end_year", "team_id", "franchise_id", "minutes"]
    single = players[(players.team != "TOT") & ~players.team.astype(str).str.endswith("TM")]
    roster = pd.concat([single[columns], splits[columns]], ignore_index=True)
    roster = roster[roster.minutes > 0]
    membership = roster.groupby(["season_end_year", "player_id"]).franchise_id.agg(set).to_dict()
    groups = {(int(year), tid): group.drop_duplicates("player_id").reset_index(drop=True)
              for (year, tid), group in roster.groupby(["season_end_year", "team_id"])}
    leagues = leagues.set_index("season_end_year")
    teams["relative_defense"] = teams.def_rtg - teams.season_end_year.map(leagues.league_points_per_75) / 75 * 100
    if teams.duplicated(["season_end_year", "franchise_id"]).any():
        raise ValueError("Ambiguous prior franchise context")
    franchise_context = teams.set_index(["season_end_year", "franchise_id"]).relative_defense.to_dict()
    target_franchise = teams.set_index(["season_end_year", "team_id"]).franchise_id.to_dict()
    schedules = teams.assign(games=teams.wins + teams.losses).groupby("season_end_year").games.max().to_dict()
    env = roster.merge(teams[["season_end_year", "team_id", "relative_defense"]],
                       on=["season_end_year", "team_id"], how="left", validate="many_to_one")
    env["weighted"] = env.relative_defense * env.minutes
    totals = env.groupby(["season_end_year", "player_id"])[["weighted", "minutes"]].sum()
    environment = totals.weighted.div(totals.minutes).to_dict()
    largest = roster.sort_values("minutes", ascending=False).drop_duplicates(["season_end_year", "player_id"])
    origin = largest.set_index(["season_end_year", "player_id"]).franchise_id.to_dict()
    source_path = ROOT / "data/analysis/simulation_defensive_source_fields.csv"
    if source_path.exists():
        source = pd.read_csv(source_path)
        source = source[source.season_end_year <= 2019]
    else:
        # Cache-only ingestion; missing HTML fails rather than fetching anything.
        source = prior.previous.export_cached_profiles(players.season_end_year.unique())
    available = source.set_index(["player_id", "season_end_year", "team"])[["defensive_box_plus_minus", "steal_pct", "block_pct", "defensive_rebound_pct"]].notna().to_dict("index")
    for row in profiles.itertuples():
        key = (row.player_id, int(row.season_end_year), row.team)
        evidence = available.setdefault(key, {})
        evidence["steals_per_75"] = pd.notna(row.steals_per_75)
        evidence["blocks_per_75"] = pd.notna(row.blocks_per_75)
        evidence["defensive_rebounds_per_75"] = pd.notna(row.defensive_rebounds) and row.estimated_possessions_minutes_team_pace > 0
    known_minutes = pd.read_csv(Path(str(prior.PREFIX) + "_validation_rotation.csv"))
    validation_minutes = known_minutes.set_index(["season", "team", "player"]).projected_mpg.to_dict()
    rows, minute_errors = [], []
    for index, team in enumerate(base.itertuples()):
        if index % 300 == 0:
            print(f"Continuity metadata: {index}/{len(base)} development teams", flush=True)
        year, tid = int(team.season_end_year), team.team_id
        group, records, past_rows = groups[(year, tid)], [], []
        for member in group.itertuples():
            history = histories.get(member.player_id, [])
            past = [row for row in history if year - 3 <= row["season_end_year"] < year]
            latest = past[-1] if past and past[-1]["minutes"] > 0 else None
            if latest:
                lag = year - latest["season_end_year"]
                positive = [row for row in reversed(past) if row["minutes"] > 0]
                history_mpg = float(np.average([row["minutes_per_game"] for row in positive], weights=.5 ** np.arange(len(positive))))
                mpg = latest["minutes_per_game"] / (1 + .15 * (lag - 1))
                availability = min(latest["games"] / schedules.get(latest["season_end_year"], 82.), 1.)
                age = latest["age"] + lag if pd.notna(latest["age"]) else 26.
                starts = latest["games_started"] / latest["games"] if pd.notna(latest["games_started"]) and latest["games"] > 0 else .5
                record = dict(prior_mpg=mpg, history_mpg=history_mpg, prior_availability=availability,
                              projected_age=age, prior_start_share=starts, known=1., returner=float(lag > 1))
            else:
                record = dict(prior_mpg=8., history_mpg=8., prior_availability=.5,
                              projected_age=26., prior_start_share=.25, known=0., returner=0.)
            records.append(record)
            past_rows.append(latest)
        rotation_inputs = pd.DataFrame(records)
        rotation_inputs["prior_rank"] = rotation_inputs.prior_mpg.rank(method="first", ascending=False)
        rotation_inputs["roster_depth"] = len(group)
        minutes = prior.projected_rotation(rotation_inputs, state["rotation"], state["rookie_priors"], state["rotation_model"])
        if not np.isclose(np.square(minutes / 240.).sum(), team.rotation_hhi, atol=1e-10):
            raise ValueError("Rotation reconstruction drifted from frozen feature matrix")
        if year >= 2015:
            for member, value in zip(group.itertuples(), minutes):
                minute_errors.append(abs(value - validation_minutes[(team.season, team.team_name, member.player_name)]))
        franchise = target_franchise[(year, tid)]
        returning = [franchise in membership.get((year - 1, pid), set()) for pid in group.player_id]
        context = franchise_context.get((year - 1, franchise), 0.)
        row = continuity(minutes, returning, context)
        row["previous_franchise_environment"] = context
        missing, contexts, origins = [], [], []
        for member, past in zip(group.itertuples(), past_rows):
            key = (member.player_id, int(past["season_end_year"]), past["team"]) if past else None
            evidence = available.get(key, {})
            missing.append(1 - sum(evidence.get(field, False) for field in SOURCE_FIELDS) / len(SOURCE_FIELDS))
            pkey = (int(past["season_end_year"]), member.player_id) if past else None
            contexts.append(environment.get(pkey, 0.))
            origins.append(origin.get(pkey, "unknown:" + member.player_id))
        weights = minutes / 240.
        context_mean = float(np.dot(weights, contexts))
        origin_weights = pd.Series(weights).groupby(origins).sum()
        row.update(defensive_missing_share=float(np.dot(weights, missing)),
                   source_context_sd=float(np.sqrt(np.dot(weights, np.square(np.asarray(contexts) - context_mean)))),
                   source_diversity=float(1 - np.square(origin_weights).sum()))
        rows.append(row)
    extra = pd.DataFrame(rows, index=base.index)
    return pd.concat([base, extra], axis=1), dict(max_validation_minutes_difference=max(minute_errors),
                                                rotation_configuration=state["rotation"],
                                                usage_configuration=state["usage"]), players, teams, leagues, available


def bucket_labels(frame, mode):
    if mode == "ranking":
        percentile = frame.groupby("season_end_year").actual_drtg.rank(method="average", pct=True)
        return pd.cut(percentile, [0, .2, .4, .6, .8, 1.], include_lowest=True,
                      labels=["elite", "above_average", "average", "below_average", "poor"])
    relative = frame.actual_drtg - frame.groupby("season_end_year").actual_drtg.transform("mean")
    return pd.cut(relative, [-np.inf, -3, -1, 1, 3, np.inf],
                  labels=["elite", "above_average", "average", "below_average", "poor"])


def diagnostics(frame, prediction, mode, box_threshold):
    require_development(frame)
    error = np.asarray(prediction) - frame.actual_drtg.to_numpy()
    work = frame.assign(error=error, bucket=bucket_labels(frame, mode))
    result = dict(overall=metrics(frame.actual_drtg, prediction), abs_error_p90=float(np.quantile(abs(error), .9)), buckets={})
    for label, group in work.groupby("bucket", observed=True):
        result["buckets"][str(label)] = dict(n=len(group), mae=float(group.error.abs().mean()),
                                             bias=float(group.error.mean()), median_bias=float(group.error.median()),
                                             p90_abs_error=float(group.error.abs().quantile(.9)),
                                             actual_residual=float((group.actual_drtg - group.anchor_drtg).mean()),
                                             predicted_residual=float((group.actual_drtg + group.error - group.anchor_drtg).mean()))
    result["suberas"] = {name: dict(mae=float(group.error.abs().mean()),
                                     anchor_mae=float((group.anchor_drtg - group.actual_drtg).abs().mean()))
                         for name, group in work.groupby(np.where(work.season_end_year <= 2016, "2015-2016", "2017-2019"))}
    box = work[work.w_defensive_box_plus_minus >= box_threshold]
    result["box_heavy"] = dict(n=len(box), bias=float(box.error.mean()) if len(box) else 0.)
    actual = frame.actual_drtg - frame.anchor_drtg
    predicted = np.asarray(prediction) - frame.anchor_drtg
    result["residual_spread"] = dict(actual_sd=float(actual.std()), predicted_sd=float(np.std(predicted, ddof=1)),
                                     spread_ratio=float(np.std(predicted, ddof=1) / actual.std()))
    return result


def fit_linear(train, columns, ridge=False):
    require_development(train)
    if train.season_end_year.max() > 2014:
        raise ValueError("All coefficients must be calibrated on TRAIN only")
    estimator = prior.previous.pipeline(Ridge(alpha=30.) if ridge else LinearRegression())
    estimator.fit(train[columns], train.actual_drtg - train.anchor_drtg)
    return dict(candidate=(estimator, columns), tail=None)


def predict(frame, model):
    result = prior.previous.predict(frame, "drtg", model["candidate"])
    if model["tail"] is not None:
        residual = result - frame.anchor_drtg.to_numpy()
        features = np.column_stack([residual, np.maximum(residual - model["knot"], 0.)])
        result = result + np.clip(model["tail"].predict(features), -2., 2.)
    return result


def fit_tail(train, columns=CONTINUITY_COLUMNS):
    """A bounded hinge correction fitted only on forward predictions within TRAIN."""
    require_development(train)
    if train.season_end_year.max() > 2014:
        raise ValueError("Tail coefficients must be calibrated on TRAIN only")
    predicted, actual = [], []
    for end in [1994, 1999, 2004, 2009]:
        earlier = train[train.season_end_year <= end]
        later = train[(train.season_end_year > end) & (train.season_end_year <= end + 5)]
        fitted = fit_linear(earlier, columns, ridge=True)
        predicted.extend(predict(later, fitted) - later.anchor_drtg)
        actual.extend(later.actual_drtg - later.anchor_drtg)
    predicted, actual = np.asarray(predicted), np.asarray(actual)
    knot = float(np.quantile(predicted, .75))
    features = np.column_stack([predicted, np.maximum(predicted - knot, 0.)])
    tail = Ridge(alpha=30.).fit(features, actual - predicted)
    result = fit_linear(train, columns, ridge=True)
    result.update(tail=tail, knot=knot, training_forward_rows=len(predicted), correction_limit=2.)
    return result


def assess(frame, model, limits, threshold):
    prediction = predict(frame, model)
    result = {}
    for view in ["legacy", "ranking"]:
        diagnostic = diagnostics(frame, prediction, view, threshold)
        result[view] = dict(diagnostics=diagnostic, **prior.accepted_defense(diagnostic, limits))
    result["accepted"] = all(result[view]["accepted"] for view in ["legacy", "ranking"])
    return result


def domain_reference(train, columns=CONTINUITY_COLUMNS):
    fields = columns + ["defensive_missing_share", "source_context_sd", "source_diversity"]
    return {field: dict(min=float(train[field].min()), max=float(train[field].max()),
                        p01=float(train[field].quantile(.01)), p99=float(train[field].quantile(.99)),
                        scale=max(float(train[field].quantile(.75) - train[field].quantile(.25)),
                                  .01 * float(train[field].max() - train[field].min()), 1e-6)) for field in fields}


def uncertainty(features, domain, thresholds):
    coverage = prior.extrapolation(features, domain)
    c, missing, sd = features["returning_minutes_share"], features["defensive_missing_share"], features["source_context_sd"]
    diversity = features["source_diversity"]
    flags = dict(low_continuity=c < .35, weak_availability=missing > .20,
                 diverse_contexts=sd > thresholds["context_sd_p90"] and diversity > thresholds["diversity_p75"],
                 extrapolation=coverage["risk"] == "high")
    moderate = c < .65 or missing > .05 or sd > thresholds["context_sd_p75"] or coverage["fraction_outside"] > 0
    category = "HIGH" if any(flags.values()) else "MODERATE" if moderate else "LOW"
    return dict(category=category, high_flags=flags, continuity=c, missing_share=missing,
                source_context_sd=sd, source_diversity=diversity, coverage=coverage,
                interpretation="Uncalibrated uncertainty/coverage indicator, not defensive quality or a probability interval")


def root_cause(train, val, baseline, limits, threshold):
    selected = baseline["candidate"]
    equation = prior.previous.equation_parameters(selected)
    labels = bucket_labels(val, "legacy")
    contributions = []
    for label, group in val.groupby(labels, observed=True):
        row = dict(bucket=str(label), n=len(group))
        for field, coefficient in equation["coefficients"].items():
            row[field] = float(coefficient * (group[field].mean() - train[field].mean()))
        contributions.append(row)
    ablations = {"without_DBPM": [f for f in selected[1] if f != "w_defensive_box_plus_minus"],
                 "without_context": [f for f in selected[1] if f != "prior_defensive_environment"],
                 "without_rotation": [f for f in selected[1] if f != "rotation_hhi"],
                 "without_box_rates": [f for f in selected[1] if f not in {"w_steal_pct", "w_block_pct", "w_defensive_rebound_pct"}],
                 "with_position": selected[1] + ["center_minutes_share"],
                 "per75_instead_of_percentages": [f for f in selected[1] if f not in {"w_steal_pct", "w_block_pct", "w_defensive_rebound_pct"}] +
                                                   ["w_steals_per_75", "w_blocks_per_75", "w_defensive_rebounds_per_75"]}
    scores = {}
    for name, fields in ablations.items():
        candidate = fit_linear(train, fields, ridge=True)
        d = diagnostics(val, predict(val, candidate), "legacy", threshold)
        scores[name] = dict(overall_mae=d["overall"]["mae"], poor_bias=d["buckets"]["poor"]["bias"],
                            poor_mae=d["buckets"]["poor"]["mae"])
    prediction = predict(val, baseline)
    strata = []
    for name, group in val.assign(error=prediction - val.actual_drtg).groupby(
            pd.cut(val.returning_minutes_share, [-.01, .35, .65, 1.01], labels=["low", "moderate", "high"]), observed=True):
        strata.append(dict(continuity=str(name), n=len(group), mae=float(group.error.abs().mean()), bias=float(group.error.mean())))
    missing = val.assign(bucket=labels).groupby("bucket", observed=True)[["defensive_missing_share", "returning_minutes_share", "prior_defensive_environment"]].mean().reset_index()
    return dict(centered_linear_contributions=contributions, train_fitted_ablations=scores,
                baseline_by_continuity=strata, missing_and_context_by_bucket=missing.to_dict("records"),
                interpretation="Residual spread and conditional means diagnose compression; ablations identify model dependence, not causal player/coaching effects. Tail buckets condition on outcomes, so some mean regression is expected.")


def custom_diagnostic(old, selected, domain, thresholds, players, teams, leagues, available):
    features = {row["feature"]: row["value"] for row in old["custom_lineup"]["extrapolation"]["features"]}
    rows, environments, contexts, missing = [], [], [], []
    for _, name, season in prior.CUSTOM_TEAM:
        player = players[(players.player_name == name) & (players.season == season)].iloc[0]
        evidence = dict(player=name, season=season, minutes=48.)
        source = available.get((player.player_id, int(player.season_end_year), player.team), {})
        missing.append(1 - sum(source.get(field, False) for field in SOURCE_FIELDS) / len(SOURCE_FIELDS))
        evidence["defensive_fields_available"] = sum(source.get(field, False) for field in SOURCE_FIELDS)
        # Existing custom feature audit already contains actual sourced weighted evidence.
        team = teams[(teams.season_end_year == player.season_end_year) & (teams.team_id == player.team_id)].iloc[0]
        baseline = leagues.loc[int(player.season_end_year)].league_points_per_75 / 75 * 100
        environments.append(float(team.def_rtg - baseline))
        contexts.append((int(player.season_end_year), player.franchise_id))
        rows.append(evidence)
    features.update(continuity(np.full(5, 48.), np.ones(5, bool), 999., synthetic=True))
    features["defensive_missing_share"] = float(np.mean(missing))
    features["source_context_sd"] = float(np.std(environments))
    features["source_diversity"] = 1 - sum((contexts.count(value) / 5) ** 2 for value in set(contexts))
    features["anchor_drtg"] = float(leagues.loc[2016].league_points_per_75 / 75 * 100)
    projection = float(predict(pd.DataFrame([features]), selected)[0])
    return dict(players=rows, continuity=0., retained_defensive_context=0., projected_drtg=projection,
                individual_evidence={field: features[field] for field in selected["candidate"][1]
                                     if field not in {"retained_defensive_context", "returning_minutes_share"}},
                uncertainty=uncertainty(features, domain, thresholds),
                full_lineup_extrapolation=old["custom_lineup"]["extrapolation"]["risk"],
                offense_pace_unchanged={key: old["custom_lineup"]["projections"][key] for key in ["ortg", "pace"]},
                semantics="Five-player ironman only, 48 minutes each, 240 total; DRtg is a diagnostic estimate, not precise truth or a game simulation")


def run():
    protected = hashes()
    old = json.loads(prior.VALIDATION.read_text(encoding="utf-8"))
    state = joblib.load(prior.BUNDLE)
    base = pd.read_csv(Path(str(prior.PREFIX) + "_validation_features.csv"))
    require_development(base)
    if state["rotation"] != "balanced_prior":
        raise ValueError("Unexpected frozen rotation; do not silently substitute a new one")
    frame, regression, players, teams, leagues, available = build_continuity(base, state, old)
    train, val = frame[frame.season_end_year <= 2014], frame[frame.season_end_year >= 2015]
    limits = old["defense_acceptance"]["tolerances"]
    threshold = old["defense_acceptance"]["box_threshold"]
    representations = {name: fit_linear(train, fields, ridge=True) for name, fields in
                       {"percentages": CONTINUITY_COLUMNS, "per75": PER75_COLUMNS}.items()}
    representation_scores = {name: assess(val, model, limits, threshold) for name, model in representations.items()}
    representation = min(representation_scores, key=lambda name: representation_scores[name]["legacy"]["diagnostics"]["overall"]["mae"])
    columns = representations[representation]["candidate"][1]
    candidates = dict(A_current=dict(candidate=state["models"]["drtg"][state["selected_models"]["drtg"]], tail=None),
                      B_continuity_additive=fit_linear(train, columns),
                      C_continuity_ridge=representations[representation],
                      D_bounded_tail=fit_tail(train, columns))
    results = {name: assess(val, model, limits, threshold) for name, model in candidates.items()}
    linear_best = min(["B_continuity_additive", "C_continuity_ridge"], key=lambda name: results[name]["legacy"]["diagnostics"]["overall"]["mae"])
    reference = results[linear_best]["legacy"]["diagnostics"]
    tail = results["D_bounded_tail"]["legacy"]["diagnostics"]
    tail_supported = results["D_bounded_tail"]["accepted"] and (
        tail["overall"]["mae"] <= reference["overall"]["mae"] - .03 or
        (abs(tail["buckets"]["poor"]["bias"]) <= abs(reference["buckets"]["poor"]["bias"]) - .25 and
         tail["overall"]["mae"] <= reference["overall"]["mae"] + .02))
    eligible = [name for name in ["B_continuity_additive", "C_continuity_ridge"] if results[name]["accepted"]]
    if tail_supported:
        eligible.append("D_bounded_tail")
    if eligible:
        selected = min(eligible, key=lambda name: results[name]["legacy"]["diagnostics"]["overall"]["mae"])
        if "B_continuity_additive" in eligible and results["B_continuity_additive"]["legacy"]["diagnostics"]["overall"]["mae"] <= results[selected]["legacy"]["diagnostics"]["overall"]["mae"] + .05:
            selected = "B_continuity_additive"
    else:
        selected = linear_best
        if results["B_continuity_additive"]["legacy"]["diagnostics"]["overall"]["mae"] <= results[selected]["legacy"]["diagnostics"]["overall"]["mae"] + .05:
            selected = "B_continuity_additive"
    for target in ["ortg", "pace"]:
        scores = metrics(val["actual_" + target], prior.previous.predict(val, target, state["models"][target][state["selected_models"][target]]))
        expected = old["selected_validation"][target]
        if not all(np.isclose(scores[key], expected[key], atol=1e-10) for key in expected):
            raise ValueError("Frozen offense/pace regression detected")
        regression[target] = dict(unchanged=True, validation=scores)
    domain = domain_reference(train, columns)
    thresholds = dict(context_sd_p75=float(train.source_context_sd.quantile(.75)),
                      context_sd_p90=float(train.source_context_sd.quantile(.9)),
                      diversity_p75=float(train.source_diversity.quantile(.75)),
                      high_continuity_below=.35, moderate_continuity_below=.65,
                      high_missing_above=.2, moderate_missing_above=.05)
    named = frame[((frame.season == "2007-08") & (frame.team_name == "Boston Celtics")) |
                  ((frame.season == "2010-11") & (frame.team_name == "Miami Heat")) |
                  ((frame.season == "2016-17") & (frame.team_name == "Golden State Warriors"))]
    examples = pd.concat([named, val.nlargest(5, "returning_minutes_share"), val.nsmallest(5, "returning_minutes_share")]).drop_duplicates(["season", "team_id"])
    example_columns = ["season", "team_name", "returning_minutes_share", "returning_top5_share", "returning_rotation_players", "transfer_or_unknown_minutes_share", "retained_defensive_context", "actual_drtg"]
    examples = examples[example_columns].copy()
    example_source = examples[["season", "team_name"]].merge(frame, on=["season", "team_name"], validate="one_to_one", sort=False)
    examples["predicted_drtg"] = predict(example_source, candidates[selected])
    accepted = results[selected]["accepted"]
    result = dict(population=dict(train=len(train), validation=len(val), frozen_test_evaluations=0),
                  selected=selected, candidates=results, tolerances=limits,
                  evidence_representation=representation, representation_comparison=representation_scores,
                  tail_supported=bool(tail_supported), root_cause=root_cause(train, val, candidates["A_current"], limits, threshold),
                  tail_parameters=dict(knot=candidates["D_bounded_tail"]["knot"],
                                       intercept=float(candidates["D_bounded_tail"]["tail"].intercept_),
                                       coefficients=candidates["D_bounded_tail"]["tail"].coef_.tolist(),
                                       forward_training_rows=candidates["D_bounded_tail"]["training_forward_rows"],
                                       correction_bounds=[-2., 2.]),
                  examples=examples.to_dict("records"), regression_checks=regression,
                  uncertainty_thresholds=thresholds,
                  custom_lineup=custom_diagnostic(old, candidates[selected], domain, thresholds, players, teams, leagues, available),
                  selected_equation=prior.previous.equation_parameters(candidates[selected]["candidate"]),
                  official_test_unchanged=OFFICIAL_TEST,
                  verdict="A. READY FOR GAME-LOG CALIBRATION" if accepted else "B. DEFENSIVE MODEL STILL OUTSIDE ACCEPTANCE TOLERANCE",
                  protected_hashes=protected)
    if protected != hashes():
        raise ValueError("Protected original models/results were modified")
    result["protected_files_unchanged"] = True
    prior.write_json(SUMMARY, result)
    frame.to_csv(OUT_DIR / "defensive_continuity_v1_features.csv", index=False)
    predictions = frame[["season", "season_end_year", "team_id", "team_name", "actual_drtg", "anchor_drtg"]].copy()
    for name, candidate in candidates.items():
        predictions[name] = predict(frame, candidate)
    predictions.to_csv(OUT_DIR / "defensive_continuity_v1_predictions.csv", index=False)
    joblib.dump(dict(model=candidates[selected], name=selected, domain=domain, uncertainty_thresholds=thresholds,
                     accepted=accepted, verdict=result["verdict"], evaluation_scope="training/validation only; no new final-test result",
                     continuity_rule="returning projected minutes share * prior same-franchise DRtg residual; synthetic continuity=0",
                     protected_baseline_hashes=protected), ROOT / "artifacts/simulation/defensive_continuity_v1.joblib")
    render(result)
    print(json.dumps(prior.previous.json_clean(dict(selected=selected, validation=results[selected],
                     custom_drtg=result["custom_lineup"]["projected_drtg"], verdict=result["verdict"])), indent=2))


def render(result):
    table = prior.previous.table
    lines = ["\n## Defensive Continuity Model", "",
             "Reproduce with `python scripts/simulation/defensive_continuity.py`. This task uses only TRAIN "
             "1980-81 through 2013-14 and VALIDATION 2014-15 through 2018-19. No final-test model evaluation "
             "is exposed by this script. Existing final artifacts are hashed as bytes only and remain unchanged. "
             "The official frozen benchmark remains ORtg MAE2.362/RMSE2.964/R2 .394; DRtg2.316/2.779/.322; "
             "pace1.668/2.052/-.062. It describes the previous model, NOT an evaluation of this new defense.", "",
             "Artifacts: `data/simulation/defensive_continuity_v1_summary.json`, development features and "
             "predictions CSVs; new fitted defensive artifact under ignored `artifacts/simulation/`. "
             "Offense/pace/usage/rotation artifacts are not changed or refitted.", "",
             "C=sum(projected minutes of players on the same franchise last season)/240. Franchise IDs preserve "
             "continuity through relocation/name changes. Also audit returning top-five minute share and the number "
             "of returning rotation players (projected minutes>=8). Source membership is known retrospectively, "
             "but target minutes/starts/performance never determine predictors. Selected projected rotations "
             "are reproduced exactly from frozen rules; current minutes define roster membership only.", "",
             "Retained context=C*(previous same-franchise DRtg - previous league scoring anchor). Transferred "
             "players carry ZERO direct previous-team context. A returning player's multi-team total is not used "
             "to import unrelated team contexts. Individual evidence, age/position mix and proposed rotation "
             "remain separate. This is a fitted context adjustment, not copying previous-team DRtg. DWS stays excluded.", "",
             "Candidates: A frozen current additive model; B continuity OLS; C continuity Ridge(alpha30); "
             "D one bounded hinge correction to C. The hinge is fitted on forward predictions entirely within "
             "TRAIN (1995-99/2000-04/2005-09/2010-14, each trained on earlier years); knot=training prediction75th "
             "percentile, Ridge30 correction clipped to[-2,2] DRtg. D is eligible only if all gates pass and "
             "validation MAE improves>=.03 or poor bias improves>=.25 with MAE regression<=.02. B is preferred "
             "within.05 MAE of another passing candidate. A is a diagnostic baseline, not eligible because it "
             "does not enforce synthetic-context semantics. All coefficients/imputation fit TRAIN only.", "",
             "Selected: **" + result["selected"] + "**. Exact raw equation, intercept and imputation medians "
             "are in `selected_equation` in the summary JSON. If all candidates fail, the simpler additive "
             "equation is also preferred within.05 MAE; this is a research diagnostic, NOT an accepted "
             "replacement for the existing engine. Acceptance status is stored in the new artifact.", "",
             "Tail correction parameters (training-only): `" + json.dumps(result["tail_parameters"]) + "`", "",
             "## Poor-Defense Bias Analysis", ""]
    lines += ["Individual evidence representation: **" + result["evidence_representation"] + "**. "
              "Compare percentage rates with prior per75 rates using the same continuity Ridge30 equation and "
              "select lower validation MAE. Per75 substitutes steals, blocks and defensive rebounds per75 for "
              "STL%/BLK%/DRB%; DBPM, age, position and rotation are preserved. No target rates are used. "
              "Per75 depends on estimated prior team pace and is not context-free talent.", "",
              table([dict(representation=name, **views["legacy"]["diagnostics"]["overall"],
                          poor_bias=views["legacy"]["diagnostics"]["buckets"]["poor"]["bias"])
                     for name, views in result["representation_comparison"].items()]), ""]
    original = result["candidates"]["A_current"]["legacy"]["diagnostics"]
    poor = original["buckets"]["poor"]
    lines += [f"The observed baseline failure is compressed conditional tail predictions: residual SD "
              f"{original['residual_spread']['predicted_sd']:.3f} versus actual "
              f"{original['residual_spread']['actual_sd']:.3f}. Poor defenses average "
              f"{poor['actual_residual']:.3f} DRtg above the prior league anchor, but predictions average only "
              f"{poor['predicted_residual']:.3f}. This creates the {poor['bias']:.3f} signed bias. "
              "Continuity masking removes an invalid synthetic-context assumption but is not, by itself, "
              "a demonstrated cure for this bias. OLS comparison distinguishes this compression from Ridge "
              "shrinkage alone. Ablations, missingness and position/rate substitutions below show which "
              "explanations have empirical support; they do not isolate unobserved coaching or scheme effects.", ""]
    lines += [result["root_cause"]["interpretation"], "",
              "Training-fitted feature ablations (diagnostics only, not extra selection candidates):", "",
              table([dict(ablation=name, **row) for name, row in result["root_cause"]["train_fitted_ablations"].items()]), "",
              "Centered contributions of the frozen current equation by original bucket:", "",
              table(result["root_cause"]["centered_linear_contributions"]), "",
              "Missingness, continuity and old inherited context:", "", table(result["root_cause"]["missing_and_context_by_bucket"]), "",
              "Baseline error by continuity:", "", table(result["root_cause"]["baseline_by_continuity"]), "",
              "Box-score measures do not observe schemes, matchup assignments, rim deterrence, defensive "
              "communication or coaching. Strong rate persistence does not make these complete defense. "
              "The earlier qualified-transfer correlations were DBPM .600, STL% .740, BLK% .874, DRB% .876, "
              "DWS .459 and defensive environment .049. Missing source values are documented separately "
              "from imputed model values; no new defensive statistics are fabricated.", "",
              "## Defensive Tail Validation", "",
              "Retain the exact original thresholds (including unrounded overall bound), and preserve the original "
              "era-relative difference buckets for acceptance comparability. ALSO report season-relative DRtg "
              "quintile rankings as requested. A requires all gates in BOTH views: the previous19-team poor bucket "
              "cannot be hidden by redefining it. Positive error underrates defense; negative error overrates it.", "",
              "Unchanged criteria: `" + json.dumps(result["tolerances"]) + "`", ""]
    rows = []
    for name, views in result["candidates"].items():
        for view in ["legacy", "ranking"]:
            d = views[view]["diagnostics"]
            rows.append(dict(candidate=name, view=view, **d["overall"], p90=d["abs_error_p90"],
                             actual_sd=d["residual_spread"]["actual_sd"], predicted_sd=d["residual_spread"]["predicted_sd"], accepted=views[view]["accepted"]))
    lines += [table(rows), "", "Bucket details:", "",
              table([dict(candidate=name, view=view, bucket=bucket, **row)
                     for name, views in result["candidates"].items() for view in ["legacy", "ranking"]
                     for bucket, row in views[view]["diagnostics"]["buckets"].items()]), "",
              "Acceptance checks:", "", table([dict(candidate=name, view=view, **views[view]["checks"])
                     for name, views in result["candidates"].items() for view in ["legacy", "ranking"]]), "",
              "Sub-era and box-heavy diagnostics:", "", table([dict(candidate=name, view=view,
                     suberas=json.dumps(views[view]["diagnostics"]["suberas"]), box_heavy=json.dumps(views[view]["diagnostics"]["box_heavy"]))
                     for name, views in result["candidates"].items() for view in ["legacy", "ranking"]]), "",
              "High/low continuity and named development sanity checks (major star arrivals do not necessarily "
              "mean low overall continuity):", "", table(result["examples"]), "",
              "2019-20 Lakers are in the frozen test period: no new inference or individual-error analysis was "
              "performed. Previously recorded context remains in the earlier experiment only.", "",
              "## Synthetic-Roster Defensive Semantics", "",
              "A completely synthetic roster has continuity=0 and retained context=0, even if selected players "
              "come from historically excellent defenses or several share a franchise name. Predictions use "
              "individual transferable evidence, proposed minutes, age/position mix and target-era league anchor. "
              "No previous NBA team's defensive environment is inherited. Historical context dispersion may "
              "raise uncertainty but does not increase predicted defensive quality.", "",
              result["custom_lineup"]["semantics"], "", table(result["custom_lineup"]["players"]), "",
              "Custom DRtg diagnostic: `" + str(result["custom_lineup"]["projected_drtg"]) + "`; continuity0, retained context0. "
              "Do not interpret this number as precise truth. Individual evidence: `" + json.dumps(result["custom_lineup"]["individual_evidence"]) + "`", "",
              "Unchanged custom offense/pace: `" + json.dumps(result["custom_lineup"]["offense_pace_unchanged"]) + "`", "",
              "## Defensive Uncertainty", "",
              "This is a coverage/uncertainty indicator, not quality or a calibrated interval. HIGH if continuity<.35, "
              "missing weighted defensive fields>.20, context SD>training p90 AND source diversity>training p75, "
              "or feature extrapolation is high (maximum beyond training range>1 IQR or >15% fields outside). "
              "Otherwise MODERATE if continuity<.65, missing>.05, context SD>training p75 or any feature outside "
              "training range; otherwise LOW. Source diversity=1-sum(source-season/franchise minute shares^2). "
              "Low continuity therefore always raises uncertainty; weak/missing evidence, distinct contexts and "
              "extrapolation are explicitly visible. Marginal feature ranges cannot establish joint causal support.", "",
              "Training-derived context thresholds: `" + json.dumps(result["uncertainty_thresholds"]) + "`", "",
              "Custom uncertainty: `" + json.dumps({k: v for k, v in result["custom_lineup"]["uncertainty"].items() if k != "coverage"}) + "`", "",
              table(result["custom_lineup"]["uncertainty"]["coverage"]["features"]), "",
              "Regression checks: `" + json.dumps(result["regression_checks"]) + "`. Protected baseline artifacts "
              "and official test records unchanged: `" + str(result["protected_files_unchanged"]) + "`. "
              "New final-test evaluations:0. Previous experiments remain above.", "",
              "**" + result["verdict"] + "**.", "",
              "If accepted, next task is team game-log acquisition and noise/home-court calibration with frozen "
              "team-strength methodology; current official test does not validate this new defense. Otherwise "
              "retain the failed tolerances explicitly and do not start game-log calibration. No game logs, "
              "Simulator UI, frontend or .NET application changes were made.", ""]
    path = REPORT_DIR / "SIMULATION_ENGINE_DESIGN.md"
    original = path.read_text(encoding="utf-8").split("\n## Defensive Continuity Model")[0]
    path.write_text(original + "\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        if args.report_only:
            render(json.loads(SUMMARY.read_text(encoding="utf-8")))
        else:
            run()
