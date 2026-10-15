"""Lagged team-strength experiment. Offline; no game logs or application changes.

Selection uses validation only. Test is evaluated once after models are frozen.
Same-season roster identity is allowed; all performance comes from older profiles.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from simulation_engine_spike import (
    ROOT, OUT_DIR, REPORT_DIR, CUSTOM_TEAM, FEATURES, load_inputs, roster_rows,
    add_derived_player_features, metrics, temporal_split,
)

sys.path.insert(0, str(ROOT / "scripts" / "historical"))
from export_cached_defensive_profiles import export_cached_profiles, DEFENSIVE_FIELDS

RECONSTRUCTION_FEATURE_SET = list(FEATURES)
PROFILE_FIELDS = [
    "relative_ts", "assist_pct", "turnover_pct", "usage_pct",
    "three_point_attempt_rate", "free_throw_rate", "offensive_rebounds_per_75",
    "defensive_rebounds_per_75", "steals_per_75", "blocks_per_75",
    "defensive_box_plus_minus", "steal_pct", "block_pct", "defensive_rebound_pct",
]
DEPLOYABLE_FEATURE_SET = [
    *("w_" + field for field in PROFILE_FIELDS),
    "usage_weighted_relative_ts", "usage_weighted_3par", "usage_weighted_ftr",
    "usage_compression", "usage_concentration", "rotation_hhi", "prior_pace_delta",
    "known_prior_minutes_share", "returner_minutes_share", "prior_age_years",
]
TARGETS = {"ortg": "off_rtg", "drtg": "def_rtg", "pace": "pace"}
FORMULA_FIELDS = {
    "ortg": ["usage_weighted_relative_ts", "usage_weighted_3par", "usage_weighted_ftr",
             "w_assist_pct", "w_turnover_pct", "w_offensive_rebounds_per_75", "usage_compression"],
    "drtg": ["w_defensive_box_plus_minus", "w_steal_pct", "w_block_pct", "w_defensive_rebound_pct"],
    "pace": ["prior_pace_delta", "w_three_point_attempt_rate", "w_assist_pct", "rotation_hhi"],
}
MIN_COVERAGE = .8
USAGE_EXPONENT = .5
MODEL_NAMES = ["league_anchor", "transparent_formula", "linear", "ridge", "nonlinear"]


def rotation(mpg):
    """Rank prior MPG as roles; top five 24-40, next five 6-26; no hidden players."""
    mpg = np.asarray(mpg, dtype=float)
    if len(mpg) < 5 or not np.isfinite(mpg).all():
        raise ValueError("At least five finite prior roles required")
    if len(mpg) == 5:
        return np.full(5, 48.)
    rank = np.argsort(-mpg, kind="stable")
    floor = np.zeros(len(mpg))
    cap = np.zeros(len(mpg))
    floor[rank[:5]], cap[rank[:5]] = 24., 40.
    floor[rank[5:10]], cap[rank[5:10]] = 6., 26.
    # Six-player edge case: capacity must allow an explicitly short/ironman roster.
    if cap.sum() < 240:
        cap[:] = 48.
    result = floor.copy()
    preference = np.maximum(mpg, 1.)
    for _ in range(100):
        remaining = 240. - result.sum()
        if remaining < 1e-9:
            break
        active = result < cap - 1e-10
        if not active.any():
            raise ValueError("Insufficient rotation capacity")
        result[active] += np.minimum(
            cap[active] - result[active], remaining * preference[active] / preference[active].sum()
        )
    assert np.isclose(result.sum(), 240.) and (result <= cap + 1e-8).all()
    return result


def normalize_usage(usage, minutes, exponent=None):
    """Shrink preference, then enforce the exposure-weighted team usage constraint."""
    exposure = np.asarray(minutes, float) / 48.
    preference = np.clip(np.asarray(usage, float), 1., 60.) ** (USAGE_EXPONENT if exponent is None else exponent)
    effective = 100. * preference / np.dot(exposure, preference)
    share = exposure * effective / 100.
    assert np.isclose(share.sum(), 1.)
    compression = float(np.dot(exposure / 5., np.maximum(np.asarray(usage) - effective, 0.)))
    return effective, share, compression


def aggregate(profiles, minutes, context):
    """Only profiles, proposed minutes and league context enter this shared builder."""
    weights = np.asarray(minutes) / 240.
    features = {}
    for field in PROFILE_FIELDS:
        value = profiles[field].to_numpy(float)
        valid = np.isfinite(value)
        features["w_" + field] = float(np.average(value[valid], weights=weights[valid])) if weights[valid].sum() else np.nan
    effective, usage_share, compression = normalize_usage(profiles.usage_pct, minutes)
    for field, name in [("relative_ts", "relative_ts"), ("three_point_attempt_rate", "3par"), ("free_throw_rate", "ftr")]:
        features["usage_weighted_" + name] = float(np.dot(usage_share, profiles[field]))
    features.update(
        usage_compression=compression, usage_concentration=float(np.square(usage_share).sum()),
        rotation_hhi=float(np.square(weights).sum()),
        prior_pace_delta=float(np.dot(weights, profiles.team_pace - profiles.league_pace)),
        known_prior_minutes_share=float(np.dot(weights, profiles.known)),
        returner_minutes_share=float(np.dot(weights, profiles.returner)),
        prior_age_years=float(np.dot(weights, profiles.profile_lag)),
        anchor_ortg=float(context.league_points_per_75 / 75. * 100.),
        anchor_drtg=float(context.league_points_per_75 / 75. * 100.),
        anchor_pace=float(context.league_pace),
    )
    return features, effective, usage_share


def prepare_inputs():
    players, splits, teams, leagues = load_inputs()
    defensive = export_cached_profiles(players.season_end_year.unique())
    supplemental = ROOT / "data" / "analysis" / "simulation_defensive_source_fields.csv"
    supplemental.parent.mkdir(parents=True, exist_ok=True)
    defensive.to_csv(supplemental, index=False)
    profiles = add_derived_player_features(players).merge(
        defensive, on=["player_id", "season_end_year", "team"], how="left", validate="one_to_one"
    )
    mismatch = profiles.source_minutes.isna() | ((profiles.source_minutes - profiles.minutes).abs() > 1.)
    identity_problems = profiles.loc[mismatch, ["player_id", "player_name", "season", "team", "minutes", "source_minutes"]].to_dict("records")
    profiles = profiles[~mismatch].copy()
    # Fixed training-only role priors. Null defensive source values stay null until model imputation.
    training = profiles[(profiles.season_end_year <= 2013) & (profiles.minutes >= 100)]
    defaults = training[PROFILE_FIELDS + ["team_pace", "league_pace"]].median()
    roles = {role: group[PROFILE_FIELDS + ["team_pace", "league_pace"]].median().fillna(defaults)
             for role, group in training.groupby("position")}
    histories = {pid: g.sort_values("season_end_year") for pid, g in profiles.groupby("player_id")}
    return players, roster_rows(players, splits), teams, leagues.set_index("season_end_year"), profiles, histories, defaults, roles, identity_problems


def prior_profiles(membership, year, histories, defaults, roles, max_lag):
    records = []
    for member in membership.itertuples():
        history = histories.get(member.player_id)
        index = int(np.searchsorted(history.season_end_year.to_numpy(), year)) - 1 if history is not None else -1
        past = history.iloc[index] if index >= 0 else None
        if past is not None and past.season_end_year >= year - max_lag and past.minutes > 0:
            row = past.to_dict()
            lag = year - int(row["season_end_year"])
            role_defaults = roles.get(str(row["position"]), defaults)
            for field in PROFILE_FIELDS:
                if pd.isna(row[field]):
                    row[field] = role_defaults[field]
            for field in ["team_pace", "league_pace"]:
                if pd.isna(row[field]):
                    row[field] = defaults[field]
            row["prior_mpg"] = float(row["minutes_per_game"]) / (1. + .15 * (lag - 1))
            row["known"], row["returner"], row["profile_lag"] = 1., float(lag > 1), float(lag)
        else:
            # No current-season position/starts/MPG used to infer an unknown entrant's role.
            row = defaults.to_dict()
            row["prior_mpg"], row["known"], row["returner"], row["profile_lag"] = 8., 0., 0., 0.
            row["season_end_year"], row["season"] = np.nan, None
            row["position"] = "unknown pooled prior"
        row["player_id"], row["player_name"] = member.player_id, member.player_name
        records.append(row)
    return pd.DataFrame(records).reset_index(drop=True)


def build_lagged(roster, teams, leagues, histories, defaults, roles, max_lag=3):
    output, audit = [], []
    groups = {(s, t): g.drop_duplicates("player_id") for (s, t), g in roster.groupby(["season_end_year", "team_id"])}
    for team in teams.itertuples():
        year = int(team.season_end_year)
        group = groups.get((year, team.team_id))
        if year - 1 not in leagues.index or group is None or len(group) < 5:
            continue
        profiles = prior_profiles(group, year, histories, defaults, roles, max_lag)
        deploy_minutes = rotation(profiles.prior_mpg)
        oracle_minutes = 240. * group.minutes.to_numpy(float) / group.minutes.sum()
        known_actual = float(np.dot(oracle_minutes / 240., profiles.known))
        for mode, minutes in [("oracle", oracle_minutes), ("deployable", deploy_minutes)]:
            features, _, _ = aggregate(profiles, minutes, leagues.loc[year - 1])
            row = dict(season=team.season, season_end_year=year, team_id=team.team_id,
                       team_name=team.team_name, mode=mode, **features,
                       actual_ortg=team.off_rtg, actual_drtg=team.def_rtg, actual_pace=team.pace,
                       known_actual_minutes_share=known_actual)
            output.append(row)
        audit.append(dict(season=team.season, team_name=team.team_name,
                          known_projected_minutes_share=output[-1]["known_prior_minutes_share"],
                          known_actual_minutes_share=known_actual,
                          roster_players=len(group), missing_profiles=int((profiles.known == 0).sum())))
    result = pd.DataFrame(output)
    # Use one membership/coverage criterion for both modes; oracle outcomes never gate eligibility.
    eligible = result[result["mode"] == "deployable"]
    keys = eligible[eligible.known_prior_minutes_share >= MIN_COVERAGE][["season", "team_id"]]
    return result.merge(keys, on=["season", "team_id"], validate="many_to_one"), pd.DataFrame(audit)


def calibrate_usage(roster, histories, defaults, roles):
    """Training outcomes calibrate a role parameter, never enter deployable inputs."""
    totals = {exponent: [0., 0.] for exponent in [.5, .75, 1.]}
    population = roster.query("1981 <= season_end_year <= 2014")
    for (year, _), group in population.groupby(["season_end_year", "team_id"]):
        group = group.drop_duplicates("player_id")
        if len(group) < 5 or group.usage_pct.isna().any():
            continue
        prior = prior_profiles(group, int(year), histories, defaults, roles, 3)
        minutes = rotation(prior.prior_mpg)
        if np.dot(minutes / 240., prior.known) < MIN_COVERAGE:
            continue
        target_usage, _, _ = normalize_usage(group.usage_pct, minutes, exponent=1.)
        for exponent in totals:
            estimated, _, _ = normalize_usage(prior.usage_pct, minutes, exponent)
            totals[exponent][0] += float(np.dot(minutes / 240., np.abs(target_usage - estimated)))
            totals[exponent][1] += 1
    scores = {str(exponent): dict(weighted_usage_mae=total / n, teams=int(n))
              for exponent, (total, n) in totals.items()}
    chosen = min(totals, key=lambda value: totals[value][0] / totals[value][1])
    return chosen, scores


def pipeline(model):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), model)


def fit_experiment(frame):
    """Fit residuals around preseason league anchors; fixed alpha, no test-time tuning."""
    splits = temporal_split(frame)
    results, fitted, selected = {}, {}, {}
    for target in TARGETS:
        actual, anchor = "actual_" + target, "anchor_" + target
        results[target], fitted[target] = {}, {}
        for name in MODEL_NAMES:
            if name == "nonlinear" and target != "ortg":
                continue
            columns = FORMULA_FIELDS[target] if name == "transparent_formula" else DEPLOYABLE_FEATURE_SET
            if name == "league_anchor":
                model = None
            else:
                regressor = (LinearRegression() if name in {"linear", "transparent_formula"}
                             else Ridge(alpha=30.) if name == "ridge"
                             else HistGradientBoostingRegressor(max_iter=100, max_leaf_nodes=7,
                                                               l2_regularization=10, random_state=7))
                model = pipeline(regressor)
                model.fit(splits["train"][columns], splits["train"][actual] - splits["train"][anchor])
            fitted[target][name] = (model, columns)
            val = predict(frame=splits["validation"], target=target, candidate=(model, columns))
            results[target][name] = {"validation": metrics(splits["validation"][actual], val)}
        selected[target] = min(results[target], key=lambda name: results[target][name]["validation"]["mae"])
        # All candidates frozen before final test evaluation; none is refit on validation or test.
        for name, candidate in fitted[target].items():
            test = predict(splits["test"], target, candidate)
            results[target][name]["test"] = metrics(splits["test"][actual], test)
    predictions = frame[["season", "season_end_year", "team_name", "mode", "known_prior_minutes_share",
                         "actual_ortg", "actual_drtg", "actual_pace"]].copy()
    for target in TARGETS:
        predictions["predicted_" + target] = predict(frame, target, fitted[target][selected[target]])
    return results, fitted, selected, predictions


def predict(frame, target, candidate):
    model, columns = candidate
    return frame["anchor_" + target].to_numpy() + (model.predict(frame[columns]) if model is not None else 0.)


def breakdown(predictions):
    result = {}
    test = predictions[predictions.season_end_year >= 2020]
    for label, group in predictions.groupby(pd.cut(predictions.season_end_year, [1979, 1989, 1999, 2009, 2014, 2019, 2022, 2026]), observed=True):
        if len(group):
            result[str(label)] = {"n": len(group), "split": "train/in-sample" if group.season_end_year.max() <= 2014 else "validation" if group.season_end_year.max() <= 2019 else "test",
                                  **{t: metrics(group["actual_" + t], group["predicted_" + t]) for t in TARGETS}}
    tiers = pd.cut(test.actual_ortg - test.actual_drtg, [-np.inf, -3, 3, np.inf], labels=["poor", "average", "elite"])
    quality = {str(label): {"n": len(g), **{t: metrics(g["actual_" + t], g["predicted_" + t]) for t in TARGETS}}
               for label, g in test.groupby(tiers, observed=True)}
    distribution = {t: {str(q): float((test["predicted_" + t] - test["actual_" + t]).abs().quantile(q))
                        for q in [.1, .5, .9, .95, 1.]} for t in TARGETS}
    return {"era": result, "test_quality_tier": quality, "test_absolute_error_quantiles": distribution}


def custom_profile(players, profiles, leagues, fitted, selected, validation_predictions):
    selected_rows = []
    for slot, name, season in CUSTOM_TEAM:
        row = profiles[(profiles.player_name == name) & (profiles.season == season)].iloc[0].copy()
        row["slot"], row["known"], row["returner"], row["profile_lag"] = slot, 1., 0., 1.
        selected_rows.append(row)
    selected_profiles = pd.DataFrame(selected_rows).reset_index(drop=True)
    minutes = rotation(selected_profiles.minutes_per_game)
    # Same prior-era anchor as a 2016-17 preseason forecast. No target-year outcome required.
    features, effective, share = aggregate(selected_profiles, minutes, leagues.loc[2016])
    frame = pd.DataFrame([features])
    projection = {target: float(predict(frame, target, fitted[target][selected[target]])[0]) for target in TARGETS}
    intervals = {}
    for target in TARGETS:
        residual = (validation_predictions["actual_" + target] - validation_predictions["predicted_" + target]).abs()
        radius = float(residual.quantile(.9))
        intervals[target] = {"validation_abs_error_p90": radius, "reference_interval": [projection[target] - radius, projection[target] + radius]}
    inputs = []
    for i, row in selected_profiles.iterrows():
        inputs.append({"player": row.player_name, "season": row.season, "minutes": float(minutes[i]),
                       "historical_usage_pct": float(row.usage_pct), "normalized_usage_pct": float(effective[i]),
                       "team_possession_share_pct": float(share[i] * 100),
                       **{f: None if pd.isna(row[f]) else float(row[f]) for f in PROFILE_FIELDS}})
    return dict(semantics="Five-player ironman lineup hypothetical; NOT an ordinary full-game roster projection",
                target_era="2016-17 preseason, 2015-16 league anchors; no game logs, fatigue or bench model",
                inputs=inputs, total_minutes=float(minutes.sum()), features=features,
                projections=projection, uncertainty_reference=intervals,
                warning="Validation intervals describe ordinary historical teams, NOT coverage guarantees for this out-of-distribution lineup")


def json_clean(obj):
    if isinstance(obj, dict):
        return {str(k): json_clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_clean(v) for v in obj]
    if isinstance(obj, (np.integer, np.floating)):
        obj = obj.item()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def table(rows):
    df = pd.DataFrame(rows)
    def cell(value):
        if pd.isna(value):
            return "null"
        if isinstance(value, (float, np.floating)):
            return f"{value:.3f}"
        return str(value).replace("|", "\\|").replace("\n", " ")
    lines = ["| " + " | ".join(map(str, df.columns)) + " |",
             "| " + " | ".join(["---"] * len(df.columns)) + " |"]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False, name=None)]
    return "\n".join(lines)


def equation_parameters(candidate):
    model, columns = candidate
    if model is None or not hasattr(model[-1], "coef_"):
        return None
    imputer, scaler, regressor = model.steps[0][1], model.steps[1][1], model.steps[2][1]
    coefficients = regressor.coef_ / scaler.scale_
    intercept = float(regressor.intercept_ - np.dot(coefficients, scaler.mean_))
    return dict(intercept=intercept, coefficients=dict(zip(columns, coefficients)),
                missing_value_medians=dict(zip(columns, imputer.statistics_)),
                equation="anchor + intercept + sum(coefficient * imputed feature)")


def write_report(summary):
    lines = ["\n## Counterfactual / Deployability Validation", "",
             "Reproduce offline: `python scripts/simulation/validate_deployability.py`. "
             "The earlier same-season experiments remain above as reconstruction references, not deployment evidence.",
             "", "Outputs: [frozen metrics, exact equation coefficients and feature audit](../../data/simulation/deployability_summary.json), "
             "[lagged inputs](../../data/simulation/lagged_roster_features.csv), "
             "[coverage](../../data/simulation/lagged_roster_coverage.csv), "
             "[predictions](../../data/simulation/lagged_roster_predictions.csv). "
             "The 23 deployable covariates are audited below; three additional explicit prediction offsets "
             "(`anchor_ortg`, `anchor_drtg`, `anchor_pace`) derive only from the prior league season: "
             "league_points_per_75/75*100 for ratings and league_pace for pace. They require no target-team outcome.",
             "", "### Population and temporal boundaries", "",
             json.dumps(summary["population"], indent=2), "",
             "Train targets: 1980-81 through 2013-14; validation: 2014-15 through 2018-19; "
             "test: 2019-20 through 2025-26. First target lacks a cached prior league anchor and is excluded. "
             "All players use profiles strictly before the target year. Historical future trades/injuries and actual "
             "season-long membership are known in hindsight; this is a roster-conditional experiment, not a preseason roster forecast.",
             "", "### Coverage and missing players", "",
             "Compare immediate previous season against latest available profile within three years. "
             "Select the latter structurally for returning players, without choosing based on final-test error. "
             "Returner MPG decays by 1/(1+0.15*(lag-1)). Unknown entrants use fixed training-only pooled "
             "profile medians, 8 MPG preference and an explicit unknown flag, never their target-year role. "
             "Known-profile missing fields use training-only prior-position medians. These are population priors, "
             "not fabricated individual observations. Primary eligibility requires >=80% known projected minutes "
             "in deployable mode; actual-minute coverage is diagnostic only. Coverage is optimistic for unknown stars "
             "because the rotation gives unknown entrants low preference. No international/rookie skill evidence is available.",
             "", table(summary["coverage_comparison"]), "", "### Shared rotation and usage builder", "",
             "Oracle rotation is NOT DEPLOYABLE: target actual minutes normalized to 240. All performance remains lagged. "
             "Deployable rotation ranks prior MPG only: first five floor24/cap40, next five floor6/cap26, others zero. "
             "Allocate the remaining minutes proportionally to prior MPG (minimum preference1), iterating at caps. "
             "For 6-player capacity shortage caps become48; five players necessarily play48 each. "
             "No target-season starts, usage, MPG, efficiency or team pace enter features. No anonymous players. "
             "These are ranking roles, not verified starter labels, and no positional lineup constraints are modeled.",
             "", "With m_i projected minutes, e_i=m_i/48, u_i=prior USG%, q_i=clip(u_i,1,60)^alpha, "
             "normalized usage v_i=100*q_i/sum(e_j*q_j); possession share s_i=e_i*v_i/100. "
             "Thus sum(s_i)=1 and sum(e_i*v_i)=100. Compression=sum((m_i/240)*max(u_i-v_i,0)); "
             "concentration=sum(s_i^2). Alpha is selected among 0.5/0.75/1 using training teams only to minimize "
             "projected-minute-weighted error against their realized, normalized usage (calibration targets only). "
             "This calibrates role matching, not a causal efficiency response. "
             "Fit coefficients on compression and other effects on training teams only. Normalization conserves "
             "usage and retains larger roles for high-usage players at equal minutes; an efficiency penalty is not "
             "assumed unless its training coefficient supports one. No chemistry score. Training calibration: `" +
             json.dumps(summary["usage_calibration"]) + "`",
             "", "### Production feature audit", "",
             "All inputs below derive from prior/selected PlayerSeason profiles, roster membership and proposed minutes. "
             "No PER, target scoring-volume aggregate, target-team outcome or PER-based composite is used. "
             "Prior per75 rates inherit prior team-pace estimation/context; they are not context-free player talent. "
             "League anchors use the preceding season, which avoids requiring future league outcomes. "
             "Reconstruction and deployable feature sets are separately exported in JSON.", ""]
    lines.append(table(summary["feature_audit"]))
    lines += ["", "### Model comparisons", "",
              "Predict residuals relative to prior league ORtg/pace. The transparent formula is a small additive "
              "OLS equation using the explicitly listed fit terms, not arbitrary hand coefficients. Full linear and "
              "fixed Ridge(alpha30) compare against the league anchor. One small HGB offense candidate checks "
              "nonlinear usage/spacing interactions; defense and pace remain additive. Imputation/scaling fit on "
              "training only. Select minimum validation MAE separately per target/mode. No test tuning or refitting. "
              "Oracle and deployable are distinct experimental fits; only the latter serves the custom lineup.", ""]
    rows = []
    for mode, targets in summary["metrics"].items():
        for target, models in targets.items():
            for model, splits in models.items():
                for split, score in splits.items():
                    rows.append(dict(mode=mode, target=target, model=model, split=split, **score))
    lines.append(table(rows))
    lines += ["", "Selected on validation: `" + json.dumps(summary["selected_models"]) + "`", "",
              "The fitted deployable offensive compression coefficient is `" +
              str(summary["equation_parameters"]["deployable"]["ortg"]["transparent_formula"]["coefficients"]["usage_compression"]) +
              "`. It is positive: normalization conserves usage and reduces individual star roles, but these "
              "historical data do NOT establish an efficiency penalty for star crowding. Do not interpret this "
              "association as a causal fit bonus or claim validated efficiency diminishing returns. This is a "
              "specific acceptance blocker requiring targeted role/transfer evidence.", "",
              "### Defensive source provenance", "",
              "Offline ingestion extension `scripts/historical/export_cached_defensive_profiles.py` reuses "
              "the canonical cache and numeric parser with row-level source IDs (names are not unique), exporting cached Basketball-Reference "
              "`NBA_YYYY_advanced.html` DBPM/DWS/STL%/BLK%/DRB% fields to a supplemental analysis file. "
              "Source URLs: https://www.basketball-reference.com/leagues/NBA_YYYY_advanced.html. "
              "No downloads, canonical schema or production application changes. DBPM and DWS are source-derived "
              "estimates, not observed individual defensive impact; percentage fields are source-computed box rates. "
              "DWS is audited but excluded from predictors because cumulative wins/playing time are context-heavy. "
              "Absent source entries remain null in the sidecar; model imputation is explicit. Older eras are supported "
              "only where the cached source records the metric. Availability by era is in JSON.", "",
              "Canonical profiles without a matching row-level source ID/team or with inconsistent source minutes are quarantined from "
              "prior histories; canonical datasets are not rewritten. This detects an inherited Charles Jones "
              "same-name identity collision and other source-join inconsistencies. It does not repair all canonical "
              "identity issues, and target membership can still inherit them. Quarantined rows: `" +
              json.dumps(summary["quarantined_canonical_profiles"]) + "`", "",
              "### Era, quality and error distribution", ""]
    for mode, groups in summary["breakdowns"].items():
        lines.append("#### " + mode)
        era_rows = []
        for era, scores in groups["era"].items():
            for target in TARGETS:
                era_rows.append(dict(era=era, split=scores["split"], n=scores["n"], target=target, **scores[target]))
        lines += ["", table(era_rows), "", "Test quality tiers use actual net rating only for diagnosis: poor<-3, average[-3,3], elite>3.", ""]
        lines.append(table([dict(tier=tier, n=score["n"], target=t, **score[t])
                            for tier, score in groups["test_quality_tier"].items() for t in TARGETS]))
        lines += ["", table([dict(target=t, **scores) for t, scores in groups["test_absolute_error_quantiles"].items()]), ""]
        lines += ["Defensive stability against the anchor (no post-test reselection):", "",
                  table([dict(model=name, era=era, **score) for name, eras in groups["defense_test_era_candidates"].items()
                         for era, score in eras.items()]), ""]
    lines += ["### Major roster changes", "", table(summary["major_roster_changes"]), "",
              "These misses combine roster/injury availability, role changes and coaching effects absent from player "
              "box profiles. Boston's new defensive system, Miami's role adjustment, Durant's arrival and the Lakers' "
              "defensive personnel are stress cases, not cases used to tune coefficients. Train-period examples are "
              "in-sample sanity checks, not held-out evidence. Specific residuals and coverage appear in the table.",
              "", "### Five-player semantics and all-time lineup", "",
              "Recommend minimum eight named players for a normal V1 full-game roster. Permit five-player "
              "lineups only as explicitly labeled ironman hypotheticals with high, uncalibrated uncertainty; "
              "48 minutes each is unavoidable. No fatigue/bench/replacement simulation is implied.", "",
              table(summary["custom_lineup"]["inputs"]), "",
              "Target-era assumptions: " + summary["custom_lineup"]["target_era"], "",
              "Shared aggregate inputs: `" + json.dumps(summary["custom_lineup"]["features"]) + "`", "",
              "Projections (diagnostic, not accepted production ratings): `" + json.dumps(summary["custom_lineup"]["projections"]) + "`", "",
              "Validation ordinary-team uncertainty reference: `" + json.dumps(summary["custom_lineup"]["uncertainty_reference"]) + "`", "",
              "Out-of-training-range features: `" + json.dumps(summary["custom_lineup"]["out_of_training_range"]) + "`", "",
              "### Acceptance and next step", "",
              "**B. NEEDS TARGETED IMPROVEMENT**. Lagged validation removes same-season performance dependence "
              "and tests the whole eligible league, but it cannot establish causal transport to cross-era star "
              "lineups. Defense lacks matchup/lineup coverage; usage shrinkage/efficiency response and role/minute "
              "allocation remain hypotheses. Ordinary-team errors are not synthetic-roster calibration. "
              "Era-level defensive stability must be assessed against the league-anchor baseline (also exported "
              "per era); a learned model is only an experimental candidate until it improves consistently. "
              "No readiness claim follows from a high R-squared.", "",
              "Next: validate rotation against held-out actual minutes diagnostically, conduct held-out "
              "transfer/roster-turnover and normalized-usage sensitivity tests, and establish explicit defensive "
              "acceptance tolerances using validation only. Keep final-test results frozen. Do not acquire "
              "game logs or start Monte Carlo calibration yet.", ""]
    path = REPORT_DIR / "SIMULATION_ENGINE_DESIGN.md"
    original = path.read_text(encoding="utf-8").split("\n## Counterfactual / Deployability Validation")[0]
    path.write_text(original + "\n".join(lines), encoding="utf-8")


def main():
    global USAGE_EXPONENT
    print("Reading cached defensive fields and checking source identities...", flush=True)
    players, roster, teams, leagues, profiles, histories, defaults, roles, identity_problems = prepare_inputs()
    print("Calibrating usage preference on training teams only...", flush=True)
    USAGE_EXPONENT, usage_scores = calibrate_usage(roster, histories, defaults, roles)
    comparisons, frames, audits = [], {}, {}
    for lag in [1, 3]:
        print(f"Building lagged rosters: maximum profile lag {lag}...", flush=True)
        frame, audit = build_lagged(roster, teams, leagues, histories, defaults, roles, lag)
        frames[lag], audits[lag] = frame, audit
        deploy = frame[frame["mode"] == "deployable"]
        comparisons.append(dict(max_profile_lag=lag, candidate_teams=len(audit), eligible_teams=len(deploy),
                                mean_projected_coverage=float(audit.known_projected_minutes_share.mean()),
                                mean_actual_coverage=float(audit.known_actual_minutes_share.mean()),
                                projected_coverage_p10=float(audit.known_projected_minutes_share.quantile(.1)),
                                actual_coverage_p10=float(audit.known_actual_minutes_share.quantile(.1)),
                                teams_actual_coverage_below_80=int((audit.known_actual_minutes_share < .8).sum())))
    # Fallback policy chosen for coverage, not predictive test error.
    frame = frames[3]
    result, models, selected, predictions, breakdowns = {}, {}, {}, {}, {}
    for mode in ["oracle", "deployable"]:
        print(f"Fitting frozen temporal experiment: {mode}...", flush=True)
        subset = frame[frame["mode"] == mode].copy()
        result[mode], models[mode], selected[mode], predictions[mode] = fit_experiment(subset)
        breakdowns[mode] = breakdown(predictions[mode])
        # Preserve an era-wise comparator for defensive stability, including every baseline/candidate.
        breakdowns[mode]["defense_test_era_candidates"] = {}
        for name, candidate in models[mode]["drtg"].items():
            rows = subset[subset.season_end_year >= 2015].copy()
            rows["prediction"] = predict(rows, "drtg", candidate)
            breakdowns[mode]["defense_test_era_candidates"][name] = {
                label: metrics(g.actual_drtg, g.prediction) for label, g in
                rows.groupby(np.select([rows.season_end_year <= 2019, rows.season_end_year <= 2022],
                                       ["validation_2015_2019", "test_2020_2022"], default="test_2023_2026"))}
    valpred = predictions["deployable"].query("2015 <= season_end_year <= 2019")
    custom = custom_profile(players, profiles, leagues, models["deployable"], selected["deployable"], valpred)
    training = frame.query("mode == 'deployable' and season_end_year <= 2014")
    custom["out_of_training_range"] = {field: dict(value=custom["features"][field], min=float(training[field].min()), max=float(training[field].max()))
                                       for field in DEPLOYABLE_FEATURE_SET
                                       if custom["features"][field] < training[field].min() or custom["features"][field] > training[field].max()}
    checks = []
    for mode, preds in predictions.items():
        for season, team in [("2007-08", "Boston Celtics"), ("2010-11", "Miami Heat"),
                             ("2016-17", "Golden State Warriors"), ("2019-20", "Los Angeles Lakers")]:
            found = preds[(preds.season == season) & (preds.team_name == team)]
            if len(found):
                checks += found.to_dict("records")
            else:
                checks.append(dict(mode=mode, season=season, team_name=team, status="excluded by coverage"))
    audit_features = []
    for field in DEPLOYABLE_FEATURE_SET:
        source = ("sum(minutes/240 * prior." + field[2:] + ") over non-null source values"
                  if field.startswith("w_") else {
                      "usage_weighted_relative_ts": "sum(normalized possession share * prior.relative_ts)",
                      "usage_weighted_3par": "sum(normalized possession share * prior.three_point_attempt_rate)",
                      "usage_weighted_ftr": "sum(normalized possession share * prior.free_throw_rate)",
                      "usage_compression": "sum(minutes/240 * max(prior.USG - normalized USG,0))",
                      "usage_concentration": "sum(normalized possession share squared)",
                      "rotation_hhi": "sum((proposed minutes/240) squared)",
                      "prior_pace_delta": "sum(minutes/240 * (prior.team_pace - prior.league_pace))",
                      "known_prior_minutes_share": "sum(proposed minutes/240 for sourced prior profiles)",
                      "returner_minutes_share": "sum(proposed minutes/240 for prior lag >1)",
                      "prior_age_years": "sum(minutes/240 * sourced prior profile lag in years); unknown=0",
                  }[field])
        audit_features.append(dict(feature=field, source=source,
                                   target_performance_required=False, synthetic_available=True))
    availability = {str(era): {field: float(group[field].notna().mean()) for field in DEFENSIVE_FIELDS.values()}
                    for era, group in profiles.groupby((profiles.season_end_year // 10) * 10)}
    summary = dict(population={"eligible_team_seasons": int(len(frame) // 2),
                              "rows_including_both_modes": len(frame),
                              "splits_per_mode": {k: len(v) // 2 for k, v in temporal_split(frame).items()},
                              "seasons": sorted(frame.season.unique()),
                              "coverage_threshold": MIN_COVERAGE},
                   coverage_comparison=comparisons, deployable_feature_set=DEPLOYABLE_FEATURE_SET,
                   usage_calibration=dict(selected_exponent=USAGE_EXPONENT, training_scores=usage_scores),
                   reconstruction_feature_set=RECONSTRUCTION_FEATURE_SET, feature_audit=audit_features,
                   formula_terms=FORMULA_FIELDS, metrics=result, selected_models=selected,
                   equation_parameters={mode: {target: {name: equation_parameters(candidate)
                                        for name, candidate in candidates.items()}
                                        for target, candidates in targets.items()} for mode, targets in models.items()},
                   quarantined_canonical_profiles=identity_problems,
                   breakdowns=breakdowns, major_roster_changes=checks, custom_lineup=custom,
                   defensive_source_availability=availability,
                   verdict="B. NEEDS TARGETED IMPROVEMENT")
    summary = json_clean(summary)
    OUT_DIR.mkdir(exist_ok=True)
    frame.to_csv(OUT_DIR / "lagged_roster_features.csv", index=False)
    audits[3].to_csv(OUT_DIR / "lagged_roster_coverage.csv", index=False)
    pd.concat(predictions.values()).to_csv(OUT_DIR / "lagged_roster_predictions.csv", index=False)
    (OUT_DIR / "deployability_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_report(summary)
    print(json.dumps({"population": summary["population"], "selected_models": selected,
                      "custom_projections": custom["projections"], "verdict": summary["verdict"]}, indent=2))
    return 0


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        raise SystemExit(main())
