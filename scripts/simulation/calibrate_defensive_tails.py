"""Training-OOF calibration with a single frozen validation evaluation; no final test.

Run --prepare, tests, then --validate once. --report only renders stored results.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import defensive_continuity as defense

prior = defense.prior
ROOT, OUT_DIR, REPORT_DIR = defense.ROOT, defense.OUT_DIR, defense.REPORT_DIR
PREFIX = OUT_DIR / "defensive_oof_calibration_v1"
TRAINING = Path(str(PREFIX) + "_training.json")
PROTOCOL = Path(str(PREFIX) + "_protocol.json")
RESULT = Path(str(PREFIX) + "_validation.json")
LEDGER = Path(str(PREFIX) + "_validation_run.json")
ARTIFACT = ROOT / "artifacts/simulation/defensive_oof_calibration_v1.joblib"
BASE_ARTIFACT = ROOT / "artifacts/simulation/defensive_continuity_v1.joblib"
FEATURES = OUT_DIR / "defensive_continuity_v1_features.csv"
BASE_SUMMARY = OUT_DIR / "defensive_continuity_v1_summary.json"
OOF_PATH = Path(str(PREFIX) + "_training_oof.csv")
ENDS = [1985, 1990, 1995, 2000, 2005, 2010]
KINDS = ["A_identity", "B_symmetric", "C_asymmetric"]


def protected_hashes():
    paths = defense.PROTECTED + [BASE_ARTIFACT, BASE_SUMMARY, FEATURES,
                                 OUT_DIR / "defensive_continuity_v1_predictions.csv",
                                 ROOT / "scripts/simulation/defensive_continuity.py",
                                 ROOT / "scripts/simulation/improve_projection.py"]
    return {str(path.relative_to(ROOT)): prior.fingerprint(path) for path in paths}


def train_only(frame):
    defense.require_development(frame)
    if frame.season_end_year.max() > 2014:
        raise ValueError("Calibration coefficients may use TRAIN through 2013-14 only")


def assert_oof(frame):
    train_only(frame)
    if (frame.base_train_end >= frame.season_end_year).any():
        raise ValueError("OOF row was predicted by a model that saw that season")
    if frame.duplicated(["season_end_year", "team_id"]).any():
        raise ValueError("Duplicate OOF targets")


def residual_metrics(frame, prediction):
    actual = frame.actual_drtg.to_numpy() - frame.anchor_drtg.to_numpy()
    estimated = np.asarray(prediction) - frame.anchor_drtg.to_numpy()
    error = np.asarray(prediction) - frame.actual_drtg.to_numpy()
    labels = defense.bucket_labels(frame, "legacy")
    elite, poor = np.asarray(labels == "elite"), np.asarray(labels == "poor")
    slope, intercept = np.linalg.lstsq(np.column_stack([estimated, np.ones(len(estimated))]), actual, rcond=None)[0]
    return dict(n=len(frame), **defense.metrics(frame.actual_drtg, prediction),
                actual_residual_sd=float(np.std(actual, ddof=1)), predicted_residual_sd=float(np.std(estimated, ddof=1)),
                actual_on_predicted_slope=float(slope), actual_on_predicted_intercept=float(intercept),
                residual_spearman=float(pd.Series(actual).corr(pd.Series(estimated), method="spearman")),
                elite_bias=float(np.mean(error[elite])) if elite.any() else None,
                poor_bias=float(np.mean(error[poor])) if poor.any() else None,
                p90_abs_error=float(np.quantile(abs(error), .9)))


def through_zero(x, y):
    denominator = float(np.dot(x, x))
    if len(x) < 10 or denominator < 1e-10:
        raise ValueError("Insufficient residual support for calibration")
    return float(np.dot(x, y) / denominator)


def fit_calibration(oof, kind):
    assert_oof(oof)
    x = oof.base_prediction.to_numpy() - oof.anchor_drtg.to_numpy()
    y = oof.actual_drtg.to_numpy() - oof.anchor_drtg.to_numpy()
    result = dict(kind=kind, fitting_rows=len(oof), fitting_max_year=int(oof.season_end_year.max()))
    if kind == "A_identity":
        result["gamma"] = 1.
    elif kind == "B_symmetric":
        result["gamma"] = through_zero(x, y)
    elif kind == "C_asymmetric":
        result["gamma_better"] = through_zero(x[x < 0], y[x < 0])
        result["gamma_worse"] = through_zero(x[x >= 0], y[x >= 0])
        result["n_better"], result["n_worse"] = int(sum(x < 0)), int(sum(x >= 0))
    elif kind == "D_piecewise":
        knot = float(np.quantile(abs(x), .75))
        hinge = np.sign(x) * np.maximum(abs(x) - knot, 0.)
        coefficients = np.linalg.lstsq(np.column_stack([x, hinge]), y, rcond=None)[0]
        result.update(gamma_center=float(coefficients[0]), gamma_tail=float(sum(coefficients)),
                      hinge_coefficient=float(coefficients[1]), knot=knot)
    else:
        raise ValueError("Unknown calibration")
    return result


def calibrated_residual(residual, parameters):
    x = np.asarray(residual, float)
    kind = parameters["kind"]
    if kind in {"A_identity", "B_symmetric"}:
        return parameters["gamma"] * x
    if kind == "C_asymmetric":
        return np.where(x < 0, parameters["gamma_better"], parameters["gamma_worse"]) * x
    return parameters["gamma_center"] * x + parameters["hinge_coefficient"] * np.sign(x) * np.maximum(abs(x) - parameters["knot"], 0.)


def apply_calibration(base, anchor, parameters, unsupported=None):
    """Guard unsupported synthetic values rather than silently clipping every team."""
    base, anchor = np.asarray(base, float), np.asarray(anchor, float)
    predicted = anchor + calibrated_residual(base - anchor, parameters)
    return np.where(np.asarray(unsupported, bool), base, predicted) if unsupported is not None else predicted


def parameter_values(parameters):
    return {name: value for name, value in parameters.items() if name in
            {"gamma", "gamma_better", "gamma_worse", "gamma_center", "gamma_tail"}}


def coefficient_stability(fold_parameters):
    rows = []
    for name in parameter_values(fold_parameters[0]):
        values = np.array([parameter_values(row)[name] for row in fold_parameters])
        mean = float(values.mean())
        cv = float(values.std(ddof=1) / abs(mean)) if abs(mean) > 1e-10 else float("inf")
        rows.append(dict(parameter=name, mean=mean, sd=float(values.std(ddof=1)), min=float(values.min()),
                         max=float(values.max()), coefficient_of_variation=cv))
    # Stability thresholds decide eligibility; coefficients are never clipped to pass them.
    stable = all(row["coefficient_of_variation"] <= .35 and row["min"] >= .25 and row["max"] <= 2.5 for row in rows)
    return dict(stable=stable, parameters=rows,
                rule="Every multiplier positive in [0.25,2.5], fold coefficient of variation<=0.35; no clipping")


def build_oof(train, columns):
    train_only(train)
    predictions, diagnostics = [], []
    for index, end in enumerate(ENDS):
        last = ENDS[index + 1] if index + 1 < len(ENDS) else 2014
        earlier, held = train[train.season_end_year <= end], train[(train.season_end_year > end) & (train.season_end_year <= last)]
        if earlier.empty or held.empty or earlier.season_end_year.max() >= held.season_end_year.min():
            raise ValueError("Invalid expanding-window fold")
        print(f"OOF fold {index+1}: train through {end}, evaluate {end+1}-{last}", flush=True)
        variants = {"frozen_formulation_OLS": defense.fit_linear(earlier, columns),
                    "same_features_Ridge30": defense.fit_linear(earlier, columns, ridge=True),
                    "without_continuity": defense.fit_linear(earlier, [f for f in columns if f not in
                                                                     {"retained_defensive_context", "returning_minutes_share"}])}
        for name, model in variants.items():
            estimate = defense.predict(held, model)
            row = dict(fold=index + 1, variant=name, train_start=1981, train_end=end,
                       evaluate_start=end+1, evaluate_end=last, train_n=len(earlier), **residual_metrics(held, estimate))
            row["missingness_strata"] = {}
            for label, group in held.assign(prediction=estimate).groupby(np.where(held.defensive_missing_share > .05, ">5% missing", "<=5% missing")):
                row["missingness_strata"][label] = residual_metrics(group, group.prediction)
            diagnostics.append(row)
            if name == "frozen_formulation_OLS":
                output = held.copy()
                output["base_prediction"], output["fold"], output["base_train_end"] = estimate, index + 1, end
                predictions.append(output)
    oof = pd.concat(predictions, ignore_index=True)
    assert_oof(oof)
    return oof, diagnostics


def piecewise_evidence(oof):
    rows = []
    for fold, group in oof.groupby("fold"):
        x = group.base_prediction.to_numpy() - group.anchor_drtg.to_numpy()
        y = group.actual_drtg.to_numpy() - group.anchor_drtg.to_numpy()
        tail = abs(x) >= np.quantile(abs(x), .75)
        center_gamma, tail_gamma = through_zero(x[~tail], y[~tail]), through_zero(x[tail], y[tail])
        rows.append(dict(fold=int(fold), center_gamma=center_gamma, tail_gamma=tail_gamma,
                         difference=tail_gamma - center_gamma, tail_n=int(sum(tail))))
    differences = np.array([row["difference"] for row in rows])
    direction = np.sign(np.median(differences))
    supported = abs(np.median(differences)) >= .20 and sum(np.sign(differences) == direction) >= 4 and all(row["tail_n"] >= 20 for row in rows)
    return dict(supported=bool(supported), folds=rows,
                rule="Evaluate one hinge only if median tail-center gamma gap>=0.20, same direction in>=4/6 folds, >=20 tail rows per fold")


def calibration_diagnostics(oof, kinds):
    results = {}
    for kind in kinds:
        fold_params = [fit_calibration(group, kind) for _, group in oof.groupby("fold")]
        prequential = []
        for fold, group in oof.groupby("fold"):
            previous = oof[oof.fold < fold]
            if previous.empty:
                continue
            parameters = fit_calibration(previous, kind)
            prediction = apply_calibration(group.base_prediction, group.anchor_drtg, parameters)
            prequential.append(dict(fold=int(fold), parameters=parameters, **residual_metrics(group, prediction)))
        stability = coefficient_stability(fold_params)
        baseline = [residual_metrics(g, g.base_prediction) for f, g in oof.groupby("fold") if f > 1]
        no_noise_amplification = sum(row["mae"] <= ref["mae"] + .15 and abs(row["elite_bias"]) <= abs(ref["elite_bias"]) + .5
                                     for row, ref in zip(prequential, baseline)) >= 4
        poor_not_worse = sum(abs(row["poor_bias"]) <= abs(ref["poor_bias"]) + .25 for row, ref in zip(prequential, baseline))
        poor_improvements = sum(abs(row["poor_bias"]) < abs(ref["poor_bias"]) - .05 for row, ref in zip(prequential, baseline))
        poor_stable = kind == "A_identity" or (poor_not_worse >= 4 and poor_improvements >= 2)
        mean_not_worse = np.mean([row["mae"] for row in prequential]) <= np.mean([row["mae"] for row in baseline]) + .05
        stability.update(prequential_not_fragile=bool(no_noise_amplification and mean_not_worse),
                         poor_not_worse_blocks=poor_not_worse, poor_improved_blocks=poor_improvements,
                         poor_correction_stable=bool(poor_stable))
        stability["eligible"] = bool(stability["stable"] and no_noise_amplification and mean_not_worse and poor_stable)
        results[kind] = dict(final_parameters=fit_calibration(oof, kind), fold_parameters=fold_params,
                             stability=stability, prequential=prequential)
    return results


def prepare_stage():
    if LEDGER.exists() or RESULT.exists():
        raise RuntimeError("Validation already consumed; no further calibration iteration")
    protected = protected_hashes()
    train = pd.read_csv(FEATURES)
    train = train[train.season_end_year <= 2014].copy()
    train_only(train)
    state = joblib.load(BASE_ARTIFACT)
    columns = state["model"]["candidate"][1]
    if state["model"]["tail"] is not None:
        raise ValueError("Expected frozen additive defensive formulation")
    oof, folds = build_oof(train, columns)
    evidence = piecewise_evidence(oof)
    kinds = KINDS + (["D_piecewise"] if evidence["supported"] else [])
    candidates = calibration_diagnostics(oof, kinds)
    x = oof.base_prediction.to_numpy() - oof.anchor_drtg.to_numpy()
    y = oof.actual_drtg.to_numpy() - oof.anchor_drtg.to_numpy()
    intercept, slope = np.linalg.lstsq(np.column_stack([np.ones(len(x)), x]), y, rcond=None)[0]
    intercept_folds = []
    for fold, group in oof.groupby("fold"):
        xx = group.base_prediction.to_numpy() - group.anchor_drtg.to_numpy()
        yy = group.actual_drtg.to_numpy() - group.anchor_drtg.to_numpy()
        offset, beta = np.linalg.lstsq(np.column_stack([np.ones(len(xx)), xx]), yy, rcond=None)[0]
        intercept_folds.append(dict(fold=int(fold), intercept=float(offset), gamma=float(beta)))
    result = dict(train_rows=len(train), warmup_rows=int(sum(train.season_end_year <= 1985)), oof_rows=len(oof),
                  folds=folds, oof_summary=residual_metrics(oof, oof.base_prediction),
                  candidates=candidates, piecewise_evidence=evidence,
                  intercept_diagnostic=dict(intercept=float(intercept), gamma=float(slope), folds=intercept_folds,
                                            production_candidate=False, reason="Diagnostic only; no extra intercept family selected on validation"),
                  protected_hashes=protected)
    if protected_hashes() != protected:
        raise RuntimeError("Original artifacts changed during preparation")
    prior.write_json(TRAINING, result)
    oof.to_csv(OOF_PATH, index=False)
    artifact = dict(candidates={name: record["final_parameters"] for name, record in candidates.items()},
                    eligibility={name: record["stability"]["eligible"] for name, record in candidates.items()},
                    base_model_sha256=prior.fingerprint(BASE_ARTIFACT), protected_hashes=protected,
                    residual_support=dict(min=float(x.min()), max=float(x.max()), p01=float(np.quantile(x, .01)), p99=float(np.quantile(x, .99))))
    joblib.dump(artifact, ARTIFACT)
    protocol = dict(status="training_OOF_calibration_frozen_before_validation", protected_hashes=protected,
                    training_sha256=prior.fingerprint(TRAINING), artifact_sha256=prior.fingerprint(ARTIFACT),
                    oof_sha256=prior.fingerprint(OOF_PATH), source_sha256=prior.fingerprint(Path(__file__)),
                    candidate_order=kinds,
                    selection="Passing original gates in both bucket views, OOF stability eligibility, then simpler family before small validation MAE gains",
                    product_B_rule="Hard gates still reported as FAIL: at most0.25 DRtg marginal original bucket-bias miss, all other original gates pass, ranking view passes, validation residual Spearman>=0.40, positive ranking in>=5/6 training folds, stable chosen formulation",
                    unsupported_synthetic_rule="Preserve base prediction, mark HIGH uncertainty; no automatic spread amplification or universal historical clamp",
                    final_test_evaluations=0)
    prior.write_json(PROTOCOL, protocol)
    print(json.dumps(prior.previous.json_clean(dict(oof_rows=len(oof), candidates=candidates,
                     optional_piecewise=evidence["supported"])), indent=2))


def assess_validation(frame, prediction, limits, threshold):
    record = {}
    for view in ["legacy", "ranking"]:
        d = defense.diagnostics(frame, prediction, view, threshold)
        record[view] = dict(diagnostics=d, **prior.accepted_defense(d, limits))
    record["accepted"] = all(record[view]["accepted"] for view in ["legacy", "ranking"])
    record["residual_metrics"] = residual_metrics(frame, prediction)
    return record


def marginal_product_rule(record, stability, positive_folds):
    checks = record["legacy"]["checks"]
    d = record["legacy"]["diagnostics"]
    worst = max(abs(row["bias"]) for row in d["buckets"].values())
    return bool(stability and all(value for key, value in checks.items() if key != "bucket_bias") and
                worst <= 3.25 and record["ranking"]["accepted"] and
                record["residual_metrics"]["residual_spearman"] >= .40 and positive_folds >= 5)


def ironman_diagnostic(base_summary, parameters):
    custom = base_summary["custom_lineup"]
    if custom["continuity"] != 0 or custom["retained_defensive_context"] != 0:
        raise ValueError("Synthetic roster cannot inherit actual team context")
    leagues = pd.read_csv(ROOT / "data/historical/processed/league_seasons.csv")
    anchor = float(leagues.loc[leagues.season_end_year == 2016, "league_points_per_75"].iloc[0] / 75 * 100)
    base_features = {**custom["individual_evidence"], "retained_defensive_context": 0.,
                     "returning_minutes_share": 0., "anchor_drtg": anchor}
    frozen_base = joblib.load(BASE_ARTIFACT)
    base = float(defense.predict(pd.DataFrame([base_features]), frozen_base["model"])[0])
    if not np.isclose(base, custom["projected_drtg"], atol=1e-10):
        raise ValueError("Frozen ironman base prediction drifted")
    raw = float(apply_calibration([base], [anchor], parameters)[0])
    outside = custom["uncertainty"]["coverage"]["fraction_outside"] > 0
    guarded = float(apply_calibration([base], [anchor], parameters, unsupported=[outside])[0])
    residual = base - anchor
    nominal = (raw - anchor) / residual if abs(residual) > 1e-10 else 1.
    return dict(base_drtg=base, base_prediction_reproduced=True, raw_calibration_diagnostic=raw,
                calibrated_drtg=guarded, league_anchor=anchor,
                nominal_multiplier=nominal, effective_multiplier=(guarded - anchor) / residual,
                continuity=0., retained_defensive_context=0., unsupported=bool(outside),
                guard_applied=bool(outside), uncertainty="HIGH" if outside else custom["uncertainty"]["category"],
                base_uncertainty=custom["uncertainty"],
                uncertainty_warning="Unsupported ironman feature vector: base value preserved; spread calibration is not validated here",
                offense_pace_unchanged=custom["offense_pace_unchanged"],
                semantics="Curry2015-16/Jordan1990-91/LeBron2012-13/Duncan2002-03/Shaq1999-00; 48 minutes each; diagnostic, not precise truth")


def validate_stage():
    if LEDGER.exists() or RESULT.exists():
        raise RuntimeError("Validation already consumed; use --report without reevaluating")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if (protected_hashes() != protocol["protected_hashes"] or prior.fingerprint(TRAINING) != protocol["training_sha256"] or
        prior.fingerprint(ARTIFACT) != protocol["artifact_sha256"] or prior.fingerprint(OOF_PATH) != protocol["oof_sha256"] or
        prior.fingerprint(Path(__file__)) != protocol["source_sha256"]):
        raise RuntimeError("Frozen training calibration changed")
    artifact = joblib.load(ARTIFACT)
    training = json.loads(TRAINING.read_text(encoding="utf-8"))
    old = json.loads(BASE_SUMMARY.read_text(encoding="utf-8"))
    frame = pd.read_csv(FEATURES)
    frame = frame[frame.season_end_year.between(2015, 2019)].copy()
    if len(frame) != 150:
        raise ValueError("Validation population changed")
    state = joblib.load(BASE_ARTIFACT)
    # Consume the single validation evaluation before any defensive predictions or metrics.
    with LEDGER.open("x", encoding="utf-8") as stream:
        json.dump(dict(status="started", evaluation_count=1, protocol_sha256=prior.fingerprint(PROTOCOL)), stream, indent=2)
    base = defense.predict(frame, state["model"])
    results, predictions = {}, frame[["season", "season_end_year", "team_name", "actual_drtg", "anchor_drtg"]].copy()
    box_threshold = json.loads(prior.VALIDATION.read_text(encoding="utf-8"))["defense_acceptance"]["box_threshold"]
    for name in protocol["candidate_order"]:
        estimate = apply_calibration(base, frame.anchor_drtg, artifact["candidates"][name])
        predictions[name] = estimate
        results[name] = assess_validation(frame, estimate, old["tolerances"], box_threshold)
    eligible = [name for name in protocol["candidate_order"] if artifact["eligibility"][name]]
    passing = [name for name in eligible if results[name]["accepted"]]
    positive_folds = sum(row["residual_spearman"] > 0 for row in training["folds"] if row["variant"] == "frozen_formulation_OLS")
    marginal = [name for name in eligible if marginal_product_rule(results[name], True, positive_folds)]
    if passing:
        selected, verdict = passing[0], "A. READY FOR GAME-LOG CALIBRATION"
    elif marginal:
        selected, verdict = marginal[0], "B. DEFENSE REMAINS IMPERFECT BUT ACCEPTABLE FOR UNCERTAINTY-AWARE V1"
    else:
        selected = min(eligible, key=lambda name: results[name]["legacy"]["diagnostics"]["overall"]["mae"]) if eligible else "A_identity"
        verdict = "C. DEFENSIVE MODEL NOT RELIABLE ENOUGH"
    if protected_hashes() != protocol["protected_hashes"]:
        raise RuntimeError("Original artifacts changed")
    integrity = dict(protected_hashes_match=True, offense_pace_usage_rotation_unchanged=True,
                     official_test_untouched=True, new_final_test_evaluations=0,
                     frozen_offense_pace_values=old["custom_lineup"]["offense_pace_unchanged"])
    result = dict(evaluation_count=1, validation_n=len(frame), candidates=results, selected=selected,
                  parameters=artifact["candidates"][selected], selected_stability=training["candidates"][selected]["stability"],
                  tolerances=old["tolerances"], verdict=verdict, product_B_is_not_hard_gate_pass=bool(not passing and marginal),
                  ironman=ironman_diagnostic(old, artifact["candidates"][selected]), integrity=integrity,
                  protocol_sha256=prior.fingerprint(PROTOCOL), retuning_after_validation=False)
    prior.write_json(RESULT, result)
    predictions.to_csv(Path(str(PREFIX) + "_validation_predictions.csv"), index=False)
    prior.write_json(LEDGER, dict(status="complete", evaluation_count=1, result_sha256=prior.fingerprint(RESULT),
                                 protocol_sha256=prior.fingerprint(PROTOCOL), new_final_test_evaluations=0))
    joblib.dump(dict(calibration=artifact["candidates"][selected], base_model_sha256=artifact["base_model_sha256"],
                     verdict=verdict, hard_gates_pass=bool(passing),
                     uncertainty_guard=protocol["unsupported_synthetic_rule"], validation_result_sha256=prior.fingerprint(RESULT)),
                ROOT / "artifacts/simulation/defensive_oof_calibration_v1_selected.joblib")
    render_report()
    print(json.dumps(prior.previous.json_clean(dict(selected=selected, parameters=result["parameters"],
                     validation=results[selected], ironman=result["ironman"], verdict=verdict)), indent=2))


def render_report():
    train = json.loads(TRAINING.read_text(encoding="utf-8"))
    result = json.loads(RESULT.read_text(encoding="utf-8")) if RESULT.exists() else None
    table = prior.previous.table
    lines = ["\n## Rolling-Origin Defensive Diagnostics", "",
             "Commands: `python scripts/simulation/calibrate_defensive_tails.py --prepare`, tests, then "
             "`--validate` exactly once. `--report` only renders stored results. No frozen 2019-20 through "
             "2025-26 evaluation is exposed. Official test artifacts and all original model artifacts are protected by hashes.", "",
             "Frozen base formulation: continuity-aware additive OLS with DBPM, steals/blocks/defensive rebounds "
             "per75, age, position mix, rotation concentration and retained context. No new feature families. "
             "Train starts1980-81. Warmup through1984-85 is excluded from calibration; expanding five-season "
             "evaluation blocks cover1985-86 through2013-14 (last block four seasons). Every base prediction "
             "uses coefficients and imputation trained strictly before its target season. The base Ridge30 "
             "and no-continuity variants are diagnostics only, not a new validation search.", "",
             table([{k:v for k,v in row.items() if k != "missingness_strata"} for row in train["folds"]]), "",
             "Feature availability by fold:", "", table([dict(fold=row["fold"], variant=row["variant"], availability=label, **stats)
                  for row in train["folds"] for label, stats in row["missingness_strata"].items()]), "",
             "Training diagnostic interpretation: predicted residual SD is below actual SD in all six "
             "OLS folds, not only one era. Ridge30 narrows it further, but unregularized OLS also has the "
             "gap: regularization is not the sole explanation. Removing continuity does not resolve the "
             "gap, and does not consistently improve MAE. Higher missing-feature share is associated "
             "with weaker residual ranking in some later folds, but not uniformly higher MAE; these "
             "observational strata do not establish a causal missingness effect. The largest OLS MAE "
             "and p90 occur in 1995-96 through1999-00. This is persistent unexplained outcome variance "
             "with era-dependent offsets, not evidence for a universal tail expansion.", "",
             "## OOF Tail Calibration", "",
             f"{train['oof_rows']} genuine training OOF rows; {train['warmup_rows']} warmup-only rows. "
             "Source CSV: `data/simulation/defensive_oof_calibration_v1_training_oof.csv`, including each "
             "row's fold and latest base training year. No in-sample fitted predictions calibrate any coefficient.", "",
             "Let x=base_D Rtg-prior_league_anchor and y=actual_D Rtg-prior_league_anchor. Symmetric "
             "gamma=sum(x*y)/sum(x^2), through zero. Asymmetric gamma_better and gamma_worse fit the same "
             "equation separately on x<0 and x>=0; actual defensive buckets are never inference inputs. "
             "An intercept+slope regression is reported diagnostically, not silently added as another family.", "",
             "Calibration coefficients:", "", table([dict(candidate=name, **row["final_parameters"]) for name,row in train["candidates"].items()]), "",
             "Intercept diagnostic: `" + json.dumps(train["intercept_diagnostic"]) + "`", "",
             "Optional piecewise justification: `" + json.dumps(train["piecewise_evidence"]) + "`. "
             "If supported, use one continuous signed hinge at training-OOF75th percentile |x|, with both "
             "coefficients fit on training OOF rows. No isotonic/quantile mapping or outcome-bucket inference.", "",
             "## Defensive Residual Spread", "",
             "Pooled uncalibrated OOF diagnostics: `" + json.dumps(train["oof_summary"]) + "`", "",
             "A narrow predicted spread is not sufficient evidence that mean predictions should be expanded: "
             "conditional means can have lower variance than realized outcomes. Estimate gamma from OOF "
             "cross-products, NOT actual_SD/predicted_SD; do not force gamma>1 or match outcome variance.", "",
             "The pooled intercept is modest, but block offsets change sign and range from about-1.107 "
             "to+0.826 DRtg. A constant intercept is therefore not a supported era-stable remedy. "
             "Symmetric scaling is numerically stable but improves poor-defense bias in zero of five "
             "honest prequential blocks. The worse-defense asymmetric multiplier is unstable. Training "
             "does not support the optional piecewise expansion; no additional nonlinear family is tried.", "",
             "## Calibration Stability", "",
             "Per-fold multipliers below are descriptive fits to each held block's genuine OOF pairs. "
             "Honest prequential calibration performance instead fits a calibrator only to EARLIER OOF blocks "
             "and scores the next block; the first OOF block is meta-calibration warmup. Final production "
             "parameters pool all training OOF rows only.", "",
             table([dict(candidate=name, **params) for name,record in train["candidates"].items() for params in record["fold_parameters"]]), "",
             table([dict(candidate=name, **row) for name,record in train["candidates"].items() for row in record["stability"]["parameters"]]), "",
             "Stable eligibility: multipliers in[.25,2.5], fold coefficient of variation<=.35; in at least4/5 "
             "prequential blocks MAE must not regress>.15 or elite bias>.5 versus identity, and mean MAE "
             "regression must be<=.05. A nonidentity correction must not worsen poor bias>.25 in>=4/5 "
             "blocks and must improve it>.05 in>=2 blocks, rejecting a one-era tail fix. No coefficients "
             "are clipped to pass stability. All candidates/parameters/eligibility are fingerprinted before validation.", "",
             table([dict(candidate=name, **{k:v for k,v in row.items() if k != "parameters"})
                    for name,record in train["candidates"].items() for row in record["prequential"]]), "",
             "## Final Defensive Modeling Decision", ""]
    if result is None:
        lines += ["Validation pending; all calibration parameters fixed from TRAIN OOF only.", ""]
    else:
        lines += ["Existing validation evaluated exactly once after parameter/protocol freeze; no retuning. "
                  "All original hard gates and BOTH original/ranked bucket views are retained. Selection order: "
                  "passing gates, training stability, simpler family, then validation error. No frozen test run.", "",
                  table([dict(candidate=name, **row["legacy"]["diagnostics"]["overall"],
                              stable=train["candidates"][name]["stability"]["eligible"], accepted=row["accepted"])
                         for name,row in result["candidates"].items()]), "",
                  "Selected calibration: `" + json.dumps(result["parameters"]) + "`", "",
                  "Validation buckets:", "", table([dict(candidate=name, view=view, bucket=bucket, **stats)
                    for name,row in result["candidates"].items() for view in ["legacy", "ranking"]
                    for bucket,stats in row[view]["diagnostics"]["buckets"].items()]), "",
                  "Unchanged limits: `" + json.dumps(result["tolerances"]) + "`", "",
                  table([dict(candidate=name, view=view, **row[view]["checks"])
                    for name,row in result["candidates"].items() for view in ["legacy", "ranking"]]), "",
                  "One final product decision, not another optimization loop: **" + result["verdict"] + "**. "
                  "Product rule B is an explicit uncertainty-aware V1 exception under the user's stop rule, "
                  "NOT a changed hard gate: at most.25 DRtg beyond the3.0 original bucket-bias limit, all other "
                  "hard gates and ranked gates pass, validation residual Spearman>=.40, positive ranking in>=5/6 "
                  "training folds, stable selected formulation. Every actual hard-gate failure remains labeled FAIL.", "",
                  "Ironman diagnostic: `" + json.dumps(result["ironman"]) + "`", "",
                  "For unsupported synthetic rosters, preserve the base projection rather than amplify an "
                  "extrapolated residual. Record raw unguarded calibration only as a diagnostic, set effective "
                  "multiplier1, add an explicit HIGH uncertainty/coverage warning. Supported rosters use the "
                  "learned function normally; no universal min/max clamp. Continuity remains0 and no actual "
                  "team outcome/bucket is needed.", "",
                  "Integrity: `" + json.dumps(result["integrity"]) + "`. Official frozen test remains the "
                  "previous model's benchmark, not evidence for a newly fitted calibration. All prior results preserved.", "",
                  "Next product step: for A, game-log calibration; for B, an explicitly approved uncertainty-aware "
                  "V1 scope before game-level calibration; for C, restrict defensive claims rather than continue "
                  "the same validation search. No game logs, UI or application changes in this task.", ""]
    path = REPORT_DIR / "SIMULATION_ENGINE_DESIGN.md"
    original = path.read_text(encoding="utf-8").split("\n## Rolling-Origin Defensive Diagnostics")[0]
    path.write_text(original + "\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare", action="store_true")
    modes.add_argument("--validate", action="store_true")
    modes.add_argument("--report", action="store_true")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        if args.prepare:
            prepare_stage()
        elif args.validate:
            validate_stage()
        else:
            render_report()
