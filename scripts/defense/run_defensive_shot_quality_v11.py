#!/usr/bin/env python3
"""Defensive Shot Quality V1.1 robustness experiment.

Experimental only. Does not modify production inference, Decision Policy v1,
frontend, or the V1 defensive artifact.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[2]
sys.modules.setdefault("run_defensive_shot_quality_v11", sys.modules[__name__])
PRED_PATH = ROOT / "data" / "analysis" / "defense" / "defensive_shot_quality_predictions.parquet"
V1_ARTIFACT = ROOT / "artifacts" / "models" / "experimental_defensive_shot_quality_v1.joblib"
REPORT = ROOT / "reports" / "modeling" / "DEFENSIVE_SHOT_QUALITY_V11.md"
METRICS_JSON = ROOT / "reports" / "modeling" / "DEFENSIVE_SHOT_QUALITY_V11_METRICS.json"
POLICY_JSON = ROOT / "data" / "metadata" / "defensive_shot_quality_v11_feature_policy.json"
ARTIFACT = ROOT / "artifacts" / "models" / "experimental_defensive_shot_quality_v11.joblib"

RANDOM_SEED = 20261011
DISTANCES = [0.5, 1, 2, 3, 4, 5, 6, 7, 8, 10]
CORE_NUMERIC = [
    "shot_distance",
    "shot_value",
    "period",
    "seconds_remaining_period",
    "is_home",
    "closest_defender_distance",
    "shot_clock",
    "dribbles",
    "touch_time",
    "shot_clock_missing",
    "touch_time_missing",
    "touch_time_invalid",
]
CORE_CATEGORICAL = ["distance_bucket"]
SHOOTER_RATES = [
    "shooter_prior_fg_pct_shrunk",
    "shooter_prior_2p_pct_shrunk",
    "shooter_prior_3p_pct_shrunk",
    "shooter_distance_prior_fg_pct_shrunk",
]
ATTEMPT_COUNTS = [
    "shooter_prior_fga",
    "shooter_prior_2pa",
    "shooter_prior_3pa",
    "shooter_distance_prior_attempts",
]
DEFENDER_PT = [
    "defender_pt_overall_d_fg_pct",
    "defender_pt_overall_d_fga",
    "defender_pt_overall_normal_fg_pct",
    "defender_pt_overall_pct_plusminus",
]
VARIANTS = {
    "V11_A": {
        "label": "tracking + shooter rates only",
        "columns": CORE_NUMERIC + CORE_CATEGORICAL + SHOOTER_RATES,
        "categorical": CORE_CATEGORICAL,
        "deployable": True,
        "uses_attempt_counts": False,
        "uses_defender_profile": False,
    },
    "V11_B": {
        "label": "tracking + shooter rates + PT defender profile",
        "columns": CORE_NUMERIC + CORE_CATEGORICAL + SHOOTER_RATES + DEFENDER_PT,
        "categorical": CORE_CATEGORICAL,
        "deployable": True,
        "uses_attempt_counts": False,
        "uses_defender_profile": True,
    },
    "V11_C": {
        "label": "tracking + shooter rates + incompatible attempt counts",
        "columns": CORE_NUMERIC + CORE_CATEGORICAL + SHOOTER_RATES + ATTEMPT_COUNTS,
        "categorical": CORE_CATEGORICAL,
        "deployable": False,
        "uses_attempt_counts": True,
        "uses_defender_profile": False,
    },
}
MODEL_GRID = {
    "logistic": {"kind": "logistic"},
    "hgb_v1_like": {"kind": "hgb", "max_iter": 90, "learning_rate": 0.06, "l2_regularization": 0.02, "max_leaf_nodes": 31, "min_samples_leaf": 20},
    "hgb_regularized": {"kind": "hgb", "max_iter": 80, "learning_rate": 0.04, "l2_regularization": 0.25, "max_leaf_nodes": 15, "max_depth": 5, "min_samples_leaf": 80},
    "hgb_conservative": {"kind": "hgb", "max_iter": 70, "learning_rate": 0.035, "l2_regularization": 0.75, "max_leaf_nodes": 9, "max_depth": 4, "min_samples_leaf": 160},
}


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns: list[str]):
        self.columns = columns

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.columns].copy()


@dataclass
class V11Spec:
    key: str
    label: str
    columns: list[str]
    categorical: list[str]
    deployable: bool
    uses_attempt_counts: bool
    uses_defender_profile: bool


class V11CalibratedModel:
    def __init__(self, pipeline: Pipeline, calibration: str, calibrator: Any = None):
        self.pipeline = pipeline
        self.calibration = calibration
        self.calibrator = calibrator

    def raw_predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = np.clip(self.pipeline.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)
        return np.column_stack([1 - p, p])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = self.raw_predict_proba(X)[:, 1]
        if self.calibration == "sigmoid":
            z = np.log(p / (1 - p)).reshape(-1, 1)
            p = self.calibrator.predict_proba(z)[:, 1]
        elif self.calibration == "isotonic":
            p = self.calibrator.predict(p)
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.column_stack([1 - p, p])


DataFrameSelector.__module__ = "run_defensive_shot_quality_v11"
V11CalibratedModel.__module__ = "run_defensive_shot_quality_v11"
V11Spec.__module__ = "run_defensive_shot_quality_v11"


def make_pipeline(spec: V11Spec, model_name: str) -> Pipeline:
    numeric = [c for c in spec.columns if c not in spec.categorical]
    params = MODEL_GRID[model_name]
    if params["kind"] == "logistic":
        prep = ColumnTransformer(
            [
                ("num", Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True)), ("scale", StandardScaler())]), numeric),
                ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), spec.categorical),
            ],
            remainder="drop",
        )
        estimator = LogisticRegression(max_iter=1000, C=0.5, solver="lbfgs")
    else:
        prep = ColumnTransformer(
            [
                ("num", SimpleImputer(strategy="median", add_indicator=True), numeric),
                ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))]), spec.categorical),
            ],
            remainder="drop",
        )
        kwargs = {k: v for k, v in params.items() if k != "kind"}
        estimator = HistGradientBoostingClassifier(random_state=RANDOM_SEED, **kwargs)
    return Pipeline([("select", DataFrameSelector(spec.columns)), ("prep", prep), ("model", estimator)])


def metric_dict(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return {
        "rows": int(len(y)),
        "actual_fg_pct": float(np.mean(y)),
        "predicted_fg_pct": float(np.mean(p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        "ece": ece_score(y, p),
    }


def ece_score(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        if mask.any():
            total += float(mask.mean()) * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return total


def fit_candidate(train: pd.DataFrame, val: pd.DataFrame, spec: V11Spec, model_name: str) -> dict[str, V11CalibratedModel]:
    pipe = make_pipeline(spec, model_name)
    pipe.fit(train, train["made"].astype(int))
    raw_val = np.clip(pipe.predict_proba(val)[:, 1], 1e-6, 1 - 1e-6)
    y_val = val["made"].astype(int).to_numpy()
    models = {"uncalibrated": V11CalibratedModel(pipe, "uncalibrated")}
    sigmoid = LogisticRegression(max_iter=1000)
    sigmoid.fit(np.log(raw_val / (1 - raw_val)).reshape(-1, 1), y_val)
    models["sigmoid"] = V11CalibratedModel(pipe, "sigmoid", sigmoid)
    isotonic = IsotonicRegression(out_of_bounds="clip")
    isotonic.fit(raw_val, y_val)
    models["isotonic"] = V11CalibratedModel(pipe, "isotonic", isotonic)
    return models


def tail_stats(frame: pd.DataFrame, p: np.ndarray, prefix: str = "") -> dict[str, Any]:
    p = np.asarray(p, dtype=float)
    out = {
        f"{prefix}pct_lt_01": float((p < 0.01).mean()),
        f"{prefix}pct_gt_90": float((p > 0.90).mean()),
        f"{prefix}pct_gt_95": float((p > 0.95).mean()),
        f"{prefix}pct_gt_99": float((p > 0.99).mean()),
        f"{prefix}min_probability": float(p.min()),
        f"{prefix}max_probability": float(p.max()),
    }
    strata = {
        "near_rim_2pt": frame["shot_value"].eq(2) & frame["shot_distance"].le(4),
        "midrange_2pt": frame["shot_value"].eq(2) & frame["shot_distance"].between(10, 22),
        "three_pt": frame["shot_value"].eq(3),
    }
    for name, mask in strata.items():
        if not mask.any():
            continue
        sp = p[mask.to_numpy()]
        out[f"{prefix}{name}_pct_gt_90"] = float((sp > 0.90).mean())
        out[f"{prefix}{name}_pct_gt_95"] = float((sp > 0.95).mean())
        out[f"{prefix}{name}_pct_gt_99"] = float((sp > 0.99).mean())
        out[f"{prefix}{name}_max_probability"] = float(sp.max())
    return out


def training_stats(train: pd.DataFrame, columns: list[str]) -> dict[str, dict[str, float | None]]:
    stats = {}
    for col in columns:
        s = pd.to_numeric(train[col], errors="coerce").dropna().astype(float)
        stats[col] = {
            "min": float(s.min()) if not s.empty else None,
            "p1": float(s.quantile(0.01)) if not s.empty else None,
            "p99": float(s.quantile(0.99)) if not s.empty else None,
            "max": float(s.max()) if not s.empty else None,
            "median": float(s.median()) if not s.empty else None,
        }
    return stats


def support_diagnostic(row: dict[str, Any], stats: dict[str, dict[str, float | None]], sparse_context: bool = False, semantic_mismatch: bool = False) -> dict[str, Any]:
    warnings = []
    out_of_range = []
    for feature, st in stats.items():
        if feature not in row:
            continue
        value = row.get(feature)
        if value is None or pd.isna(value) or st["min"] is None:
            continue
        v = float(value)
        if v < float(st["min"]) or v > float(st["max"]):
            out_of_range.append({"feature": feature, "value": v, "min": st["min"], "max": st["max"], "severity": "outside_min_max"})
        elif v < float(st["p1"]) or v > float(st["p99"]):
            out_of_range.append({"feature": feature, "value": v, "p1": st["p1"], "p99": st["p99"], "severity": "outside_p1_p99"})
    if semantic_mismatch:
        warnings.append("Input contains features with known semantic mismatch versus 2014/15 season-to-date training policy.")
    if sparse_context:
        warnings.append("Nearest 2014/15 support for this scenario is sparse.")
    if out_of_range:
        warnings.append("One or more numeric features are outside training support thresholds.")
    if any(x["severity"] == "outside_min_max" for x in out_of_range) or semantic_mismatch:
        status = "OUT_OF_DISTRIBUTION"
    elif out_of_range or sparse_context:
        status = "LIMITED_SUPPORT"
    else:
        status = "IN_DISTRIBUTION"
    return {"supportStatus": status, "supportWarnings": warnings, "outOfRangeFeatures": out_of_range, "sparseContext": bool(sparse_context)}


def representative_rows(train: pd.DataFrame, selected_spec: V11Spec) -> dict[str, dict[str, Any]]:
    out = {}
    contexts = {
        "near_rim_2pt": train[train["shot_value"].eq(2) & train["shot_distance"].le(4)],
        "midrange_2pt": train[train["shot_value"].eq(2) & train["shot_distance"].between(12, 18)],
        "three_pt": train[train["shot_value"].eq(3) & train["shot_distance"].between(24, 28)],
    }
    targets = {
        "median": train["shooter_prior_2p_pct_shrunk"].median(),
        "strong_2pt": train["shooter_prior_2p_pct_shrunk"].quantile(0.80),
        "weak_2pt": train["shooter_prior_2p_pct_shrunk"].quantile(0.20),
    }
    for context_name, subset in contexts.items():
        if subset.empty:
            continue
        for profile_name, target in targets.items():
            row = subset.assign(_gap=(subset["shooter_prior_2p_pct_shrunk"] - target).abs()).sort_values("_gap").iloc[0]
            out[f"{profile_name}_{context_name}"] = {c: row[c] for c in selected_spec.columns}
    return out


def scenario_curve(model: V11CalibratedModel, base_row: dict[str, Any], stats: dict[str, dict[str, float | None]], semantic_mismatch: bool = False) -> list[dict[str, Any]]:
    rows = []
    for distance in DISTANCES:
        row = dict(base_row)
        row["closest_defender_distance"] = float(distance)
        frame = pd.DataFrame([row])
        raw = float(model.raw_predict_proba(frame)[0, 1])
        calibrated = float(model.predict_proba(frame)[0, 1])
        diag = support_diagnostic(row, stats, sparse_context=row.get("shot_value") == 2 and row.get("shot_distance", 99) <= 4 and distance >= 5, semantic_mismatch=semantic_mismatch)
        rows.append({"closest_defender_distance": distance, "raw_probability": raw, "calibrated_probability": calibrated, **diag})
    return rows


def modern_safe_row(df: pd.DataFrame, selected_spec: V11Spec, player_id: str, shot_distance: int, shot_value: int, defender_distance: float) -> dict[str, Any]:
    profiles = pd.read_parquet(ROOT / "artifacts" / "inference" / "player_profiles.parquet")
    profiles["player_id"] = profiles["player_id"].astype(str)
    prod_bucket = "0-3" if shot_distance <= 3 else "4-8" if shot_distance <= 8 else "9-16" if shot_distance <= 16 else "17-24" if shot_distance <= 24 else "25+"
    match = profiles[profiles["player_id"].eq(player_id) & profiles["distance_bucket"].astype(str).eq(prod_bucket)]
    if match.empty:
        match = profiles[profiles["player_id"].eq(player_id)]
    if match.empty:
        match = profiles[profiles["player_id"].eq("__COLD_START__")]
    p = match.iloc[0]
    row = {
        "shot_distance": shot_distance,
        "shot_value": shot_value,
        "period": 4,
        "seconds_remaining_period": 240.0,
        "is_home": True,
        "closest_defender_distance": defender_distance,
        "shot_clock": 12.0,
        "dribbles": 0,
        "touch_time": 1.2,
        "shot_clock_missing": False,
        "touch_time_missing": False,
        "touch_time_invalid": False,
        "distance_bucket": "0-4" if shot_distance <= 4 else "4-10" if shot_distance <= 10 else "10-16" if shot_distance <= 16 else "16-24" if shot_distance <= 23.75 else "24+",
        "shooter_prior_fg_pct_shrunk": float(p["player_prior_fg_pct_shrunk"]),
        "shooter_prior_2p_pct_shrunk": float(p["player_prior_2p_pct_shrunk"]),
        "shooter_prior_3p_pct_shrunk": float(p["player_prior_3p_pct_shrunk"]),
        "shooter_distance_prior_fg_pct_shrunk": float(p["player_distance_prior_fg_pct_shrunk"]),
    }
    if selected_spec.uses_defender_profile:
        for col in DEFENDER_PT:
            row[col] = float(df.loc[df["split"].eq("train"), col].median())
    return {c: row.get(c) for c in selected_spec.columns}


def candidate_penalty(row: dict[str, Any]) -> float:
    penalty = 0.0
    if row["uses_attempt_counts"]:
        penalty += 0.012
    if row["uses_defender_profile"]:
        penalty += 0.003
    if row["calibration"] == "isotonic":
        penalty += 0.004
    penalty += 0.120 * row.get("validation_near_rim_2pt_pct_gt_95", 0.0)
    penalty += 0.240 * row.get("validation_near_rim_2pt_pct_gt_99", 0.0)
    penalty += 0.060 * row.get("validation_pct_gt_95", 0.0)
    return penalty


def md_table(rows: list[dict[str, Any]], columns: list[str], digits: int = 6) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        vals = []
        for col in columns:
            value = row.get(col)
            if isinstance(value, float):
                vals.append(f"{value:.{digits}f}")
            elif isinstance(value, list):
                vals.append("; ".join(map(str, value)))
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> int:
    for path in [REPORT.parent, METRICS_JSON.parent, POLICY_JSON.parent, ARTIFACT.parent]:
        path.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(PRED_PATH)
    split = df.groupby("split").agg(rows=("made", "size"), games=("game_id", "nunique")).to_dict(orient="index")
    expected = {"train": {"rows": 90922, "games": 632}, "validation": {"rows": 19219, "games": 136}, "test": {"rows": 17928, "games": 136}}
    if {k: {"rows": int(v["rows"]), "games": int(v["games"])} for k, v in split.items()} != expected:
        raise RuntimeError(f"Split changed: {split}")
    train = df[df["split"].eq("train")].copy()
    val = df[df["split"].eq("validation")].copy()
    test = df[df["split"].eq("test")].copy()
    specs = {k: V11Spec(key=k, **v) for k, v in VARIANTS.items()}
    rows = []
    fitted: dict[str, V11CalibratedModel] = {}
    for spec in specs.values():
        for model_name in MODEL_GRID:
            print(f"Fitting {spec.key} {model_name}", flush=True)
            calibrated_models = fit_candidate(train, val, spec, model_name)
            for calibration, model in calibrated_models.items():
                key = f"{spec.key}_{model_name}_{calibration}"
                val_p = model.predict_proba(val)[:, 1]
                test_p = model.predict_proba(test)[:, 1]
                row = {
                    "candidate_key": key,
                    "variant": spec.key,
                    "variant_label": spec.label,
                    "model": model_name,
                    "calibration": calibration,
                    "deployable": spec.deployable,
                    "uses_attempt_counts": spec.uses_attempt_counts,
                    "uses_defender_profile": spec.uses_defender_profile,
                    **{f"validation_{k}": v for k, v in metric_dict(val["made"].to_numpy(), val_p).items()},
                    **tail_stats(val, val_p, "validation_"),
                    **{f"test_{k}": v for k, v in metric_dict(test["made"].to_numpy(), test_p).items()},
                    **tail_stats(test, test_p, "test_"),
                }
                row["selection_penalty"] = candidate_penalty(row)
                row["selection_score"] = row["validation_log_loss"] + row["selection_penalty"]
                rows.append(row)
                fitted[key] = model
    metrics = pd.DataFrame(rows)
    deployable = metrics[metrics["deployable"].astype(bool)].copy()
    selected_row = deployable.sort_values(["selection_score", "validation_log_loss", "validation_brier"]).iloc[0].to_dict()
    selected_key = str(selected_row["candidate_key"])
    selected_model = fitted[selected_key]
    selected_spec = specs[str(selected_row["variant"])]

    v1 = joblib.load(V1_ARTIFACT)
    v1_val_p = v1["model"].predict_proba(val)[:, 1]
    v1_test_p = v1["model"].predict_proba(test)[:, 1]
    v1_compare = {
        "selected_model_key": v1["selected_model_key"],
        **{f"validation_{k}": v for k, v in metric_dict(val["made"].to_numpy(), v1_val_p).items()},
        **tail_stats(val, v1_val_p, "validation_"),
        **{f"test_{k}": v for k, v in metric_dict(test["made"].to_numpy(), v1_test_p).items()},
        **tail_stats(test, v1_test_p, "test_"),
    }

    stats = training_stats(train, selected_spec.columns)
    rep_curves = []
    for name, base in representative_rows(train, selected_spec).items():
        for row in scenario_curve(selected_model, base, stats):
            rep_curves.append({"scenario": name, **row})

    jokic_rows = []
    for distance in [1, 3, 6, 8]:
        row = modern_safe_row(df, selected_spec, "203999", 3, 2, distance)
        p = float(selected_model.predict_proba(pd.DataFrame([row]))[0, 1])
        diag = support_diagnostic(row, stats, sparse_context=distance >= 5)
        jokic_rows.append({"distance": distance, "probability": p, **diag})
    curry_rows = []
    for distance in [1, 3, 6, 8]:
        row = modern_safe_row(df, selected_spec, "201939", 26, 3, distance)
        p = float(selected_model.predict_proba(pd.DataFrame([row]))[0, 1])
        diag = support_diagnostic(row, stats)
        curry_rows.append({"distance": distance, "probability": p, **diag})

    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": selected_model,
            "spec": selected_spec,
            "selected_candidate_key": selected_key,
            "version": "EXPERIMENTAL defensive-shot-quality-v1.1 NOT PRODUCTION",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "selection": selected_row,
        },
        ARTIFACT,
    )
    policy = {
        "version": "defensive_shot_quality_v11_experimental",
        "not_production": True,
        "selected_candidate_key": selected_key,
        "selected_variant": selected_spec.key,
        "selected_features": selected_spec.columns,
        "categorical_features": selected_spec.categorical,
        "excluded_attempt_counts": not selected_spec.uses_attempt_counts,
        "included_defender_profile": selected_spec.uses_defender_profile,
        "modern_adapter_policy": {
            "use_shooter_rate_priors": True,
            "use_shooter_attempt_counts": selected_spec.uses_attempt_counts,
            "use_tracking_context": True,
            "use_defender_pt_profile": selected_spec.uses_defender_profile,
            "fallback": "training medians for unsupported optional defender fields; support diagnostics for out-of-domain inputs",
        },
        "support_diagnostic": {
            "statuses": ["IN_DISTRIBUTION", "LIMITED_SUPPORT", "OUT_OF_DISTRIBUTION"],
            "no_numeric_confidence_score": True,
        },
    }
    POLICY_JSON.write_text(json.dumps(policy, indent=2, default=str), encoding="utf-8")

    metric_records = metrics.sort_values(["selection_score", "validation_log_loss"]).to_dict(orient="records")
    summary = {
        "split": expected,
        "variants": VARIANTS,
        "model_grid": MODEL_GRID,
        "selected": selected_row,
        "v1_comparison": v1_compare,
        "all_candidates": metric_records,
        "representative_curves": rep_curves,
        "jokic_modern_transfer": jokic_rows,
        "curry_modern_transfer": curry_rows,
        "artifact_path": str(ARTIFACT.relative_to(ROOT)),
        "policy_path": str(POLICY_JSON.relative_to(ROOT)),
    }
    METRICS_JSON.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    top_cols = ["candidate_key", "variant", "model", "calibration", "validation_log_loss", "validation_brier", "validation_ece", "test_log_loss", "test_brier", "test_ece", "validation_pct_gt_95", "validation_near_rim_2pt_pct_gt_95", "selection_score"]
    lines = [
        "# Defensive Shot Quality V1.1",
        "",
        "Experimental robustness/deployment experiment. Production model, Decision Policy v1, production API, and frontend were not modified.",
        "",
        "## Split",
        md_table([{"split": k, **v} for k, v in expected.items()], ["split", "rows", "games"], 3),
        "",
        "## Variants Tested",
        md_table(
            [
                {
                    "variant": k,
                    "label": v["label"],
                    "uses_attempt_counts": v["uses_attempt_counts"],
                    "uses_defender_profile": v["uses_defender_profile"],
                    "deployable": v["deployable"],
                }
                for k, v in VARIANTS.items()
            ],
            ["variant", "label", "uses_attempt_counts", "uses_defender_profile", "deployable"],
        ),
        "",
        "## Candidate Ranking",
        md_table(metrics.sort_values(["selection_score", "validation_log_loss"]).head(20).to_dict(orient="records"), top_cols),
        "",
        "## Selected Model",
        "```json",
        json.dumps(selected_row, indent=2, default=str),
        "```",
        "",
        "## Experimental V1 Comparison",
        "```json",
        json.dumps(v1_compare, indent=2, default=str),
        "```",
        "",
        "## Representative Support-Aware Curves",
        md_table(rep_curves, ["scenario", "closest_defender_distance", "raw_probability", "calibrated_probability", "supportStatus", "sparseContext"], 6),
        "",
        "## Modern Transfer Scenarios",
        "Jokic:",
        md_table(jokic_rows, ["distance", "probability", "supportStatus", "sparseContext", "supportWarnings"], 6),
        "",
        "Curry:",
        md_table(curry_rows, ["distance", "probability", "supportStatus", "sparseContext", "supportWarnings"], 6),
        "",
        "## Recommendation",
        "Use display policy A: show experimental P(make) normally only when IN_DISTRIBUTION, show a warning for LIMITED_SUPPORT, and hide absolute probability for OUT_OF_DISTRIBUTION while still allowing context/sensitivity exploration.",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"selected": selected_key, "artifact": str(ARTIFACT.relative_to(ROOT)), "report": str(REPORT.relative_to(ROOT))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
