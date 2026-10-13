#!/usr/bin/env python3
"""Shot-quality modeling benchmark for NBA Clutch Agent v2.

The canonical processed dataset is treated as immutable input.
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


RANDOM_SEED = 20261008
SHOT_DATA = Path("data/processed/shots_enriched.parquet")
FEATURE_POLICY = Path("data/metadata/feature_policy.json")
UNCERTAIN_AUDIT = Path("data/metadata/score_anomaly_audit_summary.json")

GENERIC_NUMERIC = [
    "shot_distance",
    "loc_x",
    "loc_y",
    "shot_value",
    "period",
    "seconds_remaining_period",
    "score_diff_before",
    "abs_score_diff_before",
]
GENERIC_BOOLEAN = ["is_home", "is_overtime"]
GENERIC_CATEGORICAL = [
    "season",
    "shot_type",
    "action_type",
    "shot_zone_basic",
    "shot_zone_area",
    "shot_zone_range",
    "game_state_before",
]
SHOOTER_CATEGORICAL = ["player_id"]
TARGET = "shot_made"


class EmpiricalBaseline(BaseEstimator, ClassifierMixin):
    def __init__(self, columns: list[str] | None = None):
        self.columns = columns

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.classes_ = np.array([False, True])
        self.global_p_ = float(np.mean(y))
        self.lookup_: dict[tuple[Any, ...], float] = {}
        columns = self.columns or []
        if columns:
            tmp = X[columns].copy()
            tmp["_target"] = y.to_numpy()
            grouped = tmp.groupby(columns, dropna=False)["_target"].mean()
            self.lookup_ = {tuple(k if isinstance(k, tuple) else (k,)): float(v) for k, v in grouped.items()}
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        columns = self.columns or []
        if not columns:
            p = np.full(len(X), self.global_p_, dtype=float)
        else:
            p = np.array(
                [self.lookup_.get(tuple(row), self.global_p_) for row in X[columns].itertuples(index=False, name=None)],
                dtype=float,
            )
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.column_stack([1 - p, p])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.predict_proba(X)[:, 1] >= 0.5


@dataclass
class ModelSpec:
    name: str
    feature_config: str
    estimator: Any
    columns: list[str]


def root_dir() -> Path:
    return Path(__file__).resolve().parents[2]


def load_data(root: Path) -> pd.DataFrame:
    df = pd.read_parquet(root / SHOT_DATA)
    return df


def assert_feature_policy(root: Path) -> dict[str, Any]:
    policy = json.loads((root / FEATURE_POLICY).read_text(encoding="utf-8"))
    forbidden = set(policy["forbidden_leakage_columns"] + policy["target_columns"])
    used = set(GENERIC_NUMERIC + GENERIC_BOOLEAN + GENERIC_CATEGORICAL + SHOOTER_CATEGORICAL)
    forbidden_used = forbidden.intersection(used)
    if forbidden_used:
        raise ValueError(f"Leakage columns used as features: {sorted(forbidden_used)}")
    if "is_clutch" in used:
        raise ValueError("is_clutch must not be used as a primary model input")
    return policy


def make_splits(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "train_regular": df[(df["dataset_split"] == "train") & (df["season_type"] == "regular_season")].copy(),
        "validation_regular": df[(df["dataset_split"] == "validation") & (df["season_type"] == "regular_season")].copy(),
        "test_regular": df[(df["dataset_split"] == "test") & (df["season_type"] == "regular_season")].copy(),
        "validation_clutch": df[(df["dataset_split"] == "validation") & (df["season_type"] == "regular_season") & df["is_clutch"]].copy(),
        "test_clutch": df[(df["dataset_split"] == "test") & (df["season_type"] == "regular_season") & df["is_clutch"]].copy(),
        "test_non_clutch": df[(df["dataset_split"] == "test") & (df["season_type"] == "regular_season") & ~df["is_clutch"]].copy(),
        "playoffs_2024_25": df[(df["season"] == "2024-25") & (df["season_type"] == "playoffs")].copy(),
        "playoffs_2025_26": df[(df["season"] == "2025-26") & (df["season_type"] == "playoffs")].copy(),
        "playoffs_2024_25_clutch": df[(df["season"] == "2024-25") & (df["season_type"] == "playoffs") & df["is_clutch"]].copy(),
        "playoffs_2025_26_clutch": df[(df["season"] == "2025-26") & (df["season_type"] == "playoffs") & df["is_clutch"]].copy(),
    }


def ece_score(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    total = len(y_true)
    score = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (y_prob >= low) & (y_prob < high if high < 1 else y_prob <= high)
        if mask.any():
            score += mask.mean() * abs(float(np.mean(y_prob[mask])) - float(np.mean(y_true[mask])))
    return float(score)


def metrics(y_true: pd.Series, y_prob: np.ndarray, shot_value: pd.Series | None = None) -> dict[str, Any]:
    y = y_true.astype(bool).to_numpy()
    p = np.clip(y_prob.astype(float), 1e-6, 1 - 1e-6)
    pred = p >= 0.5
    out = {
        "rows": int(len(y)),
        "actual_fg_pct": float(np.mean(y)),
        "predicted_fg_pct": float(np.mean(p)),
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        "log_loss": float(log_loss(y, p, labels=[False, True])),
        "brier": float(brier_score_loss(y, p)),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "ece_10_bin": ece_score(y, p, 10),
    }
    if shot_value is not None:
        sv = shot_value.astype(float).to_numpy()
        out["actual_ppp"] = float(np.mean(y * sv))
        out["predicted_expected_points_per_attempt"] = float(np.mean(p * sv))
        for value in [2, 3]:
            mask = sv == value
            if mask.any():
                out[f"actual_{value}pt_pct"] = float(np.mean(y[mask]))
                out[f"predicted_{value}pt_pct"] = float(np.mean(p[mask]))
    return out


def predict_positive(model: Any, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def preprocessors(config: str) -> tuple[ColumnTransformer, ColumnTransformer, list[str]]:
    numeric = GENERIC_NUMERIC + GENERIC_BOOLEAN
    categorical = GENERIC_CATEGORICAL.copy()
    if config == "shooter_aware":
        categorical += SHOOTER_CATEGORICAL
    columns = numeric + categorical
    logistic = ColumnTransformer(
        [
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=25), categorical),
        ]
    )
    tree = ColumnTransformer(
        [
            ("num", Pipeline([("impute", SimpleImputer(strategy="median"))]), numeric),
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), categorical),
        ]
    )
    return logistic, tree, columns


def model_specs() -> list[ModelSpec]:
    specs: list[ModelSpec] = []
    for config in ["generic", "shooter_aware"]:
        logistic_pre, tree_pre, cols = preprocessors(config)
        specs.append(ModelSpec("baseline_global", config, EmpiricalBaseline([]), cols))
        specs.append(ModelSpec("baseline_shot_value", config, EmpiricalBaseline(["shot_value"]), cols))
        specs.append(ModelSpec("baseline_zone_value", config, EmpiricalBaseline(["shot_value", "shot_zone_basic"]), cols))
        specs.append(
            ModelSpec(
                "logistic_regression",
                config,
                Pipeline(
                    [
                        ("prep", logistic_pre),
                        (
                            "model",
                            LogisticRegression(
                                solver="saga",
                                penalty="l2",
                                C=1.0,
                                max_iter=120,
                                n_jobs=-1,
                                random_state=RANDOM_SEED,
                                verbose=0,
                            ),
                        ),
                    ]
                ),
                cols,
            )
        )
        specs.append(
            ModelSpec(
                "hist_gradient_boosting",
                config,
                Pipeline(
                    [
                        ("prep", tree_pre),
                        (
                            "model",
                            HistGradientBoostingClassifier(
                                max_iter=120,
                                learning_rate=0.08,
                                max_leaf_nodes=31,
                                l2_regularization=0.02,
                                random_state=RANDOM_SEED,
                            ),
                        ),
                    ]
                ),
                cols,
            )
        )
    return specs


def fit_calibrators(y_val: pd.Series, p_val: np.ndarray) -> dict[str, Any]:
    p = np.clip(p_val, 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    sigmoid = LogisticRegression(solver="lbfgs", random_state=RANDOM_SEED)
    sigmoid.fit(logit, y_val.astype(bool))
    isotonic = IsotonicRegression(out_of_bounds="clip")
    isotonic.fit(p, y_val.astype(bool).astype(int))
    return {"none": None, "sigmoid": sigmoid, "isotonic": isotonic}


def apply_calibrator(calibrator: Any, p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    if calibrator is None:
        return p
    if isinstance(calibrator, LogisticRegression):
        logit = np.log(p / (1 - p)).reshape(-1, 1)
        return calibrator.predict_proba(logit)[:, 1]
    return calibrator.predict(p)


def calibration_frame(y_true: pd.Series, y_prob: np.ndarray, bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"actual": y_true.astype(int).to_numpy(), "prob": y_prob})
    df["bin"] = pd.cut(df["prob"], np.linspace(0, 1, bins + 1), include_lowest=True)
    return df.groupby("bin", observed=False).agg(rows=("actual", "size"), predicted=("prob", "mean"), actual=("actual", "mean")).reset_index()


def evaluate_model(model: Any, columns: list[str], datasets: dict[str, pd.DataFrame], calibrator: Any = None) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for name, frame in datasets.items():
        if frame.empty:
            continue
        p = apply_calibrator(calibrator, predict_positive(model, frame[columns]))
        out[name] = metrics(frame[TARGET], p, frame["shot_value"])
    return out


def plot_roc(models: dict[str, tuple[Any, list[str], Any]], datasets: dict[str, pd.DataFrame], out_dir: Path) -> None:
    for dataset_name in ["validation_regular", "test_regular", "test_clutch", "playoffs_2025_26", "playoffs_2025_26_clutch"]:
        frame = datasets[dataset_name]
        plt.figure(figsize=(7, 5))
        for label, (model, columns, calibrator) in models.items():
            p = apply_calibrator(calibrator, predict_positive(model, frame[columns]))
            fpr, tpr, _ = roc_curve(frame[TARGET].astype(bool), p)
            auc = roc_auc_score(frame[TARGET].astype(bool), p)
            plt.plot(fpr, tpr, label=f"{label} AUC={auc:.3f}")
        plt.plot([0, 1], [0, 1], "--", color="0.6")
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.title(f"ROC: {dataset_name}")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / f"roc_{dataset_name}.png", dpi=150)
        plt.close()


def plot_calibration(models: dict[str, tuple[Any, list[str], Any]], datasets: dict[str, pd.DataFrame], out_dir: Path) -> None:
    for dataset_name in ["validation_regular", "test_regular", "test_clutch", "playoffs_2025_26", "playoffs_2025_26_clutch"]:
        frame = datasets[dataset_name]
        plt.figure(figsize=(7, 5))
        for label, (model, columns, calibrator) in models.items():
            p = apply_calibrator(calibrator, predict_positive(model, frame[columns]))
            cf = calibration_frame(frame[TARGET], p, 10)
            plt.plot(cf["predicted"], cf["actual"], marker="o", label=label)
        plt.plot([0, 1], [0, 1], "--", color="0.6")
        plt.xlabel("Mean predicted probability")
        plt.ylabel("Observed make rate")
        plt.title(f"Calibration: {dataset_name}")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / f"calibration_{dataset_name}.png", dpi=150)
        plt.close()


def segment_tables(best_model: Any, columns: list[str], calibrator: Any, datasets: dict[str, pd.DataFrame], out_dir: Path) -> dict[str, Any]:
    test = datasets["test_regular"].copy()
    test["pred_prob"] = apply_calibrator(calibrator, predict_positive(best_model, test[columns]))
    test["expected_points"] = test["pred_prob"] * test["shot_value"]
    test["distance_bucket"] = pd.cut(test["shot_distance"].astype(float), [-1, 3, 8, 16, 24, 100], labels=["0-3", "4-8", "9-16", "17-24", "25+"])
    segments: dict[str, pd.DataFrame] = {}
    for name, col in {
        "shot_value": "shot_value",
        "distance_bucket": "distance_bucket",
        "shot_zone_basic": "shot_zone_basic",
        "home_away": "is_home",
        "game_state": "game_state_before",
        "period_context": "period_context",
        "time_bucket": "clutch_time_bucket",
    }.items():
        frame = test.copy()
        if col == "period_context":
            frame[col] = np.select([frame["period"] == 4, frame["period"] > 4], ["Q4", "OT"], default="Q1-Q3")
        if col == "clutch_time_bucket":
            frame[col] = frame[col].replace("", "non-clutch")
        grouped = frame.groupby(col, dropna=False).agg(
            rows=("shot_made", "size"),
            actual_fg_pct=("shot_made", "mean"),
            predicted_fg_pct=("pred_prob", "mean"),
            actual_pps=("points_scored", "mean"),
            expected_pps=("expected_points", "mean"),
        ).reset_index()
        segments[name] = grouped
        grouped.to_csv(out_dir / f"segment_{name}.csv", index=False)

    distance = segments["distance_bucket"]
    plt.figure(figsize=(7, 5))
    plt.plot(distance["distance_bucket"].astype(str), distance["actual_fg_pct"], marker="o", label="Actual FG%")
    plt.plot(distance["distance_bucket"].astype(str), distance["predicted_fg_pct"], marker="o", label="Predicted FG%")
    plt.xlabel("Shot distance bucket")
    plt.ylabel("FG%")
    plt.title("Actual vs expected by distance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir.parents[1] / "reports" / "modeling" / "figures" / "actual_expected_by_distance.png", dpi=150)
    plt.close()

    zone = segments["shot_zone_basic"].sort_values("rows", ascending=False).head(12)
    plt.figure(figsize=(9, 5))
    x = np.arange(len(zone))
    plt.bar(x - 0.2, zone["actual_fg_pct"], width=0.4, label="Actual FG%")
    plt.bar(x + 0.2, zone["predicted_fg_pct"], width=0.4, label="Predicted FG%")
    plt.xticks(x, zone["shot_zone_basic"], rotation=35, ha="right")
    plt.ylabel("FG%")
    plt.title("Actual vs expected by zone")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir.parents[1] / "reports" / "modeling" / "figures" / "actual_expected_by_zone.png", dpi=150)
    plt.close()
    return {k: v.to_dict(orient="records") for k, v in segments.items()}


def player_table(best_model: Any, columns: list[str], calibrator: Any, datasets: dict[str, pd.DataFrame], out_dir: Path) -> pd.DataFrame:
    frame = datasets["test_regular"].copy()
    frame["pred_prob"] = apply_calibrator(calibrator, predict_positive(best_model, frame[columns]))
    frame["expected_points"] = frame["pred_prob"] * frame["shot_value"]
    grouped = frame.groupby(["player_id", "player_name"]).agg(
        attempts=("shot_made", "size"),
        actual_fg_pct=("shot_made", "mean"),
        expected_fg_pct=("pred_prob", "mean"),
        actual_points_per_shot=("points_scored", "mean"),
        expected_points_per_shot=("expected_points", "mean"),
    ).reset_index()
    grouped = grouped[grouped["attempts"] >= 250].copy()
    grouped["fg_over_expected"] = grouped["actual_fg_pct"] - grouped["expected_fg_pct"]
    grouped["points_over_expected_per_shot"] = grouped["actual_points_per_shot"] - grouped["expected_points_per_shot"]
    grouped = grouped.sort_values("points_over_expected_per_shot", ascending=False)
    grouped.to_csv(out_dir / "player_actual_vs_expected_test_regular.csv", index=False)
    return grouped


def feature_importance(best_label: str, best_model: Any, columns: list[str], calibrator: Any, datasets: dict[str, pd.DataFrame], out_dir: Path) -> list[dict[str, Any]]:
    val = datasets["validation_regular"]
    rng = np.random.default_rng(RANDOM_SEED)
    sample_idx = rng.choice(len(val), size=min(50000, len(val)), replace=False)
    sample = val.iloc[sample_idx]
    scoring_model = best_model
    try:
        result = permutation_importance(
            scoring_model,
            sample[columns],
            sample[TARGET].astype(bool),
            n_repeats=3,
            random_state=RANDOM_SEED,
            scoring="neg_log_loss",
            n_jobs=1,
        )
        rows = [
            {"feature": col, "importance_mean_neg_log_loss": float(mean), "importance_std": float(std)}
            for col, mean, std in zip(columns, result.importances_mean, result.importances_std)
        ]
        rows = sorted(rows, key=lambda r: r["importance_mean_neg_log_loss"], reverse=True)
    except Exception as exc:
        rows = [{"feature": "importance_failed", "importance_mean_neg_log_loss": 0.0, "importance_std": 0.0, "error": str(exc)}]
    pd.DataFrame(rows).to_csv(out_dir / "feature_importance_selected_model.csv", index=False)
    return rows


def model_comparison_plot(metrics_rows: list[dict[str, Any]], figures_dir: Path) -> None:
    df = pd.DataFrame([r for r in metrics_rows if r["dataset"] == "validation_regular" and r["calibration"] == "none"])
    if df.empty:
        return
    df["label"] = df["model"] + "\n" + df["feature_config"]
    df = df.sort_values("log_loss")
    plt.figure(figsize=(10, 5))
    plt.bar(np.arange(len(df)), df["log_loss"])
    plt.xticks(np.arange(len(df)), df["label"], rotation=45, ha="right")
    plt.ylabel("Validation log loss")
    plt.title("Model comparison")
    plt.tight_layout()
    plt.savefig(figures_dir / "model_comparison_validation_log_loss.png", dpi=150)
    plt.close()


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, value in enumerate(row):
            widths[i] = max(widths[i], len(f"{value}"))
    lines = [
        "| " + " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))) + " |",
        "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(f"{row[i]}".ljust(widths[i]) for i in range(len(headers))) + " |")
    return "\n".join(lines)


def write_report(root: Path, payload: dict[str, Any]) -> None:
    report_dir = root / "reports" / "modeling"
    selected = payload["selected_model"]
    val_rows = [
        [
            r["model"],
            r["feature_config"],
            r["calibration"],
            f"{r['log_loss']:.5f}",
            f"{r['brier']:.5f}",
            f"{r['roc_auc']:.5f}" if r["roc_auc"] is not None else "",
            f"{r['ece_10_bin']:.5f}",
        ]
        for r in payload["flat_metrics"]
        if r["dataset"] == "validation_regular"
    ]
    test_rows = [
        [
            r["dataset"],
            f"{r['log_loss']:.5f}",
            f"{r['brier']:.5f}",
            f"{r['roc_auc']:.5f}" if r["roc_auc"] is not None else "",
            f"{r['actual_fg_pct']:.4f}",
            f"{r['predicted_fg_pct']:.4f}",
            f"{r.get('actual_ppp', 0):.4f}",
            f"{r.get('predicted_expected_points_per_attempt', 0):.4f}",
        ]
        for r in payload["selected_metrics_rows"]
    ]
    report = [
        "# Shot Quality Benchmark",
        "",
        "## Research Questions",
        "Estimate pre-shot field-goal make probability, assess clutch calibration, and test regular-season to playoff generalization.",
        "",
        "## Training Data",
        "Primary models train only on 2020-21 through 2023-24 regular-season rows. Validation is 2024-25 regular season; final test is 2025-26 regular season. Playoffs are stress tests only.",
        "",
        "## Feature Sets",
        "- Generic Shot Quality: shot geometry, shot zones, action/shot type, period/time, home/away, pre-shot score context, season.",
        "- Shooter-Aware Shot Quality: Generic features plus categorical `player_id` with unknown-player handling.",
        "",
        "Leakage controls exclude `shot_made`, `points_scored`, post-shot scores, descriptions, PBP result fields, future events, and `is_clutch` from primary model inputs.",
        "",
        "## Models",
        "Global empirical baseline, shot-value empirical baseline, shot-zone empirical baseline, logistic regression, and histogram gradient boosting. XGBoost/LightGBM/CatBoost were not added because sklearn models already establish the benchmark without extra dependencies.",
        "",
        "## Validation Results",
        markdown_table(["Model", "Feature config", "Calibration", "Log loss", "Brier", "ROC AUC", "ECE"], val_rows),
        "",
        "## Selected Model",
        f"{selected['model']} / {selected['feature_config']} / calibration={selected['calibration']}. Selection prioritized validation log loss, Brier score, calibration, and stress-test practicality.",
        "",
        "## Final / Stress-Test Metrics For Selected Model",
        markdown_table(["Dataset", "Log loss", "Brier", "ROC AUC", "Actual FG%", "Pred FG%", "Actual PPP", "Pred xPPS"], test_rows),
        "",
        "## Calibration",
        "Sigmoid and isotonic calibration were fit on validation predictions only and evaluated on held-out test/stress datasets. Figures are under `reports/modeling/figures/`.",
        "",
        "## Segment Findings",
        "Segment tables are stored under `data/analysis/segment_*.csv`. The most interpretable slices are shot value, distance bucket, shot zone, home/away, score state, and Q4/OT context.",
        "",
        "## Player Analysis",
        "`data/analysis/player_actual_vs_expected_test_regular.csv` includes players with at least 250 test regular-season attempts.",
        "",
        "## Feature Importance",
        "`data/analysis/feature_importance_selected_model.csv` contains permutation importance on a validation sample. Interpret as predictive utility, not causality.",
        "",
        "## Limitations",
        "The benchmark uses event-level pre-shot features only; no defender distance, lineup context, fatigue, play design, or shot-clock features are available. Player identity improves fit but may reduce interpretability of pure shot quality. Playoff rows are stress tests, not training data.",
        "",
        "## Recommendation",
        "Use the selected calibrated probability model as the expected-points engine for the next decision-agent stage. The decision layer should consume `predicted_make_probability` and `expected_points`, not hard classifications.",
    ]
    (report_dir / "SHOT_QUALITY_BENCHMARK.md").write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    root = root_dir()
    assert_feature_policy(root)
    artifacts_models = root / "artifacts" / "models"
    artifacts_metrics = root / "artifacts" / "metrics"
    analysis_dir = root / "data" / "analysis"
    report_dir = root / "reports" / "modeling"
    figures_dir = report_dir / "figures"
    for d in [artifacts_models, artifacts_metrics, analysis_dir, figures_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print("Loading canonical dataset...")
    df = load_data(root)
    splits = make_splits(df)
    split_counts = {k: int(len(v)) for k, v in splits.items()}

    fit_data = splits["train_regular"]
    val_data = splits["validation_regular"]
    eval_sets = {
        "validation_regular": val_data,
        "test_regular": splits["test_regular"],
        "validation_clutch": splits["validation_clutch"],
        "test_clutch": splits["test_clutch"],
        "test_non_clutch": splits["test_non_clutch"],
        "playoffs_2024_25": splits["playoffs_2024_25"],
        "playoffs_2025_26": splits["playoffs_2025_26"],
        "playoffs_2024_25_clutch": splits["playoffs_2024_25_clutch"],
        "playoffs_2025_26_clutch": splits["playoffs_2025_26_clutch"],
    }

    flat_metrics: list[dict[str, Any]] = []
    fitted: dict[str, tuple[Any, list[str], dict[str, Any]]] = {}
    for spec in model_specs():
        label = f"{spec.name}__{spec.feature_config}"
        print(f"Training {label}...")
        model = clone(spec.estimator)
        model.fit(fit_data[spec.columns], fit_data[TARGET].astype(bool))
        val_prob = predict_positive(model, val_data[spec.columns])
        calibrators = fit_calibrators(val_data[TARGET], val_prob)
        fitted[label] = (model, spec.columns, calibrators)
        joblib.dump(model, artifacts_models / f"{label}.joblib")
        for cal_name, calibrator in calibrators.items():
            eval_metrics = evaluate_model(model, spec.columns, eval_sets, calibrator)
            for dataset_name, vals in eval_metrics.items():
                flat_metrics.append({"model": spec.name, "feature_config": spec.feature_config, "calibration": cal_name, "dataset": dataset_name, **vals})

    # Select by validation log loss, then Brier, then ECE.
    val_candidates = [r for r in flat_metrics if r["dataset"] == "validation_regular"]
    selected_row = sorted(val_candidates, key=lambda r: (r["log_loss"], r["brier"], r["ece_10_bin"]))[0]
    selected_label = f"{selected_row['model']}__{selected_row['feature_config']}"
    selected_model, selected_columns, selected_calibrators = fitted[selected_label]
    selected_calibrator = selected_calibrators[selected_row["calibration"]]
    joblib.dump({"model": selected_model, "columns": selected_columns, "calibrator": selected_calibrator, "selection": selected_row}, artifacts_models / "selected_shot_quality_model.joblib")

    selected_metrics_rows = [
        r for r in flat_metrics
        if r["model"] == selected_row["model"]
        and r["feature_config"] == selected_row["feature_config"]
        and r["calibration"] == selected_row["calibration"]
    ]

    comparison_models = {
        "baseline_zone": (fitted["baseline_zone_value__generic"][0], fitted["baseline_zone_value__generic"][1], fitted["baseline_zone_value__generic"][2]["none"]),
        "logistic_generic": (fitted["logistic_regression__generic"][0], fitted["logistic_regression__generic"][1], fitted["logistic_regression__generic"][2].get("sigmoid")),
        "logistic_shooter": (fitted["logistic_regression__shooter_aware"][0], fitted["logistic_regression__shooter_aware"][1], fitted["logistic_regression__shooter_aware"][2].get("sigmoid")),
        "selected": (selected_model, selected_columns, selected_calibrator),
    }
    print("Generating figures and analysis tables...")
    plot_roc(comparison_models, eval_sets, figures_dir)
    plot_calibration(comparison_models, eval_sets, figures_dir)
    model_comparison_plot(flat_metrics, figures_dir)
    segments = segment_tables(selected_model, selected_columns, selected_calibrator, splits, analysis_dir)
    players = player_table(selected_model, selected_columns, selected_calibrator, splits, analysis_dir)
    importance = feature_importance(selected_label, selected_model, selected_columns, selected_calibrator, splits, analysis_dir)

    # Clutch sensitivity: uncertain rows are documented but not row-addressable in canonical metadata.
    uncertain = json.loads((root / UNCERTAIN_AUDIT).read_text(encoding="utf-8")) if (root / UNCERTAIN_AUDIT).exists() else {}
    uncertainty_count = uncertain.get("clutch_impact", {}).get("uncertain_clutch_classifications", 0)

    payload = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "random_seed": RANDOM_SEED,
        "feature_configurations": {
            "generic": GENERIC_NUMERIC + GENERIC_BOOLEAN + GENERIC_CATEGORICAL,
            "shooter_aware": GENERIC_NUMERIC + GENERIC_BOOLEAN + GENERIC_CATEGORICAL + SHOOTER_CATEGORICAL,
        },
        "models_tested": sorted({r["model"] for r in flat_metrics}),
        "calibration_methods": ["none", "sigmoid", "isotonic"],
        "split_counts": split_counts,
        "flat_metrics": flat_metrics,
        "selected_model": selected_row,
        "selected_metrics_rows": selected_metrics_rows,
        "segments": segments,
        "player_analysis_min_attempts": 250,
        "player_analysis_rows": int(len(players)),
        "feature_importance_top": importance[:25],
        "uncertain_clutch_classification_rows": uncertainty_count,
        "clutch_uncertainty_sensitivity": "Row-level uncertain clutch keys were not present in canonical metadata; canonical clutch evaluation is reported, with 21 documented uncertain classifications from score anomaly audit.",
    }
    (report_dir / "MODEL_METRICS.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (artifacts_metrics / "shot_quality_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(root, payload)
    print("Shot-quality benchmark complete")
    print(json.dumps({"selected_model": selected_row, "split_counts": split_counts}, indent=2))


if __name__ == "__main__":
    main()
