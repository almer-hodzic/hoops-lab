#!/usr/bin/env python3
"""Shooter Skill v1.1 benchmark.

Builds leakage-safe historical shooter-profile features and compares them with
generic shot quality and raw player-ID models.
"""

from __future__ import annotations

import json
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
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


RANDOM_SEED = 20261008
TARGET = "shot_made"
PRIMARY_TRAIN_SEASONS = ["2020-21", "2021-22", "2022-23", "2023-24"]

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
RAW_PLAYER_ID = ["player_id"]
SKILL_FEATURES = [
    "player_prior_fga",
    "player_prior_fg_pct",
    "player_prior_fg_pct_shrunk",
    "player_prior_2pa",
    "player_prior_2p_pct",
    "player_prior_2p_pct_shrunk",
    "player_prior_3pa",
    "player_prior_3p_pct",
    "player_prior_3p_pct_shrunk",
    "player_zone_prior_attempts",
    "player_zone_prior_fg_pct",
    "player_zone_prior_fg_pct_shrunk",
    "player_distance_prior_attempts",
    "player_distance_prior_fg_pct",
    "player_distance_prior_fg_pct_shrunk",
    "player_season_prior_fga",
    "player_season_prior_fg_pct",
    "player_season_prior_fg_pct_shrunk",
    "player_previous_season_fga",
    "player_previous_season_fg_pct",
    "player_previous_season_fg_pct_shrunk",
    "player_previous_season_2pa",
    "player_previous_season_2p_pct_shrunk",
    "player_previous_season_3pa",
    "player_previous_season_3p_pct_shrunk",
    "player_cold_start",
]


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
class Spec:
    model_name: str
    feature_config: str
    estimator: Any
    columns: list[str]


def root_dir() -> Path:
    return Path(__file__).resolve().parents[2]


def ece_score(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    score = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (y_prob >= low) & (y_prob < high if high < 1 else y_prob <= high)
        if mask.any():
            score += mask.mean() * abs(float(np.mean(y_prob[mask])) - float(np.mean(y_true[mask])))
    return float(score)


def metrics(frame: pd.DataFrame, p: np.ndarray) -> dict[str, Any]:
    y = frame[TARGET].astype(bool).to_numpy()
    p = np.clip(p.astype(float), 1e-6, 1 - 1e-6)
    pred = p >= 0.5
    shot_value = frame["shot_value"].astype(float).to_numpy()
    return {
        "rows": int(len(frame)),
        "actual_fg_pct": float(np.mean(y)),
        "predicted_fg_pct": float(np.mean(p)),
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        "log_loss": float(log_loss(y, p, labels=[False, True])),
        "brier": float(brier_score_loss(y, p)),
        "accuracy": float(accuracy_score(y, pred)),
        "ece_10_bin": ece_score(y, p),
        "actual_ppp": float(np.mean(y * shot_value)),
        "predicted_expected_points_per_attempt": float(np.mean(p * shot_value)),
    }


def add_prior_stats(
    df: pd.DataFrame,
    group_cols: list[str],
    attempt_mask: pd.Series,
    make_mask: pd.Series,
    attempts_col: str,
    pct_col: str,
    shrunk_col: str,
    prior_mean: float,
    strength: float,
) -> None:
    attempt = attempt_mask.astype("int32")
    made = make_mask.astype("int32")
    prior_attempts = attempt.groupby([df[c] for c in group_cols], sort=False).cumsum() - attempt
    prior_makes = made.groupby([df[c] for c in group_cols], sort=False).cumsum() - made
    df[attempts_col] = prior_attempts.astype("int32")
    df[pct_col] = np.where(prior_attempts > 0, prior_makes / prior_attempts, np.nan)
    df[shrunk_col] = (prior_makes + strength * prior_mean) / (prior_attempts + strength)


def build_historical_features(df: pd.DataFrame) -> pd.DataFrame:
    print("Building leakage-safe historical shooter features...")
    df = df.copy()
    df["game_date_dt"] = pd.to_datetime(df["game_date"])
    df["event_id_num"] = pd.to_numeric(df["event_id"], errors="coerce").fillna(-1).astype("int64")
    df["distance_bucket"] = pd.cut(
        df["shot_distance"].astype(float),
        [-1, 3, 8, 16, 24, 100],
        labels=["0-3", "4-8", "9-16", "17-24", "25+"],
    ).astype(str)
    df = df.sort_values(["game_date_dt", "game_id", "event_id_num"], kind="mergesort").reset_index(drop=True)

    train = df[(df["season"].isin(PRIMARY_TRAIN_SEASONS)) & (df["season_type"] == "regular_season")]
    global_fg = float(train[TARGET].mean())
    global_2p = float(train.loc[train["shot_value"] == 2, TARGET].mean())
    global_3p = float(train.loc[train["shot_value"] == 3, TARGET].mean())
    zone_mean = train.groupby("shot_zone_basic")[TARGET].mean().to_dict()
    dist_mean = train.groupby("distance_bucket")[TARGET].mean().to_dict()

    all_attempt = pd.Series(True, index=df.index)
    all_make = df[TARGET].astype(bool)
    two_attempt = df["shot_value"].eq(2)
    two_make = two_attempt & all_make
    three_attempt = df["shot_value"].eq(3)
    three_make = three_attempt & all_make

    add_prior_stats(df, ["player_id"], all_attempt, all_make, "player_prior_fga", "player_prior_fg_pct", "player_prior_fg_pct_shrunk", global_fg, 150)
    add_prior_stats(df, ["player_id"], two_attempt, two_make, "player_prior_2pa", "player_prior_2p_pct", "player_prior_2p_pct_shrunk", global_2p, 100)
    add_prior_stats(df, ["player_id"], three_attempt, three_make, "player_prior_3pa", "player_prior_3p_pct", "player_prior_3p_pct_shrunk", global_3p, 100)
    add_prior_stats(
        df,
        ["player_id", "shot_zone_basic"],
        all_attempt,
        all_make,
        "player_zone_prior_attempts",
        "player_zone_prior_fg_pct",
        "player_zone_prior_fg_pct_shrunk",
        global_fg,
        40,
    )
    zone_prior = df["shot_zone_basic"].map(zone_mean).fillna(global_fg)
    df["player_zone_prior_fg_pct_shrunk"] = (
        df["player_zone_prior_fg_pct"].fillna(zone_prior) * df["player_zone_prior_attempts"] + 40 * zone_prior
    ) / (df["player_zone_prior_attempts"] + 40)
    add_prior_stats(
        df,
        ["player_id", "distance_bucket"],
        all_attempt,
        all_make,
        "player_distance_prior_attempts",
        "player_distance_prior_fg_pct",
        "player_distance_prior_fg_pct_shrunk",
        global_fg,
        40,
    )
    dist_prior = df["distance_bucket"].map(dist_mean).fillna(global_fg)
    df["player_distance_prior_fg_pct_shrunk"] = (
        df["player_distance_prior_fg_pct"].fillna(dist_prior) * df["player_distance_prior_attempts"] + 40 * dist_prior
    ) / (df["player_distance_prior_attempts"] + 40)
    add_prior_stats(
        df,
        ["season", "player_id"],
        all_attempt,
        all_make,
        "player_season_prior_fga",
        "player_season_prior_fg_pct",
        "player_season_prior_fg_pct_shrunk",
        global_fg,
        100,
    )

    # Previous-season profile: complete prior season only, mapped to current season rows.
    season_order = sorted(df["season"].unique())
    totals = df.groupby(["season", "player_id"]).agg(
        fga=("shot_made", "size"),
        fgm=("shot_made", "sum"),
        two_pa=("shot_value", lambda s: int((s == 2).sum())),
        two_pm=("shot_made", lambda s: 0),  # overwritten below
        three_pa=("shot_value", lambda s: int((s == 3).sum())),
        three_pm=("shot_made", lambda s: 0),
    ).reset_index()
    made_two = df[df["shot_value"] == 2].groupby(["season", "player_id"])["shot_made"].sum()
    made_three = df[df["shot_value"] == 3].groupby(["season", "player_id"])["shot_made"].sum()
    totals["two_pm"] = [int(made_two.get((r.season, r.player_id), 0)) for r in totals.itertuples(index=False)]
    totals["three_pm"] = [int(made_three.get((r.season, r.player_id), 0)) for r in totals.itertuples(index=False)]
    prev_rows = []
    for season, prev in zip(season_order[1:], season_order[:-1]):
        tmp = totals[totals["season"] == prev].copy()
        tmp["season"] = season
        prev_rows.append(tmp)
    prev = pd.concat(prev_rows, ignore_index=True) if prev_rows else totals.iloc[0:0].copy()
    prev["player_previous_season_fga"] = prev["fga"]
    prev["player_previous_season_fg_pct"] = prev["fgm"] / prev["fga"]
    prev["player_previous_season_fg_pct_shrunk"] = (prev["fgm"] + 200 * global_fg) / (prev["fga"] + 200)
    prev["player_previous_season_2pa"] = prev["two_pa"]
    prev["player_previous_season_2p_pct_shrunk"] = (prev["two_pm"] + 120 * global_2p) / (prev["two_pa"] + 120)
    prev["player_previous_season_3pa"] = prev["three_pa"]
    prev["player_previous_season_3p_pct_shrunk"] = (prev["three_pm"] + 120 * global_3p) / (prev["three_pa"] + 120)
    prev = prev[
        [
            "season",
            "player_id",
            "player_previous_season_fga",
            "player_previous_season_fg_pct",
            "player_previous_season_fg_pct_shrunk",
            "player_previous_season_2pa",
            "player_previous_season_2p_pct_shrunk",
            "player_previous_season_3pa",
            "player_previous_season_3p_pct_shrunk",
        ]
    ]
    df = df.merge(prev, on=["season", "player_id"], how="left")

    for col in SKILL_FEATURES:
        if col == "player_cold_start":
            continue
        if col.endswith("_pct"):
            df[col] = df[col].fillna(global_fg)
        elif col.endswith("_2p_pct"):
            df[col] = df[col].fillna(global_2p)
        elif col.endswith("_3p_pct"):
            df[col] = df[col].fillna(global_3p)
        else:
            df[col] = df[col].fillna(0)
    df["player_cold_start"] = df["player_prior_fga"].eq(0)
    return df.drop(columns=["game_date_dt", "event_id_num"])


def make_splits(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    val_regular = df[(df["season"] == "2024-25") & (df["season_type"] == "regular_season")].sort_values(["game_date", "game_id", "event_id"])
    midpoint = len(val_regular) // 2
    return {
        "train_regular": df[(df["season"].isin(PRIMARY_TRAIN_SEASONS)) & (df["season_type"] == "regular_season")].copy(),
        "selection_regular_2024_25_early": val_regular.iloc[:midpoint].copy(),
        "calibration_regular_2024_25_late": val_regular.iloc[midpoint:].copy(),
        "full_validation_regular_2024_25": val_regular.copy(),
        "test_regular_2025_26": df[(df["season"] == "2025-26") & (df["season_type"] == "regular_season")].copy(),
        "test_clutch_2025_26": df[(df["season"] == "2025-26") & (df["season_type"] == "regular_season") & df["is_clutch"]].copy(),
        "test_non_clutch_2025_26": df[(df["season"] == "2025-26") & (df["season_type"] == "regular_season") & ~df["is_clutch"]].copy(),
        "playoffs_2024_25": df[(df["season"] == "2024-25") & (df["season_type"] == "playoffs")].copy(),
        "playoffs_2025_26": df[(df["season"] == "2025-26") & (df["season_type"] == "playoffs")].copy(),
        "playoffs_2025_26_clutch": df[(df["season"] == "2025-26") & (df["season_type"] == "playoffs") & df["is_clutch"]].copy(),
    }


def build_transformer(columns: list[str], categorical: list[str]) -> ColumnTransformer:
    numeric = [c for c in columns if c not in categorical]
    return ColumnTransformer(
        [
            ("num", Pipeline([("impute", SimpleImputer(strategy="median"))]), numeric),
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), categorical),
        ]
    )


def specs() -> list[Spec]:
    generic_cols = GENERIC_NUMERIC + GENERIC_BOOLEAN + GENERIC_CATEGORICAL
    raw_cols = generic_cols + RAW_PLAYER_ID
    skill_cols = generic_cols + SKILL_FEATURES
    skill_raw_cols = generic_cols + SKILL_FEATURES + RAW_PLAYER_ID
    config = {
        "generic": (generic_cols, GENERIC_CATEGORICAL),
        "raw_player_id": (raw_cols, GENERIC_CATEGORICAL + RAW_PLAYER_ID),
        "historical_shooter_profile": (skill_cols, GENERIC_CATEGORICAL),
        "historical_profile_plus_raw_id": (skill_raw_cols, GENERIC_CATEGORICAL + RAW_PLAYER_ID),
    }
    out = [
        Spec("baseline_global", "generic", EmpiricalBaseline([]), generic_cols),
        Spec("baseline_shot_value", "generic", EmpiricalBaseline(["shot_value"]), generic_cols),
    ]
    for name, (cols, cats) in config.items():
        out.append(
            Spec(
                "hist_gradient_boosting",
                name,
                Pipeline(
                    [
                        ("prep", build_transformer(cols, cats)),
                        (
                            "model",
                            HistGradientBoostingClassifier(
                                max_iter=130,
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
    return out


def predict(model: Any, frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    return np.clip(model.predict_proba(frame[columns])[:, 1], 1e-6, 1 - 1e-6)


def fit_calibration(y: pd.Series, p: np.ndarray) -> dict[str, Any]:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    sigmoid = LogisticRegression(solver="lbfgs", random_state=RANDOM_SEED).fit(logit, y.astype(bool))
    iso = IsotonicRegression(out_of_bounds="clip").fit(p, y.astype(int))
    return {"none": None, "sigmoid": sigmoid, "isotonic": iso}


def apply_calibration(p: np.ndarray, calibrator: Any) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    if calibrator is None:
        return p
    if isinstance(calibrator, LogisticRegression):
        logit = np.log(p / (1 - p)).reshape(-1, 1)
        return calibrator.predict_proba(logit)[:, 1]
    return np.clip(calibrator.predict(p), 1e-6, 1 - 1e-6)


def evaluate(model: Any, columns: list[str], calibrator: Any, datasets: dict[str, pd.DataFrame]) -> dict[str, dict[str, Any]]:
    out = {}
    for name, frame in datasets.items():
        if frame.empty:
            continue
        out[name] = metrics(frame, apply_calibration(predict(model, frame, columns), calibrator))
    return out


def calibration_frame(frame: pd.DataFrame, p: np.ndarray) -> pd.DataFrame:
    tmp = pd.DataFrame({"actual": frame[TARGET].astype(int).to_numpy(), "prob": p})
    tmp["bin"] = pd.cut(tmp["prob"], np.linspace(0, 1, 11), include_lowest=True)
    return tmp.groupby("bin", observed=False).agg(rows=("actual", "size"), predicted=("prob", "mean"), actual=("actual", "mean")).reset_index()


def save_figures(root: Path, selected: tuple[Any, list[str], Any], datasets: dict[str, pd.DataFrame], label: str) -> None:
    fig_dir = root / "reports" / "modeling" / "figures" / "shooter_skill_v11"
    fig_dir.mkdir(parents=True, exist_ok=True)
    model, cols, cal = selected
    for name in ["selection_regular_2024_25_early", "test_regular_2025_26", "test_clutch_2025_26", "playoffs_2025_26"]:
        frame = datasets[name]
        p = apply_calibration(predict(model, frame, cols), cal)
        fpr, tpr, _ = roc_curve(frame[TARGET].astype(bool), p)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f"AUC={roc_auc_score(frame[TARGET].astype(bool), p):.3f}")
        plt.plot([0, 1], [0, 1], "--", color="0.6")
        plt.title(f"Shooter Skill v1.1 ROC: {name}")
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / f"roc_{label}_{name}.png", dpi=150)
        plt.close()

        cf = calibration_frame(frame, p)
        plt.figure(figsize=(6, 5))
        plt.plot(cf["predicted"], cf["actual"], marker="o")
        plt.plot([0, 1], [0, 1], "--", color="0.6")
        plt.title(f"Shooter Skill v1.1 Calibration: {name}")
        plt.xlabel("Mean predicted probability")
        plt.ylabel("Observed make rate")
        plt.tight_layout()
        plt.savefig(fig_dir / f"calibration_{label}_{name}.png", dpi=150)
        plt.close()


def representative_player_checks(root: Path, model: Any, cols: list[str], cal: Any, data: pd.DataFrame) -> pd.DataFrame:
    test = data[(data["season"] == "2025-26") & (data["season_type"] == "regular_season")].copy()
    test["pred_prob"] = apply_calibration(predict(model, test, cols), cal)
    players = (
        test.groupby(["player_id", "player_name"])
        .agg(attempts=("shot_made", "size"), actual_fg_pct=("shot_made", "mean"), predicted_fg_pct=("pred_prob", "mean"), avg_prior_fga=("player_prior_fga", "mean"), avg_prior_fg_pct=("player_prior_fg_pct_shrunk", "mean"))
        .reset_index()
    )
    players = players[players["attempts"] >= 300].copy()
    # Representative rather than leaderboard-only: high, middle, low predicted groups.
    ordered = players.sort_values("predicted_fg_pct")
    sample = pd.concat([ordered.head(8), ordered.iloc[len(ordered)//2-4:len(ordered)//2+4], ordered.tail(8)]).drop_duplicates("player_id")
    out = root / "data" / "analysis" / "shooter_skill_v11_player_sanity_checks.csv"
    sample.to_csv(out, index=False)
    return sample


def feature_importance(root: Path, model: Any, cols: list[str], selection: pd.DataFrame) -> list[dict[str, Any]]:
    sample = selection.sample(n=min(40000, len(selection)), random_state=RANDOM_SEED)
    result = permutation_importance(model, sample[cols], sample[TARGET].astype(bool), scoring="neg_log_loss", n_repeats=3, random_state=RANDOM_SEED, n_jobs=1)
    rows = [
        {"feature": col, "importance_neg_log_loss": float(mean), "std": float(std)}
        for col, mean, std in zip(cols, result.importances_mean, result.importances_std)
    ]
    rows = sorted(rows, key=lambda r: r["importance_neg_log_loss"], reverse=True)
    pd.DataFrame(rows).to_csv(root / "data" / "analysis" / "shooter_skill_v11_feature_importance.csv", index=False)
    return rows


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            widths[i] = max(widths[i], len(str(val)))
    lines = [
        "| " + " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))) + " |",
        "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[i]).ljust(widths[i]) for i in range(len(headers))) + " |")
    return "\n".join(lines)


def write_report(root: Path, payload: dict[str, Any]) -> None:
    report = root / "reports" / "modeling" / "SHOOTER_SKILL_EXPERIMENT.md"
    selection_rows = [
        [r["model"], r["feature_config"], r["calibration"], f"{r['log_loss']:.5f}", f"{r['brier']:.5f}", f"{r['roc_auc']:.5f}", f"{r['ece_10_bin']:.5f}"]
        for r in payload["flat_metrics"]
        if r["dataset"] == "selection_regular_2024_25_early"
    ]
    selected_rows = [
        [r["dataset"], f"{r['log_loss']:.5f}", f"{r['brier']:.5f}", f"{r['roc_auc']:.5f}", f"{r['actual_fg_pct']:.4f}", f"{r['predicted_fg_pct']:.4f}", f"{r['actual_ppp']:.4f}", f"{r['predicted_expected_points_per_attempt']:.4f}"]
        for r in payload["selected_metrics"]
    ]
    text = [
        "# Shooter Skill v1.1 Experiment",
        "",
        "## Method",
        "Historical shooter-profile features are computed chronologically from shots strictly before the current attempt. The canonical dataset is read-only. Training uses 2020-21 through 2023-24 regular season only. The 2024-25 regular season is split chronologically: the earlier half is used for model selection, and the later half is used only for probability calibration. 2025-26 regular season and playoffs are final evaluations.",
        "",
        "## Feature Configurations",
        "- Generic: pre-shot geometry/context only.",
        "- Raw player ID: Generic plus categorical `player_id`.",
        "- Historical shooter profile: Generic plus prior FGA/FG%, 2P/3P priors, zone priors, distance priors, shrunk/Bayesian priors, season-to-date priors, previous-season profile, and cold-start flag.",
        "- Historical profile plus raw ID: profile features plus categorical player ID.",
        "",
        "## Model-Selection Results",
        markdown_table(["Model", "Feature config", "Calibration", "Log loss", "Brier", "AUC", "ECE"], selection_rows),
        "",
        "## Selected Model",
        f"{payload['selected']['model']} / {payload['selected']['feature_config']} / calibration={payload['selected']['calibration']}",
        "",
        "## Selected Model Metrics",
        markdown_table(["Dataset", "Log loss", "Brier", "AUC", "Actual FG%", "Pred FG%", "Actual PPP", "Pred xPPS"], selected_rows),
        "",
        "## Cold Start",
        f"Cold-start rows in 2025-26 test regular season: {payload['cold_start']['test_regular_rows']}; selected-model log loss on cold-start rows: {payload['cold_start']['test_regular_log_loss']}.",
        "",
        "## Representative Player Sanity Checks",
        "`data/analysis/shooter_skill_v11_player_sanity_checks.csv` contains high/mid/low predicted-probability representative players with attempts, actual FG%, predicted FG%, and prior profile summaries.",
        "",
        "## Feature Importance",
        "`data/analysis/shooter_skill_v11_feature_importance.csv` contains permutation importance. Interpret as predictive utility, not causality.",
        "",
        "## Decision-Safe Conclusion",
        "Historical shooter-profile features are decision-safe because each value is based on prior shots only. Raw player ID is included only as a comparison condition, not recommended as the primary decision-agent feature source unless maintainability and cold-start behavior are acceptable.",
    ]
    report.write_text("\n".join(text), encoding="utf-8")


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    root = root_dir()
    for path in [
        root / "artifacts" / "models",
        root / "artifacts" / "metrics",
        root / "data" / "analysis",
        root / "reports" / "modeling" / "figures" / "shooter_skill_v11",
    ]:
        path.mkdir(parents=True, exist_ok=True)

    print("Loading canonical data...")
    df = pd.read_parquet(root / "data" / "processed" / "shots_enriched.parquet")
    df = build_historical_features(df)
    splits = make_splits(df)
    split_counts = {k: int(len(v)) for k, v in splits.items()}
    eval_sets = {
        "selection_regular_2024_25_early": splits["selection_regular_2024_25_early"],
        "calibration_regular_2024_25_late": splits["calibration_regular_2024_25_late"],
        "full_validation_regular_2024_25": splits["full_validation_regular_2024_25"],
        "test_regular_2025_26": splits["test_regular_2025_26"],
        "test_clutch_2025_26": splits["test_clutch_2025_26"],
        "test_non_clutch_2025_26": splits["test_non_clutch_2025_26"],
        "playoffs_2024_25": splits["playoffs_2024_25"],
        "playoffs_2025_26": splits["playoffs_2025_26"],
        "playoffs_2025_26_clutch": splits["playoffs_2025_26_clutch"],
    }

    flat: list[dict[str, Any]] = []
    fitted: dict[str, tuple[Any, list[str], dict[str, Any]]] = {}
    for spec in specs():
        label = f"{spec.model_name}__{spec.feature_config}"
        print(f"Training {label}...")
        model = clone(spec.estimator)
        model.fit(splits["train_regular"][spec.columns], splits["train_regular"][TARGET].astype(bool))
        base_cal_p = predict(model, splits["calibration_regular_2024_25_late"], spec.columns)
        calibrators = fit_calibration(splits["calibration_regular_2024_25_late"][TARGET], base_cal_p)
        fitted[label] = (model, spec.columns, calibrators)
        joblib.dump(model, root / "artifacts" / "models" / f"shooter_skill_v11_{label}.joblib")
        for cal_name, calibrator in calibrators.items():
            for ds_name, vals in evaluate(model, spec.columns, calibrator, eval_sets).items():
                flat.append({"model": spec.model_name, "feature_config": spec.feature_config, "calibration": cal_name, "dataset": ds_name, **vals})

    candidates = [r for r in flat if r["dataset"] == "selection_regular_2024_25_early"]
    selected = sorted(candidates, key=lambda r: (r["log_loss"], r["brier"], r["ece_10_bin"]))[0]
    selected_label = f"{selected['model']}__{selected['feature_config']}"
    model, cols, calibrators = fitted[selected_label]
    calibrator = calibrators[selected["calibration"]]
    selected_metrics = [
        r for r in flat
        if r["model"] == selected["model"] and r["feature_config"] == selected["feature_config"] and r["calibration"] == selected["calibration"]
    ]
    joblib.dump(
        {"model": model, "columns": cols, "calibrator": calibrator, "selection": selected},
        root / "artifacts" / "models" / "shooter_skill_v11_selected_model.joblib",
    )
    save_figures(root, (model, cols, calibrator), eval_sets, selected["feature_config"])
    players = representative_player_checks(root, model, cols, calibrator, df)
    importance = feature_importance(root, model, cols, splits["selection_regular_2024_25_early"])

    cold = splits["test_regular_2025_26"][splits["test_regular_2025_26"]["player_cold_start"]].copy()
    if cold.empty:
        cold_result = {"test_regular_rows": 0, "test_regular_log_loss": None}
    else:
        cold_result = {
            "test_regular_rows": int(len(cold)),
            "test_regular_log_loss": metrics(cold, apply_calibration(predict(model, cold, cols), calibrator))["log_loss"],
        }

    payload = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "random_seed": RANDOM_SEED,
        "split_counts": split_counts,
        "feature_configurations": {
            "generic": GENERIC_NUMERIC + GENERIC_BOOLEAN + GENERIC_CATEGORICAL,
            "raw_player_id": GENERIC_NUMERIC + GENERIC_BOOLEAN + GENERIC_CATEGORICAL + RAW_PLAYER_ID,
            "historical_shooter_profile": GENERIC_NUMERIC + GENERIC_BOOLEAN + GENERIC_CATEGORICAL + SKILL_FEATURES,
            "historical_profile_plus_raw_id": GENERIC_NUMERIC + GENERIC_BOOLEAN + GENERIC_CATEGORICAL + SKILL_FEATURES + RAW_PLAYER_ID,
        },
        "models_tested": sorted({r["model"] for r in flat}),
        "flat_metrics": flat,
        "selected": selected,
        "selected_metrics": selected_metrics,
        "cold_start": cold_result,
        "representative_player_rows": int(len(players)),
        "feature_importance_top": importance[:25],
    }
    (root / "reports" / "modeling" / "SHOOTER_SKILL_METRICS.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (root / "artifacts" / "metrics" / "shooter_skill_v11_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(root, payload)
    print("Shooter Skill v1.1 complete")
    print(json.dumps({"selected": selected, "split_counts": split_counts, "cold_start": cold_result}, indent=2))


if __name__ == "__main__":
    main()
