#!/usr/bin/env python3
"""Decision Policy v1 for NBA Clutch Agent.

Uses the existing Shooter Skill v1.1 historical_shooter_profile model as the
shot-value engine. Does not retrain that model or modify app/canonical files.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "modeling"))
from train_shooter_skill_v11 import (  # noqa: E402
    GENERIC_BOOLEAN,
    GENERIC_CATEGORICAL,
    GENERIC_NUMERIC,
    SKILL_FEATURES,
    TARGET,
    apply_calibration,
    build_historical_features,
    fit_calibration,
    predict,
)


MODEL_PATH = ROOT / "artifacts" / "models" / "shooter_skill_v11_hist_gradient_boosting__historical_shooter_profile.joblib"
SHOT_DATA = ROOT / "data" / "processed" / "shots_enriched.parquet"
OUTPUT_REPORT = ROOT / "reports" / "decision" / "DECISION_POLICY_V1.md"
OUTPUT_METRICS = ROOT / "reports" / "decision" / "DECISION_POLICY_METRICS.json"
RANDOM_SEED = 20261008


@dataclass
class ReplacementResult:
    name: str
    validation_mae: float
    validation_rmse: float
    threshold: float
    objective: float
    validation_summary: dict[str, Any]


def add_context_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["period_context"] = np.select([df["period"].eq(4), df["period"].gt(4)], ["Q4", "OT"], default="Q1-Q3")
    df["score_bucket"] = pd.cut(
        df["score_diff_before"].astype(float),
        [-100, -10, -4, 3, 9, 100],
        labels=["down_10_plus", "down_4_to_9", "close", "up_4_to_9", "up_10_plus"],
    ).astype(str)
    df["time_context_bucket"] = pd.cut(
        df["seconds_remaining_period"].astype(float),
        [-1, 60, 180, 360, 720],
        labels=["0-60", "61-180", "181-360", "361-720"],
    ).astype(str)
    return df


def load_and_score() -> tuple[pd.DataFrame, list[str], Any, Any]:
    print("Loading canonical data and building historical profile features...")
    df = pd.read_parquet(SHOT_DATA)
    df = build_historical_features(df)
    df = add_context_columns(df)
    model = joblib.load(MODEL_PATH)
    columns = GENERIC_NUMERIC + GENERIC_BOOLEAN + GENERIC_CATEGORICAL + SKILL_FEATURES
    calibration = df[(df["season"] == "2024-25") & (df["season_type"] == "regular_season")].sort_values(["game_date", "game_id", "event_id"])
    calibration_late = calibration.iloc[len(calibration) // 2 :].copy()
    print("Fitting sigmoid probability calibration on 2024-25 late split...")
    cal = fit_calibration(calibration_late[TARGET], predict(model, calibration_late, columns))["sigmoid"]
    df["p_make"] = apply_calibration(predict(model, df, columns), cal)
    df["shot_xpps"] = df["p_make"] * df["shot_value"]
    return df, columns, model, cal


def train_validation_splits(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = df[(df["season"].isin(["2020-21", "2021-22", "2022-23", "2023-24"])) & (df["season_type"] == "regular_season")].copy()
    validation = df[(df["season"] == "2024-25") & (df["season_type"] == "regular_season")].copy()
    return train, validation


def group_replacement(train: pd.DataFrame, target: pd.DataFrame, cols: list[str], min_n: int, global_value: float) -> pd.Series:
    grouped = train.groupby(cols).agg(mean_pps=("points_scored", "mean"), n=("points_scored", "size")).reset_index()
    grouped.loc[grouped["n"] < min_n, "mean_pps"] = np.nan
    merged = target[cols].merge(grouped[cols + ["mean_pps"]], on=cols, how="left")
    return merged["mean_pps"].fillna(global_value).astype(float)


def fit_context_model(train: pd.DataFrame) -> Pipeline:
    cols_cat = ["period_context", "game_state_before", "score_bucket", "time_context_bucket"]
    cols_num = ["period", "seconds_remaining_period", "score_diff_before", "abs_score_diff_before", "is_home"]
    model = Pipeline(
        [
            (
                "prep",
                ColumnTransformer(
                    [
                        ("num", Pipeline([("impute", SimpleImputer(strategy="median"))]), cols_num),
                        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cols_cat),
                    ]
                ),
            ),
            (
                "model",
                HistGradientBoostingRegressor(
                    max_iter=100,
                    learning_rate=0.06,
                    max_leaf_nodes=31,
                    l2_regularization=0.05,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )
    model.fit(train[cols_num + cols_cat], train["points_scored"].astype(float))
    return model


def replacement_predictions(name: str, train: pd.DataFrame, target: pd.DataFrame, global_value: float, context_model: Pipeline | None = None) -> pd.Series:
    if name == "global":
        return pd.Series(global_value, index=target.index, dtype=float)
    if name == "game_context":
        return group_replacement(train, target, ["period_context", "game_state_before", "score_bucket", "time_context_bucket"], 150, global_value).set_axis(target.index)
    if name == "team_context":
        return group_replacement(train, target, ["team_id", "period_context", "game_state_before", "score_bucket", "time_context_bucket"], 40, global_value).set_axis(target.index)
    if name == "context_model":
        assert context_model is not None
        cols = ["period", "seconds_remaining_period", "score_diff_before", "abs_score_diff_before", "is_home", "period_context", "game_state_before", "score_bucket", "time_context_bucket"]
        return pd.Series(np.clip(context_model.predict(target[cols]), 0.7, 1.4), index=target.index, dtype=float)
    raise ValueError(name)


def assign_decisions(frame: pd.DataFrame, threshold: float) -> pd.Series:
    margin = frame["decision_margin"]
    return pd.Series(np.select([margin >= threshold, margin <= -threshold], ["TAKE", "DEFER"], default="REVIEW"), index=frame.index)


def decision_summary(frame: pd.DataFrame) -> dict[str, Any]:
    rows = []
    for decision, group in frame.groupby("decision"):
        rows.append(
            {
                "decision": decision,
                "rows": int(len(group)),
                "share": float(len(group) / len(frame)) if len(frame) else 0,
                "mean_shot_xpps": float(group["shot_xpps"].mean()),
                "mean_replacement_xpps": float(group["replacement_xpps"].mean()),
                "observed_pps": float(group["points_scored"].mean()),
                "mean_margin": float(group["decision_margin"].mean()),
            }
        )
    order = {"TAKE": 0, "REVIEW": 1, "DEFER": 2}
    rows = sorted(rows, key=lambda r: order.get(r["decision"], 99))
    return {"rows": int(len(frame)), "decisions": rows}


def choose_threshold(validation: pd.DataFrame) -> tuple[float, float, dict[str, Any]]:
    best: tuple[float, float, dict[str, Any]] | None = None
    for threshold in np.round(np.arange(0.00, 0.151, 0.005), 3):
        tmp = validation.copy()
        tmp["decision"] = assign_decisions(tmp, threshold)
        counts = tmp["decision"].value_counts(normalize=True)
        if counts.get("TAKE", 0) < 0.10 or counts.get("DEFER", 0) < 0.10:
            continue
        take_pps = tmp.loc[tmp["decision"].eq("TAKE"), "points_scored"].mean()
        defer_pps = tmp.loc[tmp["decision"].eq("DEFER"), "points_scored"].mean()
        review_share = counts.get("REVIEW", 0)
        # Prefer separation, but avoid pushing everything into REVIEW.
        objective = float((take_pps - defer_pps) - 0.03 * review_share)
        summary = decision_summary(tmp)
        if best is None or objective > best[1]:
            best = (float(threshold), objective, summary)
    if best is None:
        tmp = validation.copy()
        tmp["decision"] = assign_decisions(tmp, 0.0)
        return 0.0, 0.0, decision_summary(tmp)
    return best


def evaluate_replacement_approaches(df: pd.DataFrame) -> tuple[str, float, dict[str, Any], dict[str, Any], Pipeline | None]:
    train, validation = train_validation_splits(df)
    global_value = float(train["points_scored"].mean())
    context_model = fit_context_model(train)
    approaches = ["global", "game_context", "team_context", "context_model"]
    results: dict[str, Any] = {}
    best_name = ""
    best_result: ReplacementResult | None = None
    for name in approaches:
        val = validation.copy()
        val["replacement_xpps"] = replacement_predictions(name, train, val, global_value, context_model)
        val["decision_margin"] = val["shot_xpps"] - val["replacement_xpps"]
        threshold, objective, summary = choose_threshold(val)
        mae = float(mean_absolute_error(val["points_scored"], val["replacement_xpps"]))
        rmse = float(mean_squared_error(val["points_scored"], val["replacement_xpps"]) ** 0.5)
        result = ReplacementResult(name, mae, rmse, threshold, objective, summary)
        results[name] = {
            "validation_mae": mae,
            "validation_rmse": rmse,
            "threshold": threshold,
            "objective": objective,
            "validation_summary": summary,
        }
        if best_result is None or result.objective > best_result.objective:
            best_result = result
            best_name = name
    assert best_result is not None
    return best_name, best_result.threshold, results, {"global_value": global_value}, context_model


def apply_policy(df: pd.DataFrame, replacement_name: str, threshold: float, train: pd.DataFrame, global_value: float, context_model: Pipeline | None) -> pd.DataFrame:
    out = df.copy()
    out["replacement_xpps"] = replacement_predictions(replacement_name, train, out, global_value, context_model)
    out["decision_margin"] = out["shot_xpps"] - out["replacement_xpps"]
    out["decision"] = assign_decisions(out, threshold)
    return out


def player_summaries(frame: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    grouped = frame.groupby(["player_id", "player_name"]).agg(
        shots=("decision", "size"),
        take=("decision", lambda s: int((s == "TAKE").sum())),
        review=("decision", lambda s: int((s == "REVIEW").sum())),
        defer=("decision", lambda s: int((s == "DEFER").sum())),
        avg_p_make=("p_make", "mean"),
        avg_shot_xpps=("shot_xpps", "mean"),
        avg_replacement_xpps=("replacement_xpps", "mean"),
        observed_pps=("points_scored", "mean"),
    ).reset_index()
    grouped["take_rate"] = grouped["take"] / grouped["shots"]
    grouped = grouped[grouped["shots"] >= 100].sort_values(["take_rate", "shots"], ascending=[False, False])
    grouped.to_csv(out_path, index=False)
    return grouped


def representative_examples(frame: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    samples = []
    for decision in ["TAKE", "REVIEW", "DEFER"]:
        part = frame[frame["decision"].eq(decision)].copy()
        if part.empty:
            continue
        if decision == "TAKE":
            part = part.sort_values("decision_margin", ascending=False).head(12)
        elif decision == "DEFER":
            part = part.sort_values("decision_margin", ascending=True).head(12)
        else:
            part = part.assign(abs_margin=part["decision_margin"].abs()).sort_values("abs_margin").head(12)
        samples.append(part)
    cols = [
        "decision",
        "season",
        "season_type",
        "game_id",
        "event_id",
        "player_name",
        "team_name",
        "shot_type",
        "action_type",
        "shot_distance",
        "shot_value",
        "p_make",
        "shot_xpps",
        "replacement_xpps",
        "decision_margin",
        "points_scored",
        "is_clutch",
    ]
    out = pd.concat(samples, ignore_index=True)[cols]
    out.to_csv(out_path, index=False)
    return out


def legacy_comparison() -> dict[str, Any]:
    return {
        "legacy_agent": "Existing .NET ClutchDecisionAgent + BanditPolicy/QTable",
        "legacy_behavior": "Online epsilon-greedy Q-table over binned clutch observations; cold start randomizes Shoot/DontShoot until minimum visits; DontShoot reward uses fixed baseline PPP.",
        "decision_policy_v1": "Offline calibrated shot xPPS minus replacement xPPS margin; all rows can be scored deterministically; REVIEW is a true uncertainty band rather than an executed action.",
        "main_difference": "Decision Policy v1 estimates expected value directly from pre-shot probability and contextual replacement, while the legacy bandit learns action values from historical reward observations and exploration.",
    }


def possession_continuation_assessment() -> dict[str, Any]:
    return {
        "can_current_pbp_support_future_model": True,
        "assessment": "PBP has ordered events, scores, teams, players, shot/free-throw/rebound/turnover/foul/substitution/timeout events, so a future possession-continuation model is feasible.",
        "limitations": [
            "Possession boundaries require explicit derivation from rebounds, turnovers, made shots, free throws, period boundaries, and team changes.",
            "The current canonical shot table is event-level; it does not yet include possession IDs or next-shot continuation outcomes.",
            "Some PBP score correction behavior is documented and should remain quality-flagged in any possession model.",
        ],
    }


def write_report(payload: dict[str, Any]) -> None:
    rows = []
    for name, result in payload["replacement_approaches"].items():
        rows.append([name, f"{result['validation_mae']:.4f}", f"{result['validation_rmse']:.4f}", result["threshold"], f"{result['objective']:.4f}"])
    eval_rows = []
    for name, summary in payload["evaluation"].items():
        for row in summary["decisions"]:
            eval_rows.append([name, row["decision"], row["rows"], f"{row['share']:.3f}", f"{row['mean_shot_xpps']:.3f}", f"{row['mean_replacement_xpps']:.3f}", f"{row['observed_pps']:.3f}", f"{row['mean_margin']:.3f}"])
    report = [
        "# Decision Policy v1",
        "",
        "## Method",
        "Uses the existing Shooter Skill v1.1 `historical_shooter_profile` artifact as the shot-value engine. The shooter model is not retrained. A sigmoid probability calibrator is fit on the 2024-25 late validation split because only the base historical-profile artifact exists on disk.",
        "",
        "Decision quantities: `p_make`, `shot_xpps = p_make * shot_value`, `replacement_xpps`, and `decision_margin = shot_xpps - replacement_xpps`.",
        "",
        "## Replacement Approaches",
        markdown_table(["Approach", "Validation MAE", "Validation RMSE", "Threshold", "Objective"], rows),
        "",
        f"Selected replacement approach: `{payload['selected_replacement']}`",
        f"Selected REVIEW threshold: `{payload['threshold']}`",
        "",
        "## Frozen Evaluation",
        markdown_table(["Dataset", "Decision", "Rows", "Share", "Mean shot xPPS", "Mean repl xPPS", "Observed PPS", "Mean margin"], eval_rows),
        "",
        "## Legacy Q/Bandit Comparison",
        payload["legacy_comparison"]["legacy_behavior"],
        "",
        payload["legacy_comparison"]["main_difference"],
        "",
        "## Possession Continuation Assessment",
        payload["possession_continuation"]["assessment"],
        "",
        "Limitations: " + "; ".join(payload["possession_continuation"]["limitations"]),
    ]
    OUTPUT_REPORT.write_text("\n".join(report), encoding="utf-8")


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, value in enumerate(row):
            widths[i] = max(widths[i], len(str(value)))
    lines = [
        "| " + " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))) + " |",
        "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[i]).ljust(widths[i]) for i in range(len(headers))) + " |")
    return "\n".join(lines)


def main() -> None:
    for path in [ROOT / "scripts" / "decision", ROOT / "artifacts" / "decision", ROOT / "reports" / "decision", ROOT / "data" / "analysis"]:
        path.mkdir(parents=True, exist_ok=True)
    df, columns, model, calibrator = load_and_score()
    train, validation = train_validation_splits(df)
    selected_replacement, threshold, replacement_results, replacement_meta, context_model = evaluate_replacement_approaches(df)
    global_value = replacement_meta["global_value"]
    eval_sets = {
        "2025_26_regular": df[(df["season"] == "2025-26") & (df["season_type"] == "regular_season")],
        "2025_26_regular_clutch": df[(df["season"] == "2025-26") & (df["season_type"] == "regular_season") & df["is_clutch"]],
        "2025_26_playoffs": df[(df["season"] == "2025-26") & (df["season_type"] == "playoffs")],
        "2025_26_playoff_clutch": df[(df["season"] == "2025-26") & (df["season_type"] == "playoffs") & df["is_clutch"]],
    }
    evaluation: dict[str, Any] = {}
    scored_sets: dict[str, pd.DataFrame] = {}
    for name, frame in eval_sets.items():
        scored = apply_policy(frame, selected_replacement, threshold, train, global_value, context_model)
        scored_sets[name] = scored
        evaluation[name] = decision_summary(scored)
        scored.to_parquet(ROOT / "artifacts" / "decision" / f"decision_policy_v1_{name}.parquet", index=False)
        scored[[
            "season",
            "season_type",
            "game_id",
            "event_id",
            "player_name",
            "team_name",
            "is_clutch",
            "p_make",
            "shot_xpps",
            "replacement_xpps",
            "decision_margin",
            "decision",
            "points_scored",
        ]].to_csv(ROOT / "data" / "analysis" / f"decision_{name}.csv", index=False)
    player_summary = player_summaries(scored_sets["2025_26_regular"], ROOT / "data" / "analysis" / "decision_player_summaries_2025_26_regular.csv")
    examples = representative_examples(scored_sets["2025_26_regular"], ROOT / "data" / "analysis" / "decision_representative_examples_2025_26_regular.csv")
    payload = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "shot_value_engine": "shooter_skill_v11_hist_gradient_boosting__historical_shooter_profile.joblib",
        "shot_value_engine_retrained": False,
        "probability_calibration": "sigmoid fit on 2024-25 late validation split from existing model probabilities",
        "selected_replacement": selected_replacement,
        "threshold": threshold,
        "replacement_approaches": replacement_results,
        "evaluation": evaluation,
        "player_summary_rows": int(len(player_summary)),
        "representative_example_rows": int(len(examples)),
        "legacy_comparison": legacy_comparison(),
        "possession_continuation": possession_continuation_assessment(),
    }
    OUTPUT_METRICS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    joblib.dump(
        {
            "selected_replacement": selected_replacement,
            "threshold": threshold,
            "global_value": global_value,
            "context_model": context_model,
            "shot_model_path": str(MODEL_PATH.relative_to(ROOT)),
            "columns": columns,
        },
        ROOT / "artifacts" / "decision" / "decision_policy_v1.joblib",
    )
    write_report(payload)
    print(json.dumps({"selected_replacement": selected_replacement, "threshold": threshold, "evaluation": evaluation}, indent=2))


if __name__ == "__main__":
    main()
