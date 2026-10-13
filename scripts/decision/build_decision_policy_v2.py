#!/usr/bin/env python3
"""Decision Policy v2 with possession continuation value.

This script leaves the completed shooter-skill model and Decision Policy v1
artifacts untouched. It reuses the existing historical shooter-profile scorer,
learns a possession-continuation value layer from pre-shot context, then freezes
the selected v2 policy on 2025-26 evaluation sets.
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
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "decision"))
from build_decision_policy_v1 import add_context_columns, load_and_score  # noqa: E402


LINKS_PATH = ROOT / "data" / "analysis" / "shot_possession_links.parquet"
POSSESSIONS_PATH = ROOT / "data" / "processed" / "possessions.parquet"
OUTPUT_METRICS = ROOT / "reports" / "decision" / "DECISION_POLICY_V2_METRICS.json"
OUTPUT_REPORT = ROOT / "reports" / "decision" / "DECISION_POLICY_V2.md"
RANDOM_SEED = 20261011
TRAIN_SEASONS = ["2020-21", "2021-22", "2022-23", "2023-24"]
VALIDATION_SEASON = "2024-25"
TEST_SEASON = "2025-26"


@dataclass
class ContinuationCandidate:
    name: str
    validation_mae: float
    validation_rmse: float
    validation_bias: float
    calibration: list[dict[str, Any]]


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


def attach_possession_features(shots: pd.DataFrame) -> pd.DataFrame:
    links = pd.read_parquet(LINKS_PATH)
    keys = ["season", "season_type", "game_id", "event_id"]
    keep = keys + [
        "possession_id",
        "offense_team_id",
        "possession_points",
        "shot_attempt_number_in_possession",
        "possession_continued_after_shot",
        "elapsed_after_shot_until_possession_end",
    ]
    shots = shots.copy()
    links = links.copy()
    shots["_event_occurrence"] = shots.groupby(keys).cumcount()
    links["_event_occurrence"] = links.groupby(keys).cumcount()
    merge_keys = keys + ["_event_occurrence"]
    out = shots.merge(links[keep + ["_event_occurrence"]], on=merge_keys, how="left", validate="one_to_one")
    out = out.drop(columns=["_event_occurrence"])
    out["linked_to_possession"] = out["possession_id"].notna()
    out = out[out["linked_to_possession"]].copy()
    out["offense_team_id"] = out["offense_team_id"].fillna(out["team_id"]).astype(str)
    out["shot_attempt_number_in_possession"] = out["shot_attempt_number_in_possession"].fillna(1).astype(int)
    out["is_second_chance_shot_state"] = out["shot_attempt_number_in_possession"].gt(1)
    out["possession_points"] = out["possession_points"].astype(float)
    return out


def add_prior_team_ppp(shots: pd.DataFrame) -> pd.DataFrame:
    possessions = pd.read_parquet(POSSESSIONS_PATH)
    possessions = possessions.sort_values(
        ["season", "season_type", "game_id", "first_action_number"],
        key=lambda s: s.map({"regular_season": 0, "playoffs": 1}) if s.name == "season_type" else s,
    ).copy()
    possessions["prior_team_possessions"] = possessions.groupby("offense_team_id").cumcount()
    possessions["prior_team_points"] = possessions.groupby("offense_team_id")["points_scored"].cumsum() - possessions["points_scored"]
    league_prior = float(possessions["points_scored"].mean())
    possessions["team_prior_ppp"] = np.where(
        possessions["prior_team_possessions"].gt(0),
        possessions["prior_team_points"] / possessions["prior_team_possessions"],
        league_prior,
    )
    prior = possessions[["season", "season_type", "game_id", "possession_id", "prior_team_possessions", "team_prior_ppp"]]
    out = shots.merge(prior, on=["season", "season_type", "game_id", "possession_id"], how="left", validate="many_to_one")
    out["prior_team_possessions"] = out["prior_team_possessions"].fillna(0).astype(float)
    out["team_prior_ppp"] = out["team_prior_ppp"].fillna(league_prior).astype(float)
    return out


def add_v2_context_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = add_context_columns(df)
    out["seconds_bucket"] = pd.cut(
        out["seconds_remaining_period"].astype(float),
        [-1, 24, 60, 180, 360, 720],
        labels=["0-24", "25-60", "61-180", "181-360", "361-720"],
    ).astype(str)
    out["shot_clock_proxy"] = np.select(
        [out["seconds_remaining_period"].le(4), out["seconds_remaining_period"].le(10), out["seconds_remaining_period"].le(24)],
        ["very_late", "late", "normal"],
        default="unknown",
    )
    out["season_index"] = out["season"].map({s: i for i, s in enumerate(["2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26"])})
    return out


def prepare_data() -> pd.DataFrame:
    shots, _, _, _ = load_and_score()
    shots = attach_possession_features(shots)
    shots = add_prior_team_ppp(shots)
    shots = add_v2_context_columns(shots)
    return shots


def train_validation_splits(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = df[(df["season"].isin(TRAIN_SEASONS)) & (df["season_type"].eq("regular_season"))].copy()
    validation = df[(df["season"].eq(VALIDATION_SEASON)) & (df["season_type"].eq("regular_season"))].copy()
    return train, validation


NUMERIC_FEATURES = [
    "period",
    "seconds_remaining_period",
    "score_diff_before",
    "abs_score_diff_before",
    "is_home",
    "shot_attempt_number_in_possession",
    "is_second_chance_shot_state",
    "prior_team_possessions",
    "team_prior_ppp",
    "season_index",
]
CATEGORICAL_FEATURES = [
    "period_context",
    "game_state_before",
    "score_bucket",
    "time_context_bucket",
    "seconds_bucket",
    "shot_clock_proxy",
    "team_id",
]


def calibration_bins(actual: pd.Series, pred: pd.Series, bins: int = 10) -> list[dict[str, Any]]:
    frame = pd.DataFrame({"actual": actual.astype(float), "pred": pred.astype(float)})
    frame["bin"] = pd.qcut(frame["pred"].rank(method="first"), bins, labels=False, duplicates="drop")
    rows = []
    for _, group in frame.groupby("bin"):
        rows.append(
            {
                "rows": int(len(group)),
                "predicted_ppp": float(group["pred"].mean()),
                "actual_ppp": float(group["actual"].mean()),
                "bias": float((group["pred"] - group["actual"]).mean()),
            }
        )
    return rows


def make_models(train: pd.DataFrame) -> dict[str, Any]:
    global_value = float(train["possession_points"].mean())
    group_cols = ["period_context", "game_state_before", "score_bucket", "time_context_bucket"]
    grouped = train.groupby(group_cols).agg(mean_ppp=("possession_points", "mean"), rows=("possession_points", "size")).reset_index()
    grouped.loc[grouped["rows"].lt(200), "mean_ppp"] = np.nan

    ridge = Pipeline(
        [
            (
                "prep",
                ColumnTransformer(
                    [
                        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), NUMERIC_FEATURES),
                        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=100), CATEGORICAL_FEATURES),
                    ]
                ),
            ),
            ("model", Ridge(alpha=10.0)),
        ]
    )
    ridge.fit(train[NUMERIC_FEATURES + CATEGORICAL_FEATURES], train["possession_points"].astype(float))

    hgb = Pipeline(
        [
            (
                "prep",
                ColumnTransformer(
                    [
                        ("num", Pipeline([("impute", SimpleImputer(strategy="median"))]), NUMERIC_FEATURES),
                        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CATEGORICAL_FEATURES),
                    ]
                ),
            ),
            (
                "model",
                HistGradientBoostingRegressor(
                    max_iter=160,
                    learning_rate=0.05,
                    max_leaf_nodes=31,
                    l2_regularization=0.08,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )
    hgb.fit(train[NUMERIC_FEATURES + CATEGORICAL_FEATURES], train["possession_points"].astype(float))
    return {"global": global_value, "context_table": grouped, "ridge": ridge, "hgb": hgb}


def predict_continuation(name: str, models: dict[str, Any], target: pd.DataFrame) -> pd.Series:
    if name == "global":
        return pd.Series(models["global"], index=target.index, dtype=float)
    if name == "context_table":
        cols = ["period_context", "game_state_before", "score_bucket", "time_context_bucket"]
        merged = target[cols].merge(models["context_table"][cols + ["mean_ppp"]], on=cols, how="left")
        return merged["mean_ppp"].fillna(models["global"]).clip(0.65, 1.45).set_axis(target.index).astype(float)
    if name == "ridge":
        pred = models["ridge"].predict(target[NUMERIC_FEATURES + CATEGORICAL_FEATURES])
        return pd.Series(np.clip(pred, 0.55, 1.65), index=target.index, dtype=float)
    if name == "hgb":
        pred = models["hgb"].predict(target[NUMERIC_FEATURES + CATEGORICAL_FEATURES])
        return pd.Series(np.clip(pred, 0.55, 1.65), index=target.index, dtype=float)
    raise ValueError(name)


def evaluate_continuation_models(train: pd.DataFrame, validation: pd.DataFrame) -> tuple[str, dict[str, Any], dict[str, Any]]:
    models = make_models(train)
    results: dict[str, Any] = {}
    best_name = ""
    best_rmse = np.inf
    for name in ["global", "context_table", "ridge", "hgb"]:
        pred = predict_continuation(name, models, validation)
        actual = validation["possession_points"].astype(float)
        candidate = ContinuationCandidate(
            name=name,
            validation_mae=float(mean_absolute_error(actual, pred)),
            validation_rmse=float(mean_squared_error(actual, pred) ** 0.5),
            validation_bias=float((pred - actual).mean()),
            calibration=calibration_bins(actual, pred),
        )
        results[name] = {
            "validation_mae": candidate.validation_mae,
            "validation_rmse": candidate.validation_rmse,
            "validation_bias": candidate.validation_bias,
            "calibration": candidate.calibration,
        }
        if candidate.validation_rmse < best_rmse:
            best_rmse = candidate.validation_rmse
            best_name = name
    return best_name, models, results


def assign_decisions(margin: pd.Series, threshold: float) -> pd.Series:
    return pd.Series(np.select([margin >= threshold, margin <= -threshold], ["TAKE", "DEFER"], default="REVIEW"), index=margin.index)


def decision_summary(frame: pd.DataFrame) -> dict[str, Any]:
    rows = []
    for decision, group in frame.groupby("decision"):
        rows.append(
            {
                "decision": decision,
                "rows": int(len(group)),
                "share": float(len(group) / len(frame)) if len(frame) else 0.0,
                "mean_p_make": float(group["p_make"].mean()),
                "mean_shot_xpps": float(group["shot_xpps"].mean()),
                "mean_continuation_xpps": float(group["continuation_xpps"].mean()),
                "mean_decision_margin_v2": float(group["decision_margin_v2"].mean()),
                "observed_shot_pps": float(group["points_scored"].mean()),
                "observed_possession_ppp": float(group["possession_points"].mean()),
            }
        )
    order = {"TAKE": 0, "REVIEW": 1, "DEFER": 2}
    return {"rows": int(len(frame)), "decisions": sorted(rows, key=lambda r: order.get(r["decision"], 99))}


def choose_threshold(validation: pd.DataFrame) -> tuple[float, dict[str, Any]]:
    best: tuple[float, float, dict[str, Any]] | None = None
    for threshold in np.round(np.arange(0.00, 0.201, 0.005), 3):
        tmp = validation.copy()
        tmp["decision"] = assign_decisions(tmp["decision_margin_v2"], threshold)
        shares = tmp["decision"].value_counts(normalize=True)
        if shares.get("TAKE", 0.0) < 0.08 or shares.get("DEFER", 0.0) < 0.08:
            continue
        take_pps = tmp.loc[tmp["decision"].eq("TAKE"), "points_scored"].mean()
        defer_pps = tmp.loc[tmp["decision"].eq("DEFER"), "points_scored"].mean()
        take_poss = tmp.loc[tmp["decision"].eq("TAKE"), "possession_points"].mean()
        defer_poss = tmp.loc[tmp["decision"].eq("DEFER"), "possession_points"].mean()
        review_share = shares.get("REVIEW", 0.0)
        objective = float((take_pps - defer_pps) + 0.35 * (take_poss - defer_poss) - 0.025 * review_share)
        summary = decision_summary(tmp)
        if best is None or objective > best[1]:
            best = (float(threshold), objective, summary)
    if best is None:
        tmp = validation.copy()
        tmp["decision"] = assign_decisions(tmp["decision_margin_v2"], 0.0)
        return 0.0, {"objective": 0.0, "validation_summary": decision_summary(tmp)}
    return best[0], {"objective": best[1], "validation_summary": best[2]}


def apply_policy(frame: pd.DataFrame, model_name: str, models: dict[str, Any], threshold: float) -> pd.DataFrame:
    out = frame.copy()
    out["continuation_xpps"] = predict_continuation(model_name, models, out)
    out["decision_margin_v2"] = out["shot_xpps"] - out["continuation_xpps"]
    out["decision"] = assign_decisions(out["decision_margin_v2"], threshold)
    return out


def write_eval_outputs(scored: pd.DataFrame, name: str) -> None:
    parquet_path = ROOT / "data" / "analysis" / f"decision_policy_v2_{name}.parquet"
    scored.to_parquet(parquet_path, index=False)
    if name == "2025_26_regular":
        scored.to_parquet(ROOT / "artifacts" / "decision" / "decision_policy_v2_2025_26_regular.parquet", index=False)


def player_decision_summary(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = frame.groupby(["player_id", "player_name"]).agg(
        shots=("decision", "size"),
        take=("decision", lambda s: int((s == "TAKE").sum())),
        review=("decision", lambda s: int((s == "REVIEW").sum())),
        defer=("decision", lambda s: int((s == "DEFER").sum())),
        avg_p_make=("p_make", "mean"),
        avg_shot_xpps=("shot_xpps", "mean"),
        avg_continuation_xpps=("continuation_xpps", "mean"),
        observed_shot_pps=("points_scored", "mean"),
        observed_possession_ppp=("possession_points", "mean"),
    ).reset_index()
    grouped["take_rate"] = grouped["take"] / grouped["shots"]
    grouped["defer_rate"] = grouped["defer"] / grouped["shots"]
    grouped = grouped[grouped["shots"] >= 100].sort_values(["take_rate", "shots"], ascending=[False, False])
    grouped.to_csv(ROOT / "data" / "analysis" / "decision_policy_v2_player_summaries_2025_26_regular.csv", index=False)
    return grouped


def representative_examples(frame: pd.DataFrame) -> pd.DataFrame:
    samples: list[pd.DataFrame] = []
    rules = [
        ("strong_take", frame[frame["decision"].eq("TAKE")].sort_values("decision_margin_v2", ascending=False).head(10)),
        ("strong_defer", frame[frame["decision"].eq("DEFER")].sort_values("decision_margin_v2", ascending=True).head(10)),
        ("review_boundary", frame[frame["decision"].eq("REVIEW")].assign(abs_margin=lambda d: d["decision_margin_v2"].abs()).sort_values("abs_margin").head(10)),
        ("clutch_take", frame[frame["decision"].eq("TAKE") & frame["is_clutch"]].sort_values("decision_margin_v2", ascending=False).head(5)),
        ("clutch_defer", frame[frame["decision"].eq("DEFER") & frame["is_clutch"]].sort_values("decision_margin_v2", ascending=True).head(5)),
        ("second_chance", frame[frame["is_second_chance_shot_state"]].sort_values("decision_margin_v2", ascending=False).head(5)),
    ]
    for label, part in rules:
        if part.empty:
            continue
        part = part.copy()
        part["example_type"] = label
        samples.append(part)
    cols = [
        "example_type",
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
        "continuation_xpps",
        "decision_margin_v2",
        "points_scored",
        "possession_points",
        "is_clutch",
        "shot_attempt_number_in_possession",
    ]
    out = pd.concat(samples, ignore_index=True)[cols] if samples else pd.DataFrame(columns=cols)
    out.to_csv(ROOT / "data" / "analysis" / "decision_policy_v2_examples.csv", index=False)
    return out


def compare_with_v1(v2_sets: dict[str, pd.DataFrame]) -> dict[str, Any]:
    rows = []
    for name, v2 in v2_sets.items():
        v1_path = ROOT / "data" / "analysis" / f"decision_{name}.csv"
        if not v1_path.exists():
            continue
        v1 = pd.read_csv(v1_path, dtype={"game_id": str, "event_id": str})
        keys = ["season", "season_type", "game_id", "event_id"]
        merged = v2[keys + ["decision", "decision_margin_v2"]].merge(
            v1[keys + ["decision", "decision_margin"]].rename(columns={"decision": "decision_v1", "decision_margin": "decision_margin_v1"}),
            on=keys,
            how="inner",
        )
        merged = merged.rename(columns={"decision": "decision_v2"})
        for (d1, d2), group in merged.groupby(["decision_v1", "decision_v2"]):
            rows.append({"dataset": name, "decision_v1": d1, "decision_v2": d2, "rows": int(len(group))})
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "data" / "analysis" / "decision_policy_v1_vs_v2.csv", index=False)
    comparison: dict[str, Any] = {}
    if not out.empty:
        for dataset, group in out.groupby("dataset"):
            total = int(group["rows"].sum())
            agreement = int(group.loc[group["decision_v1"].eq(group["decision_v2"]), "rows"].sum())
            comparison[dataset] = {
                "rows": total,
                "agreement_rate": float(agreement / total) if total else 0.0,
                "transitions": group.sort_values("rows", ascending=False).to_dict(orient="records"),
            }
    return comparison


def write_report(payload: dict[str, Any]) -> None:
    cont_rows = [
        [name, f"{m['validation_mae']:.4f}", f"{m['validation_rmse']:.4f}", f"{m['validation_bias']:.4f}"]
        for name, m in payload["continuation_models"].items()
    ]
    eval_rows = []
    for dataset, summary in payload["evaluation"].items():
        for row in summary["decisions"]:
            eval_rows.append(
                [
                    dataset,
                    row["decision"],
                    row["rows"],
                    f"{row['share']:.3f}",
                    f"{row['mean_shot_xpps']:.3f}",
                    f"{row['mean_continuation_xpps']:.3f}",
                    f"{row['observed_shot_pps']:.3f}",
                    f"{row['observed_possession_ppp']:.3f}",
                ]
            )
    comp_rows = []
    for dataset, summary in payload["v1_vs_v2"].items():
        comp_rows.append([dataset, summary["rows"], f"{summary['agreement_rate']:.3f}"])
    lines = [
        "# Decision Policy v2",
        "",
        "## Method",
        "Decision Policy v2 keeps the completed Shooter Skill v1.1 historical shooter-profile model frozen and replaces the v1 generic replacement value with a learned possession-continuation value from pre-shot context.",
        "",
        "`decision_margin_v2 = shot_xpps - continuation_xpps`; TAKE and DEFER are assigned outside the validation-selected REVIEW band.",
        "",
        "## Continuation Models",
        markdown_table(["Model", "Validation MAE", "Validation RMSE", "Validation Bias"], cont_rows),
        "",
        f"Selected continuation model: `{payload['selected_continuation_model']}`",
        f"Selected v2 threshold: `{payload['threshold']}`",
        "",
        "## Frozen Evaluation",
        markdown_table(["Dataset", "Decision", "Rows", "Share", "Shot xPPS", "Cont xPPS", "Shot PPS", "Poss PPP"], eval_rows),
        "",
        "## V1 vs V2",
        markdown_table(["Dataset", "Rows", "Agreement"], comp_rows) if comp_rows else "No v1 comparison rows were available.",
        "",
        "## Validation Notes",
        "The possession builder linked every canonical shot to a possession. Remaining possession-quality flags are carried forward in `reports/possession/POSSESSION_VALIDATION.md` and should be treated as known noise for continuation modeling.",
    ]
    OUTPUT_REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    for path in [ROOT / "scripts" / "decision", ROOT / "artifacts" / "decision", ROOT / "reports" / "decision", ROOT / "data" / "analysis"]:
        path.mkdir(parents=True, exist_ok=True)
    df = prepare_data()
    train, validation = train_validation_splits(df)
    selected_model, continuation_models, continuation_results = evaluate_continuation_models(train, validation)
    validation_scored = apply_policy(validation, selected_model, continuation_models, 0.0)
    threshold, threshold_meta = choose_threshold(validation_scored)
    eval_sets = {
        "2025_26_regular": df[(df["season"].eq(TEST_SEASON)) & (df["season_type"].eq("regular_season"))],
        "2025_26_regular_clutch": df[(df["season"].eq(TEST_SEASON)) & (df["season_type"].eq("regular_season")) & df["is_clutch"]],
        "2025_26_playoffs": df[(df["season"].eq(TEST_SEASON)) & (df["season_type"].eq("playoffs"))],
        "2025_26_playoff_clutch": df[(df["season"].eq(TEST_SEASON)) & (df["season_type"].eq("playoffs")) & df["is_clutch"]],
    }
    evaluation: dict[str, Any] = {}
    scored_sets: dict[str, pd.DataFrame] = {}
    for name, frame in eval_sets.items():
        scored = apply_policy(frame, selected_model, continuation_models, threshold)
        scored_sets[name] = scored
        evaluation[name] = decision_summary(scored)
        write_eval_outputs(scored, name)
    player_summary = player_decision_summary(scored_sets["2025_26_regular"])
    examples = representative_examples(scored_sets["2025_26_regular"])
    comparison = compare_with_v1(scored_sets)
    payload = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "shot_value_engine": "existing Shooter Skill v1.1 historical_shooter_profile artifact via Decision Policy v1 scorer",
        "shot_value_engine_retrained": False,
        "continuation_target": "possession_points linked to each shot from the possession parser",
        "continuation_features": {"numeric": NUMERIC_FEATURES, "categorical": CATEGORICAL_FEATURES},
        "train_seasons": TRAIN_SEASONS,
        "validation_season": VALIDATION_SEASON,
        "test_season": TEST_SEASON,
        "selected_continuation_model": selected_model,
        "continuation_models": continuation_results,
        "threshold": threshold,
        "threshold_selection": threshold_meta,
        "evaluation": evaluation,
        "v1_vs_v2": comparison,
        "player_summary_rows": int(len(player_summary)),
        "representative_example_rows": int(len(examples)),
    }
    OUTPUT_METRICS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    joblib.dump(
        {
            "selected_continuation_model": selected_model,
            "continuation_models": continuation_models,
            "threshold": threshold,
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "shot_value_engine_retrained": False,
        },
        ROOT / "artifacts" / "decision" / "decision_policy_v2.joblib",
    )
    write_report(payload)
    print(json.dumps({"selected_continuation_model": selected_model, "threshold": threshold, "evaluation": evaluation}, indent=2))


if __name__ == "__main__":
    main()
