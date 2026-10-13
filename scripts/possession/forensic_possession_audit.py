#!/usr/bin/env python3
"""Forensic audit for possession parser and Decision Policy v2 target semantics."""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "data"))
from build_canonical_dataset import discover_archives, read_archive_csv  # noqa: E402


def team_from_row(row: Any, home: str, away: str) -> str:
    team = str(row.teamId)
    if team and team != "0":
        return team
    person = str(row.personId)
    if person in {home, away}:
        return person
    desc = str(row.description).upper()
    return ""


def infer_home_away(pbp: pd.DataFrame) -> tuple[dict[str, str], dict[str, str]]:
    team_rows = pbp[(pbp["teamId"] != "0") & (pbp["teamId"] != "") & (pbp["location"].isin(["h", "v"]))]
    home = team_rows[team_rows["location"].eq("h")].groupby("gameId")["teamId"].first().to_dict()
    away = team_rows[team_rows["location"].eq("v")].groupby("gameId")["teamId"].first().to_dict()
    return home, away


def independent_estimates() -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ft_types: Counter[str] = Counter()
    rebound_pairs: Counter[str] = Counter()
    for (kind, season, season_type), path in sorted(discover_archives(ROOT).items()):
        if kind != "play_by_play":
            continue
        pbp = read_archive_csv(path)
        pbp["action_number_int"] = pd.to_numeric(pbp["actionNumber"], errors="coerce").fillna(-1).astype(int)
        pbp["period_int"] = pd.to_numeric(pbp["period"], errors="coerce").fillna(0).astype(int)
        pbp = pbp.sort_values(["gameId", "period_int", "action_number_int"], kind="mergesort")
        home_map, away_map = infer_home_away(pbp)
        for game_id, game in pbp.groupby("gameId", sort=False):
            home = home_map.get(game_id, "")
            away = away_map.get(game_id, "")
            stats: dict[str, Counter[str]] = {home: Counter(), away: Counter()}
            last_shot_team = ""
            last_shot_missed = False
            for row in game.itertuples(index=False):
                action = str(row.actionType)
                team = team_from_row(row, home, away)
                if action == "Free Throw":
                    ft_types[str(row.subType)] += 1
                if action in {"Made Shot", "Missed Shot"} and str(row.isFieldGoal) == "1" and team in stats:
                    stats[team]["fga"] += 1
                    last_shot_team = team
                    last_shot_missed = action == "Missed Shot"
                elif action == "Free Throw" and team in stats:
                    stats[team]["fta"] += 1
                elif action == "Turnover" and team in stats:
                    stats[team]["tov"] += 1
                elif action == "Rebound":
                    rebound_team = team
                    if last_shot_missed and rebound_team in stats and last_shot_team in stats:
                        if rebound_team == last_shot_team:
                            stats[rebound_team]["oreb"] += 1
                            rebound_pairs["miss_oreb_same_team"] += 1
                        else:
                            rebound_pairs["miss_dreb_other_team"] += 1
                    elif last_shot_missed and not rebound_team:
                        rebound_pairs["miss_rebound_unknown_team"] += 1
                    last_shot_missed = False
                    last_shot_team = ""
            for team in [home, away]:
                if not team:
                    continue
                c = stats[team]
                estimate = c["fga"] - c["oreb"] + c["tov"] + 0.44 * c["fta"]
                rows.append(
                    {
                        "season": season,
                        "season_type": season_type,
                        "game_id": game_id,
                        "team_id": team,
                        "fga": c["fga"],
                        "oreb": c["oreb"],
                        "tov": c["tov"],
                        "fta": c["fta"],
                        "estimated_possessions": estimate,
                    }
                )
    return pd.DataFrame(rows), {"free_throw_subtypes": dict(ft_types), "rebound_pairs": dict(rebound_pairs)}


def summarize_possession_counts(possessions: pd.DataFrame, estimates: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    inferred = possessions.groupby(["season", "season_type", "game_id", "offense_team_id"]).size().reset_index(name="inferred_possessions")
    inferred = inferred.rename(columns={"offense_team_id": "team_id"})
    merged = estimates.merge(inferred, on=["season", "season_type", "game_id", "team_id"], how="left")
    merged["inferred_possessions"] = merged["inferred_possessions"].fillna(0)
    merged["diff"] = merged["inferred_possessions"] - merged["estimated_possessions"]
    summary = {
        "inferred_mean_team_game": float(merged["inferred_possessions"].mean()),
        "inferred_median_team_game": float(merged["inferred_possessions"].median()),
        "inferred_p05_team_game": float(merged["inferred_possessions"].quantile(0.05)),
        "inferred_p95_team_game": float(merged["inferred_possessions"].quantile(0.95)),
        "estimated_mean_team_game": float(merged["estimated_possessions"].mean()),
        "estimated_median_team_game": float(merged["estimated_possessions"].median()),
        "estimated_p05_team_game": float(merged["estimated_possessions"].quantile(0.05)),
        "estimated_p95_team_game": float(merged["estimated_possessions"].quantile(0.95)),
        "mean_absolute_difference": float(merged["diff"].abs().mean()),
    }
    largest = merged.reindex(merged["diff"].abs().sort_values(ascending=False).index).head(15)
    return summary, largest


def anomaly_summary(possessions: pd.DataFrame) -> dict[str, Any]:
    neg = possessions[possessions["possession_duration_seconds"].fillna(0).lt(0)].copy()
    order = possessions[possessions["first_action_number"].gt(possessions["last_action_number"])].copy()
    high = possessions[possessions["points_scored"].gt(8)].copy()
    neg_reasons = {
        "period_transition_close_with_new_period_row": int((neg["first_action_number"] > neg["last_action_number"]).sum()),
        "clock_increases_with_action_order_preserved": int(((neg["first_action_number"] <= neg["last_action_number"]) & (neg["start_clock_seconds"] < neg["end_clock_seconds"])).sum()),
    }
    high_reasons = {
        "multiple_field_goal_attempts": int(high["field_goal_attempts"].gt(3).sum()),
        "many_free_throw_attempts": int(high["free_throw_attempts"].gt(4).sum()),
        "ended_at_period_boundary": int(high["ended_in_period_end"].sum()),
        "had_offensive_rebound": int(high["had_offensive_rebound"].sum()),
    }
    return {
        "negative_duration_rows": int(len(neg)),
        "ordering_error_rows": int(len(order)),
        "gt8_point_rows": int(len(high)),
        "negative_duration_categories": neg_reasons,
        "ordering_error_categories": {"period_transition_or_action_reset": int(len(order))},
        "gt8_point_categories": high_reasons,
        "gt8_samples": high.sort_values("points_scored", ascending=False).head(12)[
            [
                "season",
                "season_type",
                "game_id",
                "possession_id",
                "period",
                "offense_team_id",
                "points_scored",
                "field_goal_attempts",
                "free_throw_attempts",
                "turnovers",
                "possession_duration_seconds",
                "ended_in_period_end",
            ]
        ].to_dict(orient="records"),
    }


def continuation_signal() -> dict[str, Any]:
    metrics = json.loads((ROOT / "reports" / "decision" / "DECISION_POLICY_V2_METRICS.json").read_text(encoding="utf-8"))
    cont = metrics["continuation_models"]
    global_mae = cont["global"]["validation_mae"]
    global_rmse = cont["global"]["validation_rmse"]
    hgb_mae = cont["hgb"]["validation_mae"]
    hgb_rmse = cont["hgb"]["validation_rmse"]
    ridge_mae = cont["ridge"]["validation_mae"]
    ridge_rmse = cont["ridge"]["validation_rmse"]
    return {
        "global_mae": global_mae,
        "hgb_mae": hgb_mae,
        "hgb_absolute_mae_improvement": global_mae - hgb_mae,
        "hgb_relative_mae_improvement": (global_mae - hgb_mae) / global_mae,
        "global_rmse": global_rmse,
        "hgb_rmse": hgb_rmse,
        "hgb_absolute_rmse_improvement": global_rmse - hgb_rmse,
        "ridge_mae": ridge_mae,
        "ridge_rmse": ridge_rmse,
        "ridge_minus_hgb_mae": ridge_mae - hgb_mae,
        "hgb_bias": cont["hgb"]["validation_bias"],
        "ridge_bias": cont["ridge"]["validation_bias"],
    }


def defer_diagnosis() -> dict[str, Any]:
    regular = pd.read_parquet(ROOT / "data" / "analysis" / "decision_policy_v2_2025_26_regular.parquet")
    regular["margin_bin"] = pd.cut(regular["decision_margin_v2"], [-10, -0.5, -0.2, 0, 0.2, 0.5, 10])
    by_decision = regular.groupby("decision").agg(
        rows=("decision", "size"),
        mean_shot_xpps=("shot_xpps", "mean"),
        mean_continuation_xpps=("continuation_xpps", "mean"),
        mean_margin=("decision_margin_v2", "mean"),
        observed_pps=("points_scored", "mean"),
    ).reset_index()
    by_value = regular.groupby(["decision", "shot_value"]).size().reset_index(name="rows")
    by_zone = regular.groupby(["decision", "shot_zone_basic"]).agg(rows=("decision", "size"), mean_margin=("decision_margin_v2", "mean")).reset_index()
    by_time = regular.groupby(["decision", "time_context_bucket"]).agg(rows=("decision", "size"), mean_margin=("decision_margin_v2", "mean")).reset_index()
    margin_dist = regular["decision_margin_v2"].describe(percentiles=[0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]).to_dict()
    return {
        "by_decision": by_decision.to_dict(orient="records"),
        "by_shot_value": by_value.to_dict(orient="records"),
        "top_defer_zones": by_zone[by_zone["decision"].eq("DEFER")].sort_values("rows", ascending=False).head(10).to_dict(orient="records"),
        "by_time": by_time.to_dict(orient="records"),
        "margin_distribution": {k: float(v) for k, v in margin_dist.items()},
    }


def write_report(payload: dict[str, Any]) -> None:
    count = payload["count_sanity"]
    signal = payload["continuation_signal"]
    anomalies = payload["anomalies"]
    largest_rows = [
        [
            r["season"],
            r["season_type"],
            r["game_id"],
            r["team_id"],
            f"{r['inferred_possessions']:.1f}",
            f"{r['estimated_possessions']:.1f}",
            f"{r['diff']:.1f}",
        ]
        for r in payload["largest_disagreements"]
    ]
    def table(headers: list[str], rows: list[list[Any]]) -> str:
        lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
        lines += ["| " + " | ".join(map(str, row)) + " |" for row in rows]
        return "\n".join(lines)

    report = [
        "# Possession Forensic Audit",
        "",
        "## Target Semantics",
        "Decision Policy v2 trains the continuation model on `possession_points` from `shot_possession_links.parquet`, which is copied from the full parsed possession's `points_scored` value. That target includes the current evaluated shot when the shot scores, includes free throws kept in that possession, and includes points after offensive rebounds if the parser keeps the possession open.",
        "",
        "Classification: `CURRENT_POSSESSION_OUTCOME_MODEL`. It is not a true defer/continue counterfactual and should not be described as expected value of deferring the shot.",
        "",
        "Future information used as target: final possession score, events after the shot, offensive-rebound continuations, and any free throws assigned to the possession. Future leakage into v2 features is limited but not zero-risk: `shot_attempt_number_in_possession` is produced by grouping final shot links and can encode whether this is a later attempt in a possession; the model does not use final possession points, final FGA count, or post-shot elapsed time as features.",
        "",
        "## Possession Count Sanity",
        f"Mean inferred possessions per team/game: `{count['inferred_mean_team_game']:.2f}`; median `{count['inferred_median_team_game']:.2f}`; p05 `{count['inferred_p05_team_game']:.2f}`; p95 `{count['inferred_p95_team_game']:.2f}`.",
        f"Independent estimate mean per team/game: `{count['estimated_mean_team_game']:.2f}`; median `{count['estimated_median_team_game']:.2f}`; p05 `{count['estimated_p05_team_game']:.2f}`; p95 `{count['estimated_p95_team_game']:.2f}`.",
        f"Mean absolute difference: `{count['mean_absolute_difference']:.2f}` possessions per team/game.",
        "",
        table(["Season", "Type", "Game", "Team", "Inferred", "Estimate", "Diff"], largest_rows),
        "",
        "## Anomalies",
        f"Negative durations: `{anomalies['negative_duration_rows']}`. Primary causes: period transition closure using the first row of the next period, and clock increases within preserved action order from corrections/replay/same-clock administrative events.",
        f"Ordering errors: `{anomalies['ordering_error_rows']}`. Primary cause: action-number reset/period transition closure with the next-period row.",
        f">8 point possessions: `{anomalies['gt8_point_rows']}`. Categories: `{json.dumps(anomalies['gt8_point_categories'])}`.",
        "",
        "## Free Throws and Rebounds",
        "The current parser does not explicitly model final free throw possession endings, and it closes made field goals before checking for same-clock and-one free throws. Team rebounds represented with `teamId == 0` and team ID in `personId` require special handling. These are parser-design issues, not just noisy rows.",
        "",
        "Offensive-rebound validation from raw PBP: " + json.dumps(payload["event_audit"]["rebound_pairs"]),
        "",
        "## Counterfactual Limitation",
        "NBA PBP records events that happened. It does not record passes, pump fakes, declined open shots, or comparable unchosen shot opportunities. Therefore true empirical defer labels are not observable from this PBP alone.",
        "",
        "## Continuation Model Value",
        f"HGB MAE improvement over global: `{signal['hgb_absolute_mae_improvement']:.4f}` absolute, `{signal['hgb_relative_mae_improvement'] * 100:.2f}%` relative.",
        f"HGB RMSE improvement over global: `{signal['hgb_absolute_rmse_improvement']:.4f}`.",
        f"Ridge MAE `{signal['ridge_mae']:.4f}` vs HGB MAE `{signal['hgb_mae']:.4f}`; ridge is simpler and essentially tied.",
        "",
        "## V1 vs V2 Methodology",
        "V1 is a transparent shot xPPS versus replacement benchmark and does not claim to observe a defer counterfactual. V2 adds contextual possession information, but the current target is full possession outcome including the current shot, so V2 is not scientifically defensible as a defer policy.",
        "",
        "## Decision",
        "`USE V2 AS CONTEXTUAL BENCHMARK ONLY` until a corrected possession parser and honestly named contextual target are used. Do not adopt v2 as a production defer/continue policy.",
    ]
    out = ROOT / "reports" / "possession" / "POSSESSION_FORENSIC_AUDIT.md"
    out.write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    possessions = pd.read_parquet(ROOT / "data" / "processed" / "possessions.parquet")
    estimates, event_audit = independent_estimates()
    count_sanity, largest = summarize_possession_counts(possessions, estimates)
    payload = {
        "count_sanity": count_sanity,
        "largest_disagreements": largest.to_dict(orient="records"),
        "anomalies": anomaly_summary(possessions),
        "event_audit": event_audit,
        "continuation_signal": continuation_signal(),
        "defer_diagnosis": defer_diagnosis(),
    }
    (ROOT / "data" / "metadata" / "possession_forensic_audit.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(payload)
    print(json.dumps(payload, indent=2)[:12000])


if __name__ == "__main__":
    main()
