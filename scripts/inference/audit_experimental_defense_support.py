#!/usr/bin/env python3
"""Forensic support audit for experimental defensive-context inference."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "defense"))
sys.path.insert(0, str(ROOT / "scripts" / "inference"))
import run_defensive_shot_quality_experiment_v1  # noqa: F401,E402 - required for joblib custom classes.
from experimental_defense_core import ExperimentalDefenseEngine  # noqa: E402


PRED_PATH = ROOT / "data" / "analysis" / "defense" / "defensive_shot_quality_predictions.parquet"
ARTIFACT = ROOT / "artifacts" / "models" / "experimental_defensive_shot_quality_v1.joblib"
REPORT = ROOT / "reports" / "modeling" / "EXPERIMENTAL_INFERENCE_SUPPORT_AUDIT.md"

DISTANCES = [0.5, 1, 2, 3, 4, 5, 6, 7, 8, 10]
JOKIC_PAYLOAD = {
    "playerId": "203999",
    "locX": 0,
    "locY": 32,
    "shotValue": 2,
    "period": 4,
    "secondsRemaining": 240,
    "isHome": True,
    "closestDefenderDistance": 3,
    "shotClock": 12,
    "dribbles": 0,
    "touchTime": 1.2,
    "defenderId": "203110",
    "season": "2025-26",
    "seasonType": "regular_season",
}
CURRY_PAYLOAD = {
    "playerId": "201939",
    "locX": 80,
    "locY": 250,
    "shotValue": 3,
    "period": 4,
    "secondsRemaining": 240,
    "isHome": True,
    "closestDefenderDistance": 3,
    "shotClock": 12,
    "dribbles": 0,
    "touchTime": 1.2,
    "defenderId": "203110",
    "season": "2025-26",
    "seasonType": "regular_season",
}
SEMANTICS = {
    "shooter_prior_fga": "2014/15 season-to-date FGA by shooter from completed previous games only; current game excluded.",
    "shooter_prior_fg_pct_shrunk": "2014/15 season-to-date overall FG% by shooter from completed previous games only, Bayesian-shrunk toward train population overall FG%.",
    "shooter_prior_2pa": "2014/15 season-to-date 2PA by shooter from completed previous games only; current game excluded.",
    "shooter_prior_2p_pct_shrunk": "2014/15 season-to-date 2P% by shooter from completed previous games only, Bayesian-shrunk toward train population 2P%.",
    "shooter_prior_3pa": "2014/15 season-to-date 3PA by shooter from completed previous games only; current game excluded.",
    "shooter_prior_3p_pct_shrunk": "2014/15 season-to-date 3P% by shooter from completed previous games only, Bayesian-shrunk toward train population 3P%.",
    "shooter_distance_prior_attempts": "2014/15 season-to-date attempts by shooter and experimental distance bucket from completed previous games only; current game excluded.",
    "shooter_distance_prior_fg_pct_shrunk": "2014/15 season-to-date FG% by shooter and experimental distance bucket from completed previous games only, Bayesian-shrunk toward train population.",
}


def fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def md_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    out = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(fmt(row.get(c), 6).replace("|", "\\|") for c in columns) + " |")
    return "\n".join(out)


def raw_and_calibrated(bundle: dict[str, Any], row: dict[str, Any]) -> tuple[float, float]:
    frame = pd.DataFrame([row])
    wrapper = bundle["model"]
    raw = float(wrapper.pipeline.predict_proba(frame)[:, 1][0])
    cal = float(wrapper.predict_proba(frame)[:, 1][0])
    return raw, cal


def transformed_trace(bundle: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    frame = pd.DataFrame([row])
    pipeline = bundle["model"].pipeline
    selected = pipeline.named_steps["select"].transform(frame)
    transformed = pipeline.named_steps["prep"].transform(selected)
    arr = transformed.toarray()[0] if hasattr(transformed, "toarray") else np.asarray(transformed)[0]
    return {
        "selectedColumns": selected.iloc[0].to_dict(),
        "hgbInputLength": int(len(arr)),
        "hgbInputHead": [float(x) if isinstance(x, (int, float, np.number)) and math.isfinite(float(x)) else None for x in arr[:40]],
    }


def training_stats(train: pd.DataFrame, columns: list[str]) -> dict[str, dict[str, float | None]]:
    stats = {}
    for col in columns:
        s = pd.to_numeric(train[col], errors="coerce").dropna().astype(float)
        if s.empty:
            stats[col] = {k: None for k in ["min", "p1", "p5", "median", "p95", "p99", "max"]}
            continue
        stats[col] = {
            "min": float(s.min()),
            "p1": float(s.quantile(0.01)),
            "p5": float(s.quantile(0.05)),
            "median": float(s.median()),
            "p95": float(s.quantile(0.95)),
            "p99": float(s.quantile(0.99)),
            "max": float(s.max()),
        }
    return stats


def support_flag(value: Any, st: dict[str, float | None], semantic_mismatch: bool = False) -> str:
    if semantic_mismatch:
        return "SEMANTIC_MISMATCH"
    if value is None or pd.isna(value):
        return "MISSING"
    v = float(value)
    if st["min"] is not None and (v < st["min"] or v > st["max"]):
        return "OUTSIDE_MIN_MAX"
    if st["p1"] is not None and (v < st["p1"] or v > st["p99"]):
        return "OUTSIDE_P1_P99"
    return "IN_SUPPORT"


def predict_curve(bundle: dict[str, Any], row: dict[str, Any], distances: list[float] = DISTANCES) -> list[dict[str, Any]]:
    rows = []
    for d in distances:
        x = dict(row)
        x["closest_defender_distance"] = float(d)
        raw, cal = raw_and_calibrated(bundle, x)
        rows.append({"closest_defender_distance": d, "raw_probability": raw, "calibrated_probability": cal, "feature_row": x})
    return rows


def nearest_neighborhood(train: pd.DataFrame, row: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    pool = train[
        train["shot_value"].eq(row["shot_value"])
        & train["shot_distance"].between(max(0, row["shot_distance"] - 1), row["shot_distance"] + 1)
        & train["shot_clock"].between(row["shot_clock"] - 4, row["shot_clock"] + 4)
        & train["touch_time"].between(max(0, row["touch_time"] - 1.5), row["touch_time"] + 1.5)
    ].copy()
    features = [
        "shooter_prior_2p_pct_shrunk",
        "shooter_distance_prior_fg_pct_shrunk",
        "shooter_prior_2pa",
        "shooter_distance_prior_attempts",
        "dribbles",
        "touch_time",
        "shot_clock",
        "closest_defender_distance",
    ]
    if pool.empty:
        return pool, {}
    for col in features:
        scale = float(train[col].std(skipna=True) or 1.0)
        pool[f"_z_{col}"] = (pd.to_numeric(pool[col], errors="coerce").fillna(train[col].median()) - float(row[col])) / scale
    zcols = [f"_z_{c}" for c in features]
    pool["_dist"] = np.sqrt(np.square(pool[zcols]).sum(axis=1))
    nearest = pool.sort_values("_dist").head(min(500, len(pool))).copy()
    summary = {
        "candidate_rows": int(len(pool)),
        "nearest_rows": int(len(nearest)),
        "actual_fg_pct": float(nearest["made"].mean()),
        "mean_shooter_prior_2p_pct_shrunk": float(nearest["shooter_prior_2p_pct_shrunk"].mean()),
        "mean_shooter_distance_prior_fg_pct_shrunk": float(nearest["shooter_distance_prior_fg_pct_shrunk"].mean()),
        "defender_distance_min": float(nearest["closest_defender_distance"].min()),
        "defender_distance_median": float(nearest["closest_defender_distance"].median()),
        "defender_distance_p95": float(nearest["closest_defender_distance"].quantile(0.95)),
        "rows_5_to_7_ft": int(nearest["closest_defender_distance"].between(5, 7).sum()),
        "fg_pct_5_to_7_ft": float(nearest.loc[nearest["closest_defender_distance"].between(5, 7), "made"].mean())
        if nearest["closest_defender_distance"].between(5, 7).any()
        else None,
    }
    return nearest, summary


def representative_rows(train: pd.DataFrame) -> dict[str, dict[str, Any]]:
    rows = {}
    scenarios = {
        "near_rim_2pt": train[train["shot_value"].eq(2) & train["shot_distance"].le(4)],
        "midrange_2pt": train[train["shot_value"].eq(2) & train["shot_distance"].between(12, 18)],
        "three_pt": train[train["shot_value"].eq(3) & train["shot_distance"].between(24, 28)],
    }
    for shot_name, base in scenarios.items():
        if base.empty:
            continue
        median_2p = train["shooter_prior_2p_pct_shrunk"].median()
        p80_2p = train["shooter_prior_2p_pct_shrunk"].quantile(0.80)
        p20_2p = train["shooter_prior_2p_pct_shrunk"].quantile(0.20)
        picks = {
            "median_2014_15_shooter": median_2p,
            "strong_2014_15_2pt_shooter": p80_2p,
            "weak_2014_15_2pt_shooter": p20_2p,
        }
        for profile_name, target in picks.items():
            sample = base.assign(_gap=(base["shooter_prior_2p_pct_shrunk"] - target).abs()).sort_values("_gap").iloc[0]
            rows[f"{profile_name}_{shot_name}"] = {c: sample[c] for c in train.attrs["spec_columns"]}
    return rows


def adapter_alternatives(train: pd.DataFrame, row: dict[str, Any], stats: dict[str, dict[str, float | None]]) -> dict[str, dict[str, Any]]:
    shooter_cols = [c for c in SEMANTICS]
    out = {"A_current": dict(row)}
    clipped = dict(row)
    for col in stats:
        if col in clipped and stats[col]["p1"] is not None:
            clipped[col] = min(max(float(clipped[col]), float(stats[col]["p1"])), float(stats[col]["p99"]))
    out["B_clip_numeric_p1_p99"] = clipped
    rates_only = dict(row)
    for col in ["shooter_prior_fga", "shooter_prior_2pa", "shooter_prior_3pa", "shooter_distance_prior_attempts"]:
        rates_only[col] = float(train[col].median())
    out["C_rates_only_counts_training_median"] = rates_only
    scaled = dict(row)
    for col in ["shooter_prior_fga", "shooter_prior_2pa", "shooter_prior_3pa", "shooter_distance_prior_attempts"]:
        denom = max(float(row[col]), 1.0)
        scaled[col] = min(float(row[col]) * float(train[col].median()) / denom, float(stats[col]["p99"]))
    out["D_counts_season_to_date_median_scale"] = scaled
    cold = dict(row)
    for col in shooter_cols:
        cold[col] = float(train[col].median())
    cold["shooter_cold_start"] = True
    out["E_training_population_shooter_profile"] = cold
    return out


def main() -> int:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    bundle = joblib.load(ARTIFACT)
    spec_cols = bundle["spec"].columns
    engine = ExperimentalDefenseEngine()
    df = pd.read_parquet(PRED_PATH)
    train = df[df["split"].eq("train")].copy()
    train.attrs["spec_columns"] = spec_cols
    numeric_cols = [c for c in spec_cols if c not in bundle["spec"].categorical]
    stats = training_stats(train, numeric_cols)

    jokic_row, jokic_shooter, jokic_defender = engine._feature_row(JOKIC_PAYLOAD)
    curry_row, curry_shooter, _ = engine._feature_row(CURRY_PAYLOAD)
    jokic_curve = predict_curve(bundle, jokic_row)
    curry_curve = predict_curve(bundle, curry_row)
    trace = transformed_trace(bundle, {**jokic_row, "closest_defender_distance": 6.0})

    support_rows = []
    semantic_mismatch_cols = {
        "shooter_prior_fga",
        "shooter_prior_2pa",
        "shooter_prior_3pa",
        "shooter_distance_prior_attempts",
    }
    for col in numeric_cols:
        st = stats[col]
        support_rows.append(
            {
                "feature": col,
                "min": st["min"],
                "p1": st["p1"],
                "p5": st["p5"],
                "median": st["median"],
                "p95": st["p95"],
                "p99": st["p99"],
                "max": st["max"],
                "Jokic": jokic_row.get(col),
                "Jokic flag": support_flag(jokic_row.get(col), st, col in semantic_mismatch_cols),
                "Curry": curry_row.get(col),
                "Curry flag": support_flag(curry_row.get(col), st, col in semantic_mismatch_cols),
            }
        )

    semantics_rows = []
    for col, description in SEMANTICS.items():
        st = stats[col]
        semantics_rows.append(
            {
                "feature": col,
                "2014/15 training semantics": description,
                "training median": st["median"],
                "training p95": st["p95"],
                "training max": st["max"],
                "Jokic modern value": jokic_row[col],
                "support": support_flag(jokic_row[col], st, col in semantic_mismatch_cols),
            }
        )

    nearest, neighborhood = nearest_neighborhood(train, jokic_row)
    bucket_rows = []
    if not nearest.empty:
        buckets = pd.cut(nearest["closest_defender_distance"], [-0.1, 2, 4, 6, 8, 100], labels=["0-2", "2-4", "4-6", "6-8", "8+"])
        grouped = nearest.groupby(buckets, observed=False)
        for name, grp in grouped:
            if grp.empty:
                continue
            bucket_rows.append({"defender_distance_bucket": str(name), "rows": int(len(grp)), "actual_fg_pct": float(grp["made"].mean())})

    reps = representative_rows(train)
    rep_curve_rows = []
    for name, row in reps.items():
        for curve in predict_curve(bundle, row):
            rep_curve_rows.append(
                {
                    "scenario": name,
                    "closest_defender_distance": curve["closest_defender_distance"],
                    "calibrated_probability": curve["calibrated_probability"],
                }
            )

    alternatives = adapter_alternatives(train, jokic_row, stats)
    alt_rows = []
    for name, row in alternatives.items():
        for curve in predict_curve(bundle, row):
            alt_rows.append(
                {
                    "adapter_strategy": name,
                    "closest_defender_distance": curve["closest_defender_distance"],
                    "calibrated_probability": curve["calibrated_probability"],
                }
            )

    defender_rows = []
    for defender_id in ["203110", "203497", "202695", ""]:
        row, _, defender = engine._feature_row({**CURRY_PAYLOAD, "defenderId": defender_id})
        raw, cal = raw_and_calibrated(bundle, row)
        defender_rows.append(
            {
                "defenderId": defender_id or "none",
                "defenderName": defender["defenderName"],
                "overallDfgPct": row["defender_pt_overall_d_fg_pct"],
                "overallDfgA": row["defender_pt_overall_d_fga"],
                "overallNormalFgPct": row["defender_pt_overall_normal_fg_pct"],
                "overallPctPlusminus": row["defender_pt_overall_pct_plusminus"],
                "raw_probability": raw,
                "calibrated_probability": cal,
            }
        )

    jokic_curve_rows = [
        {
            "closest_defender_distance": x["closest_defender_distance"],
            "raw_probability": x["raw_probability"],
            "calibrated_probability": x["calibrated_probability"],
        }
        for x in jokic_curve
    ]
    raw_cal_rows = [r for r in jokic_curve_rows if r["closest_defender_distance"] in {1, 3, 6}]
    raw_input_rows = [
        {"closest_defender_distance": x["closest_defender_distance"], **x["feature_row"]}
        for x in jokic_curve
    ]

    lines = [
        "# Experimental Inference Support Audit",
        "",
        "Forensic audit of the experimental Defensive Shot Quality V1 integration. Production inference and Decision Policy v1 were not modified.",
        "",
        "## Exact Jokic Probability Curve",
        md_table(jokic_curve_rows, ["closest_defender_distance", "raw_probability", "calibrated_probability"]),
        "",
        "## Raw Model Input Rows",
        md_table(raw_input_rows, ["closest_defender_distance", *spec_cols]),
        "",
        "## Raw vs Calibrated Trace",
        md_table(raw_cal_rows, ["closest_defender_distance", "raw_probability", "calibrated_probability"]),
        "",
        "## Modern Shooter Adapter Semantics",
        md_table(semantics_rows, ["feature", "2014/15 training semantics", "training median", "training p95", "training max", "Jokic modern value", "support"]),
        "",
        "## Numeric Distribution Support",
        md_table(support_rows, ["feature", "min", "p1", "p5", "median", "p95", "p99", "max", "Jokic", "Jokic flag", "Curry", "Curry flag"]),
        "",
        "## Transformed Feature Trace",
        "Key selected Jokic row at 6 ft:",
        "",
        "```json",
        json.dumps(trace, indent=2, default=str),
        "```",
        "",
        "## Nearest Training Neighborhood",
        "Summary:",
        "",
        "```json",
        json.dumps(neighborhood, indent=2, default=str),
        "```",
        "",
        md_table(bucket_rows, ["defender_distance_bucket", "rows", "actual_fg_pct"]),
        "",
        "## Representative Training-Supported Sensitivity Curves",
        md_table(rep_curve_rows, ["scenario", "closest_defender_distance", "calibrated_probability"]),
        "",
        "## Adapter Alternatives",
        md_table(alt_rows, ["adapter_strategy", "closest_defender_distance", "calibrated_probability"]),
        "",
        "## Defender Profile Sanity",
        md_table(defender_rows, ["defenderId", "defenderName", "overallDfgPct", "overallDfgA", "overallNormalFgPct", "overallPctPlusminus", "raw_probability", "calibrated_probability"]),
        "",
        "## Findings",
        "- The 2014/15 shooter prior count features are season-to-date and current-game-excluded; the modern adapter supplies multi-season/canonical production profile counts. That is a semantic mismatch even when values land inside min/max.",
        "- Jokic's modern near-rim row combines elite/shrunk shooting rates, non-cold-start flags, and modern count semantics with a sparse 2014/15 near-rim open-defender region.",
        "- Isotonic calibration amplifies the high raw HGB region at 6 ft rather than creating it from a low raw score.",
        "- Selected Draymond/Rudy/Kawhi defender profiles differ numerically, but the HGB places this Curry scenario in the same leaves after preprocessing; no-defender differs because missing defender PT fields trigger missing indicators/imputation.",
        "",
        "## Support Diagnostic Recommendation",
        "Add explicit experimental-only support diagnostics before showing modern experimental probabilities without qualification: `supportStatus` as `IN_DISTRIBUTION`, `LIMITED_SUPPORT`, or `OUT_OF_DISTRIBUTION`; `supportWarnings`; and `outOfRangeFeatures`. Attempt-count semantic mismatch should trigger at least `LIMITED_SUPPORT`, and outside p1-p99 plus sparse nearest-neighborhood support should trigger `OUT_OF_DISTRIBUTION`.",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"report": str(REPORT.relative_to(ROOT)), "jokic_curve": jokic_curve_rows, "defender_rows": defender_rows}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
