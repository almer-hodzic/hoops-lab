#!/usr/bin/env python3
"""Build academic submission assets from existing verified project artifacts."""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "submission"
ASSETS = OUT / "report_assets"
DATA = OUT / "report_data"
DOCS = OUT / "docs"


def read_json(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def ensure_dirs() -> None:
    for path in [OUT, ASSETS, DATA, DOCS]:
        path.mkdir(parents=True, exist_ok=True)


def pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def savefig(name: str) -> None:
    plt.tight_layout()
    plt.savefig(ASSETS / name, dpi=180, bbox_inches="tight")
    plt.close()


def calibration_bins(y: pd.Series, p: pd.Series, bins: int = 10) -> pd.DataFrame:
    frame = pd.DataFrame({"y": y.astype(int), "p": p.astype(float)})
    frame["bin"] = pd.qcut(frame["p"], q=bins, duplicates="drop")
    return frame.groupby("bin", observed=True).agg(pred=("p", "mean"), obs=("y", "mean"), n=("y", "size")).reset_index()


def roc_curve_points(y: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    order = np.argsort(-p)
    y_sorted = y[order]
    positives = y.sum()
    negatives = len(y) - positives
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    tpr = np.r_[0, tp / positives, 1]
    fpr = np.r_[0, fp / negatives, 1]
    auc = float(np.trapezoid(tpr, fpr))
    return fpr, tpr, auc


def selected_metric(metrics: dict, dataset: str) -> dict:
    for row in metrics["flat_metrics"]:
        if (
            row["model"] == "hist_gradient_boosting"
            and row["feature_config"] == "historical_profile_plus_raw_id"
            and row["calibration"] == "sigmoid"
            and row["dataset"] == dataset
        ):
            return row
    raise KeyError(dataset)


def best_rows_by_model(metrics: dict, dataset: str) -> list[dict]:
    rows = [r for r in metrics["flat_metrics"] if r["dataset"] == dataset and r["calibration"] in {"none", "sigmoid"}]
    keep = []
    for key in sorted({(r["model"], r["feature_config"]) for r in rows}):
        group = [r for r in rows if (r["model"], r["feature_config"]) == key]
        keep.append(min(group, key=lambda r: r["log_loss"]))
    return keep


def dataset_stats(shots: pd.DataFrame) -> dict:
    made = int(shots["shot_made"].sum())
    missed = int(len(shots) - made)
    regular = shots[shots["season_type"].eq("regular_season")]
    playoffs = shots[shots["season_type"].eq("playoffs")]
    return {
        "primary_dataset": "data/processed/shots_enriched.parquet",
        "rows": int(len(shots)),
        "games": int(shots["game_id"].nunique()),
        "seasons": sorted(shots["season"].dropna().astype(str).unique().tolist()),
        "season_types": sorted(shots["season_type"].dropna().astype(str).unique().tolist()),
        "regular_season_rows": int(len(regular)),
        "playoff_rows": int(len(playoffs)),
        "clutch_rows": int(shots["is_clutch"].sum()),
        "unique_shooters": int(shots["player_id"].nunique()),
        "date_min": str(pd.to_datetime(shots["game_date"]).min().date()),
        "date_max": str(pd.to_datetime(shots["game_date"]).max().date()),
        "target": {
            "made": made,
            "missed": missed,
            "made_rate": made / len(shots),
            "miss_rate": missed / len(shots),
        },
        "shot_value_distribution": {
            str(k): int(v) for k, v in shots["shot_value"].value_counts().sort_index().items()
        },
        "shot_value_success": {
            str(k): float(v) for k, v in shots.groupby("shot_value")["shot_made"].mean().sort_index().items()
        },
        "rows_by_season_type": {
            str(k): int(v) for k, v in shots["season_type"].value_counts().sort_index().items()
        },
        "rows_by_season": {
            str(k): int(v) for k, v in shots["season"].value_counts().sort_index().items()
        },
    }


def build_charts(shots: pd.DataFrame, metrics: dict, decision_metrics: dict, defense_metrics: dict) -> dict:
    decisions = pd.read_csv(ROOT / "data" / "analysis" / "decision_2025_26_regular.csv")
    decisions["made"] = decisions["points_scored"].gt(0).astype(int)
    fi = pd.read_csv(ROOT / "data" / "analysis" / "shooter_skill_v11_feature_importance.csv")
    defense_csv = pd.read_csv(ROOT / "data" / "analysis" / "defense" / "defensive_shot_quality_model_metrics.csv")

    plt.figure(figsize=(7, 4.5))
    counts = shots["shot_made"].map({True: "Made", False: "Missed"}).value_counts().reindex(["Made", "Missed"])
    bars = plt.bar(counts.index, counts.values, color=["#3b7f5f", "#9b4d4d"])
    plt.title("Target distribution: made vs missed shots")
    plt.ylabel("Shot attempts")
    for bar in bars:
        val = int(bar.get_height())
        plt.text(bar.get_x() + bar.get_width() / 2, val, f"{val:,}\n{val / len(shots):.1%}", ha="center", va="bottom", fontsize=9)
    savefig("01_target_distribution.png")

    plt.figure(figsize=(8, 4.8))
    made_dist = shots.loc[shots["shot_made"], "shot_distance"].dropna()
    miss_dist = shots.loc[~shots["shot_made"], "shot_distance"].dropna()
    bins = np.arange(0, min(45, int(shots["shot_distance"].max()) + 2), 1)
    plt.hist([made_dist, miss_dist], bins=bins, label=["Made", "Missed"], color=["#3b7f5f", "#9b4d4d"], alpha=0.75, stacked=False)
    plt.title("Shot distance distribution")
    plt.xlabel("Shot distance (feet)")
    plt.ylabel("Attempts")
    plt.legend()
    savefig("02_shot_distance_distribution.png")

    zone = shots.groupby("shot_zone_basic").agg(attempts=("shot_made", "size"), fg_pct=("shot_made", "mean")).query("attempts >= 5000")
    zone = zone.sort_values("fg_pct", ascending=True)
    value = shots.groupby("shot_value")["shot_made"].mean()
    plt.figure(figsize=(9, 5))
    labels = [f"{idx}" for idx in zone.index] + [f"{int(k)}PT" for k in value.index]
    vals = list(zone["fg_pct"]) + list(value.values)
    colors = ["#6b8fb3"] * len(zone) + ["#8a6f3d"] * len(value)
    plt.barh(labels, vals, color=colors)
    plt.title("Shot value and zone success rate")
    plt.xlabel("Field-goal percentage")
    plt.xlim(0, max(vals) * 1.15)
    savefig("03_shot_value_and_zone_success.png")

    clutch = shots.assign(context=np.where(shots["is_clutch"], "Clutch", "Non-clutch"))
    ctx = clutch.groupby("context").agg(fg_pct=("shot_made", "mean"), attempts=("shot_made", "size"), avg_distance=("shot_distance", "mean"))
    plt.figure(figsize=(7, 4.5))
    x = np.arange(len(ctx))
    w = 0.35
    plt.bar(x - w / 2, ctx["fg_pct"], width=w, label="FG%", color="#4f7c9f")
    plt.bar(x + w / 2, ctx["avg_distance"] / 40, width=w, label="Avg distance / 40", color="#9a7b45")
    plt.xticks(x, ctx.index)
    plt.title("Clutch vs non-clutch context")
    plt.ylabel("Rate / scaled distance")
    plt.legend()
    savefig("04_clutch_vs_non_clutch.png")

    cal = calibration_bins(decisions["made"], decisions["p_make"], bins=10)
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "--", color="#777", label="Ideal")
    plt.plot(cal["pred"], cal["obs"], marker="o", color="#356fa3", label="Production model")
    plt.title("Calibration curve: 2025-26 regular season")
    plt.xlabel("Mean predicted make probability")
    plt.ylabel("Observed make rate")
    plt.legend()
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    savefig("05_calibration_curve.png")

    compare = best_rows_by_model(metrics, "test_regular_2025_26")
    compare_df = pd.DataFrame(compare).sort_values("log_loss")
    compare_df.to_csv(DATA / "model_comparison_test_regular.csv", index=False)
    plt.figure(figsize=(9, 5))
    labels = [f"{r.model}\n{r.feature_config}" for r in compare_df.itertuples()]
    plt.bar(labels, compare_df["log_loss"], color="#587da8")
    plt.title("Model comparison: test log loss")
    plt.ylabel("Log loss (lower is better)")
    plt.xticks(rotation=30, ha="right")
    savefig("06_model_comparison.png")

    top_fi = fi.sort_values("importance_neg_log_loss", ascending=False).head(15).sort_values("importance_neg_log_loss")
    plt.figure(figsize=(9, 6))
    plt.barh(top_fi["feature"], top_fi["importance_neg_log_loss"], color="#5d8f68")
    plt.title("Permutation feature importance")
    plt.xlabel("Increase in negative log-loss importance")
    savefig("07_feature_importance.png")

    plt.figure(figsize=(8, 4.8))
    plt.hist(decisions.loc[decisions["made"].eq(1), "p_make"], bins=30, alpha=0.7, label="Made", color="#3b7f5f")
    plt.hist(decisions.loc[decisions["made"].eq(0), "p_make"], bins=30, alpha=0.7, label="Missed", color="#9b4d4d")
    plt.title("Predicted make-probability distribution")
    plt.xlabel("Predicted make probability")
    plt.ylabel("Attempts")
    plt.legend()
    savefig("08_probability_distribution.png")

    dec = pd.DataFrame(decision_metrics["evaluation"]["2025_26_regular"]["decisions"])
    plt.figure(figsize=(8, 4.8))
    x = np.arange(len(dec))
    plt.bar(x - 0.2, dec["share"], width=0.4, label="Share", color="#4f7c9f")
    plt.bar(x + 0.2, dec["observed_pps"] / dec["observed_pps"].max(), width=0.4, label="Observed PPS scaled", color="#9a7b45")
    plt.xticks(x, dec["decision"])
    plt.title("Decision Policy analysis: TAKE / REVIEW / DEFER")
    plt.ylabel("Share / scaled observed PPS")
    plt.legend()
    savefig("09_decision_policy_analysis.png")

    dplot = defense_csv[defense_csv["estimator"].eq("hgb")].copy()
    keep_labels = ["Base shot quality", "Defensive pressure", "Offensive creation context", "Full tracking context"]
    dplot = dplot[dplot["label"].isin(keep_labels)]
    v11 = defense_metrics["selected"]
    plt.figure(figsize=(8, 4.8))
    labels = list(dplot["label"]) + ["V1.1 selected\n(experimental)"]
    vals = list(dplot["test_log_loss"]) + [v11["test_log_loss"]]
    plt.bar(labels, vals, color=["#637fa0"] * len(dplot) + ["#8a6f3d"])
    plt.title("Defensive context experiment: test log loss")
    plt.ylabel("Log loss (lower is better)")
    plt.xticks(rotation=20, ha="right")
    savefig("10_defensive_context_experiment.png")

    y = decisions["made"].to_numpy()
    fpr_model, tpr_model, auc_model = roc_curve_points(y, decisions["p_make"].to_numpy())
    base_p = np.repeat(decisions["made"].mean(), len(decisions))
    fpr_base, tpr_base, auc_base = roc_curve_points(y, base_p)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr_model, tpr_model, label=f"Production model AUC={auc_model:.3f}", color="#356fa3")
    plt.plot(fpr_base, tpr_base, label=f"Global baseline AUC={auc_base:.3f}", color="#9b4d4d")
    plt.plot([0, 1], [0, 1], "--", color="#777")
    plt.title("ROC comparison: 2025-26 regular season")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.legend()
    savefig("11_roc_comparison.png")

    pred = decisions["p_make"].ge(0.5).astype(int)
    cm = pd.crosstab(pd.Series(y, name="Actual"), pd.Series(pred, name="Predicted")).reindex(index=[0, 1], columns=[0, 1], fill_value=0)
    plt.figure(figsize=(5.5, 5))
    plt.imshow(cm.values, cmap="Blues")
    plt.title("Confusion matrix at 0.5 threshold")
    plt.xticks([0, 1], ["Miss", "Make"])
    plt.yticks([0, 1], ["Miss", "Make"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, f"{int(cm.values[i, j]):,}", ha="center", va="center", color="black")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    savefig("12_confusion_matrix.png")

    return {
        "roc_auc_from_decision_csv": auc_model,
        "confusion_matrix_threshold_0_5": cm.astype(int).to_dict(),
    }


def write_metrics(stats: dict, metrics: dict, decision_metrics: dict, defense_metrics: dict, chart_support: dict) -> dict:
    final_test = selected_metric(metrics, "test_regular_2025_26")
    validation = selected_metric(metrics, "full_validation_regular_2024_25")
    clutch = selected_metric(metrics, "test_clutch_2025_26")
    baseline_rows = [
        r
        for r in metrics["flat_metrics"]
        if r["dataset"] == "test_regular_2025_26"
        and r["calibration"] in {"none", "sigmoid"}
        and r["model"] in {"baseline_global", "baseline_shot_value", "hist_gradient_boosting"}
    ]
    baseline_rows = best_rows_by_model({"flat_metrics": baseline_rows}, "test_regular_2025_26")
    out = {
        "dataset": stats,
        "production_model": {
            "model_name": "HistGradientBoosting",
            "feature_config": "historical_profile_plus_raw_id",
            "calibration": "sigmoid",
            "train_seasons": ["2020-21", "2021-22", "2022-23", "2023-24"],
            "validation_season": "2024-25",
            "test_season": "2025-26",
            "validation_regular": validation,
            "test_regular_2025_26": final_test,
        },
        "baseline_models": baseline_rows,
        "clutch_performance": clutch,
        "decision_policy": {
            "selected_replacement": decision_metrics["selected_replacement"],
            "replacement_xpps": decision_metrics["replacement_approaches"]["global"]["validation_summary"]["decisions"][0]["mean_replacement_xpps"],
            "threshold": decision_metrics["threshold"],
            "test_regular_2025_26": decision_metrics["evaluation"]["2025_26_regular"],
            "test_clutch_2025_26": decision_metrics["evaluation"]["2025_26_regular_clutch"],
            "note": "Observed PPS by decision is descriptive and not a causal counterfactual.",
        },
        "defensive_experiment": {
            "status": "EXPERIMENTAL",
            "selected_v11": defense_metrics["selected"],
            "v1_comparison": defense_metrics.get("v1_comparison"),
        },
        "supplementary_figures": chart_support,
        "source_of_truth": {
            "dataset": "data/processed/shots_enriched.parquet",
            "model_metrics": "artifacts/metrics/shooter_skill_v11_metrics.json",
            "decision_metrics": "reports/decision/DECISION_POLICY_METRICS.json",
            "defensive_metrics": "reports/modeling/DEFENSIVE_SHOT_QUALITY_V11_METRICS.json",
        },
    }
    (OUT / "metrics_summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    (DATA / "dataset_statistics.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return out


def write_data_dictionary() -> None:
    text = """# Data Dictionary

Primary dataset: `data/processed/shots_enriched.parquet`.

The project predicts the pre-shot probability that an NBA field-goal attempt will be made. The target is `shot_made`; all production features must be known before the shot result.

## 1. Identification / Context

| Field | Type | Description | Example | Used by production model |
| --- | --- | --- | --- | --- |
| `season` | string | NBA season label. | `2025-26` | Yes |
| `season_type` | string | Regular season or playoffs. | `regular_season` | Used for splits/evaluation |
| `dataset_split` | string | Chronological modeling split. | `test` | No, split control only |
| `game_id` | string | NBA game identifier. | `22500001` | No |
| `event_id` | string/int | Shot event identifier within game. | `18` | No |
| `game_date` | date/string | Date of the game. | `2025-10-21` | Split/order only |
| `player_id`, `player_name` | id/string | Shooter identity. | `201939`, Stephen Curry | ID used only in comparison/raw-ID model; profile features preferred |
| `team_id`, `team_name` | id/string | Shooter team. | Golden State Warriors | No |
| `opponent_team_id` | id | Opponent team identifier. | `1610612747` | No |
| `home_team_id`, `away_team_id`, `is_home` | id/bool | Home/away context. | `is_home=True` | `is_home` yes |

## 2. Shot Geometry

| Field | Type | Description | Example | Used |
| --- | --- | --- | --- | --- |
| `shot_distance` | integer | Canonical NBA shot distance in feet. | `26` | Yes |
| `loc_x`, `loc_y` | integer | Shot coordinates in NBA court units. | `80`, `250` | Yes |
| `shot_value` | integer | Point value of the attempt. | `2` or `3` | Yes |

## 3. Game Context

| Field | Type | Description | Example | Used |
| --- | --- | --- | --- | --- |
| `period` | integer | Quarter or overtime period. | `4` | Yes |
| `clock` | string | Original clock display. | `04:00` | No; converted field used |
| `seconds_remaining_period` | integer | Seconds left in the current period. | `240` | Yes |
| `is_overtime` | bool | Whether period is overtime. | `False` | Yes |
| `home_score_before`, `away_score_before` | integer | Reconstructed pre-shot score. | `98`, `100` | No direct model input |
| `shooter_score_before`, `opponent_score_before` | integer | Pre-shot score from shooter perspective. | `98`, `100` | No direct model input |
| `score_diff_before` | integer | Shooter team score minus opponent score before shot. | `-2` | Yes |
| `abs_score_diff_before` | integer | Absolute pre-shot score differential. | `2` | Yes |
| `game_state_before` | string | Leading, trailing, or tied. | `trailing` | Yes |
| `is_clutch` | bool | Q4/OT, <=5:00, abs score diff <=5. | `True` | Evaluation/analysis, not final model input |
| `clutch_time_bucket` | string | Clutch time grouping. | `0-1 min` | Analysis only |

## 4. Zone / Category Fields

| Field | Type | Description | Example | Used |
| --- | --- | --- | --- | --- |
| `shot_type` | string | 2PT or 3PT field goal. | `3PT Field Goal` | Yes |
| `action_type` | string | NBA action label. | `Jump Shot` | Yes |
| `shot_zone_basic` | string | NBA zone. | `Above the Break 3` | Yes |
| `shot_zone_area` | string | Court side/area. | `Center(C)` | Yes |
| `shot_zone_range` | string | Distance range label. | `24+ ft.` | Yes |

## 5. Target Variable

| Field | Type | Description | Used |
| --- | --- | --- | --- |
| `shot_made` | bool | Target: whether the shot was made. | Target only |
| `points_scored` | integer | Outcome points. | Not used as pre-shot feature |

## 6. Leakage-Safe Historical Shooter Features

These are engineered during modeling from prior shots only, before the current attempt:

| Feature family | Description | Used by final production model |
| --- | --- | --- |
| `player_prior_fga`, `player_prior_fg_pct`, shrunk FG% | Shooter career-to-date attempts and field-goal rate before this shot. | Yes |
| `player_prior_2pa`, `player_prior_2p_pct_shrunk` | Prior 2PT profile. | Yes |
| `player_prior_3pa`, `player_prior_3p_pct_shrunk` | Prior 3PT profile. | Yes |
| `player_zone_prior_attempts`, `player_zone_prior_fg_pct_shrunk` | Prior performance in the same shot zone. | Yes |
| `player_distance_prior_attempts`, `player_distance_prior_fg_pct_shrunk` | Prior performance in the same distance bucket. | Yes |
| `player_season_prior_*` | Season-to-date shooter profile. | Yes |
| `player_previous_season_*` | Previous-season shooter profile. | Yes |
| `player_cold_start` | Indicates limited/no prior NBA shot history. | Yes |

## 7. Derived Features

| Field/family | Description | Used |
| --- | --- | --- |
| `abs_score_diff_before` | Absolute value of pre-shot score differential. | Yes |
| `game_state_before` | Derived from score differential. | Yes |
| `is_overtime` | Derived from period. | Yes |
| `is_clutch` | Official project clutch definition. | Analysis/evaluation |
| `shot_xpps` | `p_make * shot_value`, produced by Decision Policy. | Decision layer |
| `decision_margin` | `shot_xpps - replacement_xpps`. | Decision layer |

## 8. Fields Excluded From Modeling And Why

| Field/family | Reason |
| --- | --- |
| `shot_made`, `points_scored` | Post-outcome leakage; target/result only. |
| `home_score_before`, `away_score_before` | Raw score components are summarized by pre-shot differential/context. |
| `game_id`, `event_id`, `game_date` | Identifiers/order controls, not stable basketball predictors. |
| Team names/IDs | Avoid overfitting and roster/context instability in the production model. |
| Raw player ID alone | Tested as comparison; historical profile is more interpretable and cold-start safer. |
| Defensive tracking fields | 2014/15 research extension only; not part of production model. |
"""
    (OUT / "data_dictionary.md").write_text(text, encoding="utf-8")


def write_docs(stats: dict, metrics_summary: dict) -> None:
    prod = metrics_summary["production_model"]["test_regular_2025_26"]
    clutch = metrics_summary["clutch_performance"]
    dec = metrics_summary["decision_policy"]
    dsel = metrics_summary["defensive_experiment"]["selected_v11"]
    comp = pd.read_csv(DATA / "model_comparison_test_regular.csv")
    comp_table_lines = ["| Model | Feature config | Calibration | Log Loss | Brier | ROC AUC |", "| --- | --- | --- | ---: | ---: | ---: |"]
    for row in comp.itertuples():
        comp_table_lines.append(
            f"| {row.model} | {row.feature_config} | {row.calibration} | {row.log_loss:.5f} | {row.brier:.5f} | {row.roc_auc:.5f} |"
        )
    comp_table = "\n".join(comp_table_lines)

    seminar = f"""# NBA Clutch Agent

## Predikcija kvaliteta NBA šuta i podrška odlučivanju u clutch situacijama

## 1. O projektu i zašto je drugačiji

NBA Clutch Agent je akademski ML projekat koji predviđa vjerovatnoću pogođenog NBA šuta prije nego što je šut izveden, a zatim tu vjerovatnoću pretvara u očekivane poene po šutu. Projekat nije samo preuzeti CSV i jednostavni klasifikator: uključuje obradu NBA Shot Detail i PlayByPlay podataka, konstrukciju canonical dataseta, leakage-safe historijske profile šutera, kalibrirani model vjerovatnoće, decision policy i web aplikaciju za demonstraciju.

Produkcijski dio projekta koristi moderne NBA sezone 2020-21 do 2025-26. Eksperimenti sa defanzivnim tracking kontekstom iz 2014/15 tretiraju se kao odvojena istraživačka ekstenzija i ne utiču na produkcijsku preporuku.

## 2. Izvor i nastanak dataseta

Primarni dataset je `data/processed/shots_enriched.parquet`. Nastao je iz NBA Shot Detail arhiva i PlayByPlay V3 podataka. Shot Detail daje lokaciju, tip i rezultat šuta, dok PlayByPlay omogućava provjeru i rekonstrukciju pre-shot rezultata.

Dataset pokriva sezone {', '.join(stats['seasons'])}, regularnu sezonu i playoff. Ukupno sadrži {stats['rows']:,} pokušaja šuta, {stats['games']:,} utakmica i {stats['unique_shooters']:,} jedinstvenih šutera. Regularna sezona ima {stats['regular_season_rows']:,} redova, playoff {stats['playoff_rows']:,}, a clutch subset {stats['clutch_rows']:,} redova.

Clutch je definisan kao četvrta četvrtina ili produžetak, 5:00 ili manje do kraja perioda, i apsolutna pre-shot razlika u rezultatu manja ili jednaka 5. Matching kvaliteta i audit nalaze se u `reports/data-audit/`.

## 3. Struktura podataka i ciljna varijabla

Ciljna varijabla je `shot_made`. U cijelom canonical datasetu pogođeno je {stats['target']['made']:,} šuteva, a promašeno {stats['target']['missed']:,}; ukupni FG% je {pct(stats['target']['made_rate'])}. Model koristi pre-shot geometriju, kontekst utakmice, kategorije šuta i historijski profil šutera.

2PT pokušaja ima {stats['shot_value_distribution'].get('2', 0):,}, a 3PT pokušaja {stats['shot_value_distribution'].get('3', 0):,}. Relevantne kolone i razlozi za uključivanje/isključivanje opisani su u `submission/data_dictionary.md`.

## 4. Analiza podataka (EDA)

Generisani grafici u `submission/report_assets/` prikazuju distribuciju targeta, distance šuta, uspješnost po zoni, clutch/non-clutch razlike, distribuciju predikcija i decision policy analizu. EDA potvrđuje očekivane košarkaške obrasce: bliski šutevi imaju viši procenat realizacije, 3PT šutevi imaju niži FG% ali veću vrijednost po pogotku, a clutch subset je manji i zahtijeva opreznu interpretaciju.

## 5. Priprema podataka

Shot i PBP podaci su spojeni da bi se dobio pre-shot kontekst, uključujući rezultat prije šuta. Redovi su hronološki uređeni, a split nije nasumičan: trening koristi regularne sezone 2020-21 do 2023-24, validacija 2024-25, a finalni test 2025-26. Ovakav temporalni split bolje simulira stvarnu upotrebu modela na budućim utakmicama.

Leakage prevencija je ključna: historijski shooter priors računaju se samo iz šuteva prije trenutnog pokušaja. Cold-start slučajevi koriste populacione fallback vrijednosti. Kategorijske varijable obrađuju se kroz pipeline, a numeričke varijable se imputiraju i skaliraju prema modelu.

## 6. Izbor metoda i metrika

Testirane su jednostavne baseline metode, Logistic Regression i HistGradientBoosting. Pošto je cilj vjerovatnoća, primarne metrike su Log Loss, Brier Score, ROC AUC i kalibracija/ECE. Accuracy nije dovoljna jer decision layer koristi numeričku vjerovatnoću u formuli očekivane vrijednosti.

## 7. Treniranje i iteracije

Eksperimenti su išli od globalnog baseline modela, preko shot-value baselinea i generičkog HGB modela, do modela sa sirovim `player_id` i leakage-safe historijskim profilom šutera. Finalna selekcija je HistGradientBoosting sa historical profile + raw ID konfiguracijom i sigmoid kalibracijom.

## 8. Evaluacija finalnog modela

Na 2025-26 regular season test skupu finalni model ima Log Loss {prod['log_loss']:.5f}, Brier {prod['brier']:.5f}, ROC AUC {prod['roc_auc']:.5f}, stvarni FG% {prod['actual_fg_pct']:.4f} i predviđeni FG% {prod['predicted_fg_pct']:.4f}. Na 2025-26 clutch subsetu Log Loss je {clutch['log_loss']:.5f}, Brier {clutch['brier']:.5f}, a ROC AUC {clutch['roc_auc']:.5f}.

Playoff evaluacija pokazuje drift kalibracije, što je važno ograničenje. ROC i confusion matrix su uključeni kao dopunski akademski prikazi, ali model se bira prema vjerovatnosnim metrikama.

## 9. Usporedba modela

| Model | Feature config | Calibration | Log Loss | Brier | ROC AUC |
| --- | --- | --- | ---: | ---: | ---: |
""" + "\n".join(
        f"| {r.model} | {r.feature_config} | {r.calibration} | {r.log_loss:.5f} | {r.brier:.5f} | {r.roc_auc:.5f} |"
        for r in comp.itertuples()
    ) + f"""

Finalni model je izabran jer daje najbolji balans vjerovatnosne tačnosti, kalibracije i basketball interpretabilnosti.

## 10. Decision Policy — TAKE / REVIEW / DEFER

Decision Policy koristi:

`shot_xpps = p_make * shot_value`

Zatim poredi `shot_xpps` sa replacement vrijednošću. U finalnoj politici korišten je global replacement xPPS {dec['replacement_xpps']:.6f} i prag ±{dec['threshold']:.2f}. Ako je margina iznad praga, preporuka je TAKE; ispod negativnog praga DEFER; unutar opsega REVIEW.

Observed PPS po odluci je deskriptivan rezultat iz historijskih podataka, ne dokaz kauzalnog efekta agenta.

## 11. Dodatni eksperiment: kontekst odbrane

Defanzivni eksperiment koristi 2014/15 shot-level tracking podatke sa closest defender distance, shot clock, dribbles i touch time. V1.1 izabrani model je `{dsel['candidate_key']}` sa Log Loss {dsel['test_log_loss']:.5f}, Brier {dsel['test_brier']:.5f} i ROC AUC {dsel['test_roc_auc']:.5f} na test splitu.

Zbog domain shift problema i razlike između 2014/15 tracking podataka i modernog produkcijskog dataseta, ovaj modul ostaje eksperimentalan. U aplikaciji je support-aware: OOD scenariji ne prikazuju apsolutnu eksperimentalnu vjerovatnoću kao pouzdan rezultat.

## 12. Integracija u aplikaciju

Inference je organizovan kroz Python sklearn servis, tanki .NET Web sloj i Decision Lab UI. Korisnik bira igrača, lokaciju na terenu, kontekst utakmice i dobija produkcijski p_make, xPPS, replacement xPPS i TAKE/REVIEW/DEFER. Advanced Context prikazuje eksperimentalni tracking model bez mijenjanja produkcijske odluke.

## 13. Mogućnosti poboljšanja

Najvažniji nastavci su moderni shot-level tracking podaci, tačniji defender identity u modernim sezonama, ozbiljniji possession/counterfactual model za odbijene šuteve, domain adaptation za playoff i tracking podatke, te produkcijski deployment sa bazom i autentifikacijom.

## 14. Zaključak

NBA Clutch Agent pokazuje puni akademski ML tok: izgradnja dataseta, leakage-safe feature engineering, vjerovatnosno modeliranje, kalibracija, evaluacija i decision policy. Sistem ne tvrdi kauzalnost, nego koristi prediktivnu vjerovatnoću i očekivanu vrijednost kao pomoć u analizi kvaliteta šuta.
"""

    ml_report = f"""# ML Report Content

## Dataset

Primary dataset is `data/processed/shots_enriched.parquet` with {stats['rows']:,} NBA shot attempts, {stats['games']:,} games, {stats['unique_shooters']:,} shooters, seasons {', '.join(stats['seasons'])}, regular season plus playoffs.

## Problem Definition

Binary probability estimation: predict whether a shot is made before the outcome is known. The decision layer converts probability into expected points per shot.

## Data Preparation

The project uses chronological train/validation/test splits, reconstructed pre-shot scores, canonical shot geometry, categorical shot zones/actions, and leakage-safe historical shooter priors.

## Model Experiments

Tested baselines, Logistic Regression, HistGradientBoosting, raw player ID, historical shooter profile, and calibrated variants.

## Final Model

HistGradientBoosting with historical shooter profile plus raw player ID and sigmoid calibration.

## Metrics Table

{comp_table}

Final 2025-26 regular season: Log Loss {prod['log_loss']:.5f}, Brier {prod['brier']:.5f}, ROC AUC {prod['roc_auc']:.5f}, actual FG% {prod['actual_fg_pct']:.4f}, predicted FG% {prod['predicted_fg_pct']:.4f}.

## Main Graphs

See `submission/report_assets/01_target_distribution.png` through `12_confusion_matrix.png`.

## Final Observations

The model is best evaluated as a calibrated probability model, not as a hard classifier. The TAKE/REVIEW/DEFER layer is an expected-value policy; observed PPS by decision is descriptive rather than causal.
"""

    project_doc = """# Project Documentation

## Architecture Overview

The project has four layers:

1. Data pipeline: raw NBA archives to canonical parquet datasets.
2. Modeling: shot-quality probability models and decision-policy artifacts.
3. Inference: Python service loads sklearn/joblib artifacts.
4. Web app: .NET application proxies API calls and serves the Decision Lab frontend.

## Folder Structure

- `data/raw/`: raw NBA archives.
- `data/processed/`: canonical datasets.
- `data/analysis/`: generated analysis outputs.
- `artifacts/models/`: trained model artifacts.
- `artifacts/inference/`: production inference bundle and player profiles.
- `artifacts/decision/`: Decision Policy artifacts.
- `scripts/data/`: data audit/build scripts.
- `scripts/modeling/`: model training scripts.
- `scripts/decision/`: decision-policy scripts.
- `scripts/inference/`: Python inference service.
- `nba-clutch-agent/`: .NET/Web application.
- `reports/`: audit, modeling, decision, and productization reports.

## Prerequisites

- Python 3.12 with packages from project requirements files.
- .NET SDK 9.
- Existing generated artifacts in `artifacts/` and `data/processed/`.

## Run Python Inference Service

From the project root:

```powershell
python scripts/inference/inference_service.py --host 127.0.0.1 --port 8011
```

## Run .NET Web App

From the project root:

```powershell
dotnet run --project nba-clutch-agent/AiAgents.NbaClutchAgent.Web/AiAgents.NbaClutchAgent.Web.csproj
```

## Key API Endpoints

- `GET /api/model`
- `GET /api/players?q=...`
- `GET /api/players/{id}/profile`
- `POST /api/decision/analyze`
- `GET /api/analysis/decisions`
- `POST /api/experimental/defense/analyze`
- `GET /api/experimental/defense/contract`
- `GET /api/defenders?q=...`

## Production Model Flow

User context is converted into canonical pre-shot features. The production inference bundle predicts `p_make`; Decision Policy computes `shot_xpps`, `replacement_xpps`, `decision_margin`, and TAKE/REVIEW/DEFER.

## Experimental Defensive Flow

Advanced Context calls the V1.1 defensive tracking experiment. It uses closest defender distance, shot clock, dribbles, touch time, and safe shooter rates. Defender profile display is context-only and does not alter V1.1 probability. OOD support hides absolute experimental probability in the UI.

## Important Artifact Locations

- Production inference: `artifacts/inference/production_inference_bundle.joblib`
- Production selected shot model: `artifacts/models/shooter_skill_v11_selected_model.joblib`
- Decision Policy v1: `artifacts/decision/decision_policy_v1.joblib`
- Defensive V1.1: `artifacts/models/experimental_defensive_shot_quality_v11.joblib`
- Metrics summary: `submission/metrics_summary.json`

## Limitations

The model is predictive, not causal. Playoff calibration drift exists. Defensive context is trained on 2014/15 tracking data and remains experimental. Possession-continuation/counterfactual modeling is feasible but not production-ready in the current artifact set.
"""

    ai_template = """# AI Usage Template

Do not fabricate prompts. Populate this document later from the real conversation history.

## 1. Diskusija ideje

- User prompt:
- AI response summary:
- What was accepted:
- What was changed manually / verified:
- Result:

## 2. Izgradnja canonical NBA dataseta

- User prompt:
- AI response summary:
- What was accepted:
- What was changed manually / verified:
- Result:

## 3. Shooter Skill model

- User prompt:
- AI response summary:
- What was accepted:
- What was changed manually / verified:
- Result:

## 4. Decision Policy

- User prompt:
- AI response summary:
- What was accepted:
- What was changed manually / verified:
- Result:

## 5. Productization / Python + .NET

- User prompt:
- AI response summary:
- What was accepted:
- What was changed manually / verified:
- Result:

## 6. Defensive Context research

- User prompt:
- AI response summary:
- What was accepted:
- What was changed manually / verified:
- Result:

## 7. UI / Decision Lab

- User prompt:
- AI response summary:
- What was accepted:
- What was changed manually / verified:
- Result:
"""

    (DOCS / "SEMINARSKI_CONTENT.md").write_text(seminar, encoding="utf-8")
    (DOCS / "ML_REPORT_CONTENT.md").write_text(ml_report, encoding="utf-8")
    (DOCS / "PROJECT_DOCUMENTATION.md").write_text(project_doc, encoding="utf-8")
    (DOCS / "AI_USAGE_TEMPLATE.md").write_text(ai_template, encoding="utf-8")


def write_readme(stats: dict) -> None:
    text = f"""# NBA Clutch Agent Submission

This folder contains academic-submission materials generated from the existing project artifacts.

- `data_dictionary.md`: student-friendly explanation of the final dataset and modeling variables.
- `metrics_summary.json`: consolidated verified metrics from dataset, model, decision policy, and defensive experiment artifacts.
- `report_assets/`: report-ready PNG figures.
- `report_data/`: small derived tables/statistics used by the report assets.
- `docs/SEMINARSKI_CONTENT.md`: main Bosnian seminar-paper content.
- `docs/ML_REPORT_CONTENT.md`: shorter ML-report content.
- `docs/PROJECT_DOCUMENTATION.md`: run/architecture documentation.
- `docs/AI_USAGE_TEMPLATE.md`: placeholder template for real AI-usage documentation.

Primary dataset: `data/processed/shots_enriched.parquet` ({stats['rows']:,} shot attempts). Large raw datasets are not duplicated here; the full source data, code, reports, and application remain in the main project folders.
"""
    (OUT / "README.md").write_text(text, encoding="utf-8")


def copy_reference_data() -> None:
    for src in [
        "data/analysis/shooter_skill_v11_feature_importance.csv",
        "data/analysis/defense/defensive_shot_quality_model_metrics.csv",
    ]:
        shutil.copy2(ROOT / src, DATA / Path(src).name)


def audit_metrics(stats: dict, metrics_summary: dict) -> list[str]:
    issues: list[str] = []
    shooter_metrics = read_json("artifacts/metrics/shooter_skill_v11_metrics.json")
    if stats["regular_season_rows"] != (
        shooter_metrics["split_counts"]["train_regular"]
        + shooter_metrics["split_counts"]["full_validation_regular_2024_25"]
        + shooter_metrics["split_counts"]["test_regular_2025_26"]
    ):
        issues.append("Regular-season row count differs from shooter_skill_v11 split_counts.")
    decision = read_json("reports/decision/DECISION_POLICY_METRICS.json")
    if not math.isclose(metrics_summary["decision_policy"]["threshold"], decision["threshold"]):
        issues.append("Decision threshold mismatch.")
    if not issues:
        issues.append("No metric inconsistencies found across selected source artifacts.")
    (DATA / "consistency_audit.json").write_text(json.dumps({"issues": issues}, indent=2), encoding="utf-8")
    return issues


def main() -> None:
    ensure_dirs()
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.25,
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
        }
    )

    shots = pd.read_parquet(ROOT / "data" / "processed" / "shots_enriched.parquet")
    metrics = read_json("artifacts/metrics/shooter_skill_v11_metrics.json")
    decision_metrics = read_json("reports/decision/DECISION_POLICY_METRICS.json")
    defense_metrics = read_json("reports/modeling/DEFENSIVE_SHOT_QUALITY_V11_METRICS.json")

    stats = dataset_stats(shots)
    chart_support = build_charts(shots, metrics, decision_metrics, defense_metrics)
    metrics_summary = write_metrics(stats, metrics, decision_metrics, defense_metrics, chart_support)
    write_data_dictionary()
    write_docs(stats, metrics_summary)
    write_readme(stats)
    copy_reference_data()
    issues = audit_metrics(stats, metrics_summary)
    print(json.dumps({"ok": True, "submission": str(OUT), "issues": issues}, indent=2))


if __name__ == "__main__":
    main()
