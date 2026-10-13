#!/usr/bin/env python3
"""Design/validation spike for HoopsLab single-game simulation inputs."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LinearRegression, RidgeCV
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.base import clone
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
except ImportError as exc:  # pragma: no cover - validation script should fail loudly.
    raise SystemExit(
        "scikit-learn is required for the simulation spike. Install project modeling dependencies first."
    ) from exc


ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "historical" / "processed"
OUT_DIR = ROOT / "data" / "simulation"
REPORT_DIR = ROOT / "reports" / "simulation"

FAMOUS_TEAMS = [
    ("1985-86", "Boston Celtics"),
    ("1995-96", "Chicago Bulls"),
    ("2000-01", "Los Angeles Lakers"),
    ("2012-13", "Miami Heat"),
    ("2016-17", "Golden State Warriors"),
]

CUSTOM_TEAM = [
    ("PG", "Stephen Curry", "2015-16"),
    ("SG", "Michael Jordan", "1990-91"),
    ("SF", "LeBron James", "2012-13"),
    ("PF", "Tim Duncan", "2002-03"),
    ("C", "Shaquille O'Neal", "1999-00"),
]

FEATURES = [
    "league_ortg_estimate",
    "league_pace",
    "league_ts_pct",
    "w_points_per_75",
    "w_assists_per_75",
    "w_rebounds_per_75",
    "w_turnovers_per_75",
    "w_steals_per_75",
    "w_blocks_per_75",
    "w_offensive_rebounds_per_75",
    "w_defensive_rebounds_per_75",
    "w_relative_ts",
    "w_true_shooting_pct",
    "w_usage_pct",
    "w_assist_pct",
    "w_turnover_pct",
    "w_total_rebound_pct",
    "w_three_point_attempt_rate",
    "w_free_throw_rate",
    "w_age",
    "w_per",
    "team_3par_from_players",
    "team_ftr_from_players",
    "team_ast_to_tov_from_players",
    "minutes_top1_share",
    "minutes_top3_share",
    "minutes_top5_share",
    "minutes_hhi",
    "rotation_players_500",
    "rotation_players_1000",
    "weighted_usage_top3",
    "usage_pressure",
    "spacing_proxy",
    "rebounding_proxy",
    "defense_box_proxy",
]


def safe_div(num: float | int | None, den: float | int | None) -> float | None:
    if num is None or den in (None, 0):
        return None
    if pd.isna(num) or pd.isna(den):
        return None
    return float(num) / float(den)


def weighted_mean(df: pd.DataFrame, col: str, weights: pd.Series) -> float:
    values = pd.to_numeric(df[col], errors="coerce")
    mask = values.notna() & weights.notna()
    if not mask.any():
        return np.nan
    denom = weights[mask].sum()
    if denom == 0:
        return np.nan
    return float((values[mask] * weights[mask]).sum() / denom)


def add_derived_player_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["offensive_rebounds_per_75"] = out.apply(
        lambda r: safe_div(r.get("offensive_rebounds"), r.get("estimated_possessions_minutes_team_pace")) * 75
        if safe_div(r.get("offensive_rebounds"), r.get("estimated_possessions_minutes_team_pace")) is not None
        else np.nan,
        axis=1,
    )
    out["defensive_rebounds_per_75"] = out.apply(
        lambda r: safe_div(r.get("defensive_rebounds"), r.get("estimated_possessions_minutes_team_pace")) * 75
        if safe_div(r.get("defensive_rebounds"), r.get("estimated_possessions_minutes_team_pace")) is not None
        else np.nan,
        axis=1,
    )
    out["win_shares_per_48"] = out.apply(
        lambda r: safe_div(r.get("win_shares"), r.get("minutes")) * 48
        if safe_div(r.get("win_shares"), r.get("minutes")) is not None
        else np.nan,
        axis=1,
    )
    return out


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    players = pd.read_csv(PROCESSED / "player_seasons.csv")
    splits = pd.read_csv(PROCESSED / "player_team_splits.csv")
    teams = pd.read_csv(PROCESSED / "team_seasons.csv")
    leagues = pd.read_csv(PROCESSED / "league_seasons.csv")
    return players, splits, teams, leagues


def roster_rows(players: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    base = players[(players["team"] != "TOT") & (~players["team"].astype(str).str.endswith("TM"))].copy()
    split = splits.copy()
    cols = sorted(set(base.columns) & set(split.columns))
    roster = pd.concat([base[cols], split[cols]], ignore_index=True)
    roster = roster[pd.to_numeric(roster["minutes"], errors="coerce").fillna(0) > 0].copy()
    return add_derived_player_features(roster)


def build_team_rows(roster: pd.DataFrame, teams: pd.DataFrame, leagues: pd.DataFrame) -> pd.DataFrame:
    league = leagues.copy()
    league["league_ortg_estimate"] = league["league_points_per_75"] / 75 * 100
    league_cols = ["season", "season_end_year", "league_ts_pct", "league_pace", "league_points_per_75", "league_ortg_estimate"]
    team_context = teams.merge(league[league_cols], on=["season", "season_end_year"], how="left")

    rows: list[dict] = []
    for _, team in team_context.iterrows():
        group = roster[(roster["season"] == team["season"]) & (roster["team_id"] == team["team_id"])].copy()
        if group.empty:
            continue
        minutes = pd.to_numeric(group["minutes"], errors="coerce").fillna(0)
        total_minutes = minutes.sum()
        if total_minutes <= 0:
            continue
        weights = minutes / total_minutes
        sorted_weights = weights.sort_values(ascending=False).reset_index(drop=True)
        total_fga = pd.to_numeric(group["field_goal_attempts"], errors="coerce").sum()
        total_3pa = pd.to_numeric(group["three_point_attempts"], errors="coerce").sum()
        total_fta = pd.to_numeric(group["free_throw_attempts"], errors="coerce").sum()
        total_ast = pd.to_numeric(group["assists"], errors="coerce").sum()
        total_tov = pd.to_numeric(group["turnovers"], errors="coerce").sum()
        usage_sorted = group.assign(_w=weights.values).sort_values("minutes", ascending=False).head(3)

        row = {
            "season": team["season"],
            "season_end_year": int(team["season_end_year"]),
            "team_id": team["team_id"],
            "team_name": team["team_name"],
            "wins": team["wins"],
            "losses": team["losses"],
            "actual_ortg": team["off_rtg"],
            "actual_drtg": team["def_rtg"],
            "actual_pace": team["pace"],
            "actual_net_rtg": team["net_rtg"],
            "league_ortg_estimate": team["league_ortg_estimate"],
            "league_pace": team["league_pace"],
            "league_ts_pct": team["league_ts_pct"],
            "league_points_per_75": team["league_points_per_75"],
            "roster_players": int(len(group)),
            "total_player_minutes": float(total_minutes),
            "team_3par_from_players": safe_div(total_3pa, total_fga),
            "team_ftr_from_players": safe_div(total_fta, total_fga),
            "team_ast_to_tov_from_players": safe_div(total_ast, total_tov),
            "minutes_top1_share": float(sorted_weights.iloc[:1].sum()),
            "minutes_top3_share": float(sorted_weights.iloc[:3].sum()),
            "minutes_top5_share": float(sorted_weights.iloc[:5].sum()),
            "minutes_hhi": float((weights * weights).sum()),
            "rotation_players_500": int((minutes >= 500).sum()),
            "rotation_players_1000": int((minutes >= 1000).sum()),
            "weighted_usage_top3": float((pd.to_numeric(usage_sorted["usage_pct"], errors="coerce") * usage_sorted["_w"]).sum()),
        }
        weighted_cols = {
            "points_per_75": "w_points_per_75",
            "assists_per_75": "w_assists_per_75",
            "rebounds_per_75": "w_rebounds_per_75",
            "turnovers_per_75": "w_turnovers_per_75",
            "steals_per_75": "w_steals_per_75",
            "blocks_per_75": "w_blocks_per_75",
            "offensive_rebounds_per_75": "w_offensive_rebounds_per_75",
            "defensive_rebounds_per_75": "w_defensive_rebounds_per_75",
            "relative_ts": "w_relative_ts",
            "true_shooting_pct": "w_true_shooting_pct",
            "usage_pct": "w_usage_pct",
            "assist_pct": "w_assist_pct",
            "turnover_pct": "w_turnover_pct",
            "total_rebound_pct": "w_total_rebound_pct",
            "three_point_attempt_rate": "w_three_point_attempt_rate",
            "free_throw_rate": "w_free_throw_rate",
            "age": "w_age",
            "player_efficiency_rating": "w_per",
        }
        for source, dest in weighted_cols.items():
            row[dest] = weighted_mean(group, source, weights)
        row["usage_pressure"] = max(0.0, row["weighted_usage_top3"] - 18.0) if not pd.isna(row["weighted_usage_top3"]) else np.nan
        row["spacing_proxy"] = (row["team_3par_from_players"] or 0) * (row["w_relative_ts"] if not pd.isna(row["w_relative_ts"]) else 0)
        row["rebounding_proxy"] = row["w_rebounds_per_75"] + 0.5 * row["w_offensive_rebounds_per_75"]
        row["defense_box_proxy"] = (
            0.8 * row["w_steals_per_75"]
            + 0.8 * row["w_blocks_per_75"]
            + 0.35 * row["w_defensive_rebounds_per_75"]
            + (row["w_per"] if not pd.isna(row["w_per"]) else 0) * 0.03
        )
        rows.append(row)
    return pd.DataFrame(rows)


def temporal_split(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "train": df[df["season_end_year"] <= 2014].copy(),
        "validation": df[(df["season_end_year"] >= 2015) & (df["season_end_year"] <= 2019)].copy(),
        "test": df[df["season_end_year"] >= 2020].copy(),
    }


def metrics(y_true: Iterable[float], y_pred: Iterable[float]) -> dict[str, float]:
    y_true = np.asarray(list(y_true), dtype=float)
    y_pred = np.asarray(list(y_pred), dtype=float)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def formula_predict(df: pd.DataFrame, target: str) -> np.ndarray:
    if target == "actual_ortg":
        return (
            df["league_ortg_estimate"]
            + 85 * df["w_relative_ts"].fillna(0)
            + 0.18 * (df["w_points_per_75"].fillna(df["w_points_per_75"].median()) - 15)
            + 0.14 * (df["w_assist_pct"].fillna(df["w_assist_pct"].median()) - 13)
            - 0.22 * (df["w_turnover_pct"].fillna(df["w_turnover_pct"].median()) - 12)
            + 5.0 * df["team_3par_from_players"].fillna(0)
            + 2.0 * df["team_ftr_from_players"].fillna(0)
            - 4.0 * df["usage_pressure"].fillna(0)
        ).to_numpy()
    if target == "actual_drtg":
        return (
            df["league_ortg_estimate"]
            - 0.9 * (df["defense_box_proxy"].fillna(df["defense_box_proxy"].median()) - df["defense_box_proxy"].median())
            - 0.12 * (df["w_defensive_rebounds_per_75"].fillna(df["w_defensive_rebounds_per_75"].median()) - df["w_defensive_rebounds_per_75"].median())
            - 0.25 * (df["w_steals_per_75"].fillna(df["w_steals_per_75"].median()) - df["w_steals_per_75"].median())
            - 0.18 * (df["w_blocks_per_75"].fillna(df["w_blocks_per_75"].median()) - df["w_blocks_per_75"].median())
        ).to_numpy()
    return (
        df["league_pace"]
        + 0.35 * (df["w_points_per_75"].fillna(df["w_points_per_75"].median()) - df["w_points_per_75"].median())
        - 10.0 * (df["minutes_hhi"].fillna(df["minutes_hhi"].median()) - df["minutes_hhi"].median())
        + 0.18 * (df["w_steals_per_75"].fillna(df["w_steals_per_75"].median()) - df["w_steals_per_75"].median())
    ).to_numpy()


def fit_models(df: pd.DataFrame) -> tuple[dict, dict]:
    splits = temporal_split(df)
    model_specs = {
        "formula": None,
        "linear": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LinearRegression()),
        "ridge": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), RidgeCV(alphas=[0.1, 1, 3, 10, 30, 100])),
        "hist_gradient_boosting": make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingRegressor(max_iter=160, learning_rate=0.04, l2_regularization=0.1, random_state=7),
        ),
    }
    results: dict[str, dict] = {}
    fitted: dict[str, dict] = {}
    for target in ["actual_ortg", "actual_drtg", "actual_pace"]:
        results[target] = {}
        fitted[target] = {}
        for name, model in model_specs.items():
            if name == "formula":
                pred_val = formula_predict(splits["validation"], target)
                pred_test = formula_predict(splits["test"], target)
            else:
                target_model = clone(model)
                target_model.fit(splits["train"][FEATURES], splits["train"][target])
                pred_val = target_model.predict(splits["validation"][FEATURES])
                pred_test = target_model.predict(splits["test"][FEATURES])
                fitted[target][name] = target_model
            results[target][name] = {
                "validation": metrics(splits["validation"][target], pred_val),
                "test": metrics(splits["test"][target], pred_test),
            }
    return results, fitted


def coefficient_sanity(fitted: dict[str, dict]) -> dict:
    sanity: dict[str, dict] = {}
    for target, models in fitted.items():
        ridge = models.get("ridge")
        if not ridge:
            continue
        coefs = ridge.named_steps["ridgecv"].coef_
        items = sorted(zip(FEATURES, coefs), key=lambda x: abs(x[1]), reverse=True)
        sanity[target] = {
            "top_coefficients": [{"feature": f, "coefficient": float(c)} for f, c in items[:12]],
            "flags": [],
        }
        lookup = dict(zip(FEATURES, coefs))
        if target == "actual_ortg" and lookup.get("w_turnover_pct", 0) > 0:
            sanity[target]["flags"].append("Ridge assigns positive offensive coefficient to turnover percentage.")
        if target == "actual_ortg" and lookup.get("w_relative_ts", 0) < 0:
            sanity[target]["flags"].append("Ridge assigns negative offensive coefficient to relative true shooting.")
        if target == "actual_drtg" and lookup.get("defense_box_proxy", 0) > 0:
            sanity[target]["flags"].append("Ridge says better box-score defense increases DRtg; defensive signal is unstable.")
    return sanity


def select_recommended(results: dict) -> dict[str, str]:
    # Recommendation is deliberately conservative: learned models can reconstruct
    # historical ORtg, but defense/pace need robust era-anchored baselines before
    # they are safe for custom all-time roster extrapolation.
    return {
        "actual_ortg": "ridge",
        "actual_drtg": "formula",
        "actual_pace": "formula",
    }


def predictions_table(df: pd.DataFrame, fitted: dict, selected: dict) -> pd.DataFrame:
    out = df[["season", "team_name", "actual_ortg", "actual_drtg", "actual_pace"]].copy()
    for target, model_name in selected.items():
        pred_col = target.replace("actual_", "pred_")
        if model_name == "formula":
            out[pred_col] = formula_predict(df, target)
        else:
            out[pred_col] = fitted[target][model_name].predict(df[FEATURES])
    return out


def famous_team_checks(preds: pd.DataFrame) -> list[dict]:
    rows = []
    for season, team in FAMOUS_TEAMS:
        r = preds[(preds["season"] == season) & (preds["team_name"] == team)].iloc[0]
        rows.append(
            {
                "season": season,
                "team_name": team,
                "actual_ortg": round(float(r["actual_ortg"]), 2),
                "pred_ortg": round(float(r["pred_ortg"]), 2),
                "actual_drtg": round(float(r["actual_drtg"]), 2),
                "pred_drtg": round(float(r["pred_drtg"]), 2),
                "actual_pace": round(float(r["actual_pace"]), 2),
                "pred_pace": round(float(r["pred_pace"]), 2),
            }
        )
    return rows


def default_custom_minutes(entries: list[dict]) -> list[dict]:
    # Transparent V1 proposal for custom rosters: historical MPG as evidence, capped.
    # If the user supplies only a starting five, leave the remaining minutes explicit
    # instead of pretending five players can carry a full 240-minute game.
    base = []
    for entry in entries:
        cap = 40 if entry["slot"] in {"PG", "SG", "SF", "PF", "C"} else 28
        floor = 28 if entry["slot"] in {"PG", "SG", "SF", "PF", "C"} else 8
        mpg = entry.get("minutes_per_game")
        estimate = min(cap, max(floor, mpg if not pd.isna(mpg) else floor))
        base.append({**entry, "raw_minutes": estimate})
    total = sum(x["raw_minutes"] for x in base)
    if total <= 240:
        return [{**x, "projected_minutes": x["raw_minutes"]} for x in base]
    return [{**x, "projected_minutes": x["raw_minutes"] / total * 240} for x in base]


def custom_team_profile(players: pd.DataFrame, fitted: dict, selected: dict) -> dict:
    entries = []
    for slot, player_name, season in CUSTOM_TEAM:
        row = players[(players["player_name"] == player_name) & (players["season"] == season)].iloc[0].to_dict()
        row["slot"] = slot
        entries.append(row)
    rotation = default_custom_minutes(entries)
    group = pd.DataFrame(rotation).copy()
    group = add_derived_player_features(group)
    group["minutes"] = group["projected_minutes"] * 82
    group["team_id"] = "custom:all-time-demo"
    group["team_name"] = "Curry/Jordan/LeBron/Duncan/Shaq Demo"
    group["season"] = "custom-neutral"
    group["season_end_year"] = 2017
    league = pd.read_csv(PROCESSED / "league_seasons.csv")
    target_league = league[league["season"] == "2016-17"].iloc[0]
    team_stub = pd.DataFrame(
        [
            {
                "season": "custom-neutral",
                "season_end_year": 2017,
                "team_id": "custom:all-time-demo",
                "team_name": "Curry/Jordan/LeBron/Duncan/Shaq Demo",
                "wins": np.nan,
                "losses": np.nan,
                "off_rtg": np.nan,
                "def_rtg": np.nan,
                "pace": np.nan,
                "net_rtg": np.nan,
            }
        ]
    )
    league_stub = pd.DataFrame(
        [
            {
                "season": "custom-neutral",
                "season_end_year": 2017,
                "league_ts_pct": target_league["league_ts_pct"],
                "league_pace": target_league["league_pace"],
                "league_points_per_75": target_league["league_points_per_75"],
            }
        ]
    )
    rows = build_team_rows(group, team_stub, league_stub)
    row = rows.iloc[0]
    formula_preds = {
        target.replace("actual_", "projected_"): float(formula_predict(rows, target)[0])
        for target in ["actual_ortg", "actual_drtg", "actual_pace"]
    }
    learned_preds = {}
    for target, model_name in selected.items():
        if model_name == "formula":
            value = formula_predict(rows, target)[0]
        else:
            value = fitted[target][model_name].predict(rows[FEATURES])[0]
        learned_preds[target.replace("actual_", "projected_")] = float(value)
    selected_minutes = sum(x["projected_minutes"] for x in rotation)
    return {
        "team_name": "Curry/Jordan/LeBron/Duncan/Shaq Demo",
        "target_environment": "2016-17 neutral NBA environment",
        "rotation": [
            {
                "slot": x["slot"],
                "player_name": x["player_name"],
                "season": x["season"],
                "historical_team": x["team"],
                "historical_mpg": None if pd.isna(x["minutes_per_game"]) else round(float(x["minutes_per_game"]), 1),
                "projected_minutes": round(float(x["projected_minutes"]), 1),
            }
            for x in rotation
        ],
        "profile_features": {
            key: round(float(row[key]), 4) if not pd.isna(row[key]) else None
            for key in [
                "w_points_per_75",
                "w_relative_ts",
                "w_usage_pct",
                "weighted_usage_top3",
                "usage_pressure",
                "team_3par_from_players",
                "team_ftr_from_players",
                "spacing_proxy",
                "rebounding_proxy",
                "defense_box_proxy",
            ]
        },
        "projections": {k: round(v, 2) for k, v in formula_preds.items()},
        "transparent_formula_projections": {k: round(v, 2) for k, v in formula_preds.items()},
        "selected_model_raw_projections": {k: round(v, 2) for k, v in learned_preds.items()},
        "unassigned_bench_minutes": round(max(0.0, 240 - selected_minutes), 1),
        "caveats": [
            "Actual high-usage stars cannot all retain original usage; usage pressure is included as a penalty/proxy, not chemistry.",
            "Individual defense is weakly observed in current historical PlayerSeason data.",
            "Only five players were supplied; remaining bench minutes are explicit instead of silently invented.",
            "Learned reconstruction models can extrapolate badly on all-star custom rosters, so the transparent formula projection is shown as the safer prototype output.",
            "Projection is a team-strength profile, not a simulated win total.",
        ],
    }


def write_report(summary: dict) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_table = []
    for target, models in summary["model_metrics"].items():
        for model, values in models.items():
            metrics_table.append(
                f"| {target.replace('actual_', '')} | {model} | "
                f"{values['validation']['mae']:.2f} | {values['validation']['rmse']:.2f} | {values['validation']['r2']:.3f} | "
                f"{values['test']['mae']:.2f} | {values['test']['rmse']:.2f} | {values['test']['r2']:.3f} |"
            )
    famous_rows = [
        f"| {r['season']} | {r['team_name']} | {r['actual_ortg']:.1f} | {r['pred_ortg']:.1f} | {r['actual_drtg']:.1f} | {r['pred_drtg']:.1f} | {r['actual_pace']:.1f} | {r['pred_pace']:.1f} |"
        for r in summary["famous_team_checks"]
    ]
    custom = summary["custom_team_profile"]
    rotation_rows = [
        f"| {r['slot']} | {r['player_name']} | {r['season']} | {r['historical_team']} | {r['historical_mpg']:.1f} | {r['projected_minutes']:.1f} |"
        for r in custom["rotation"]
    ]
    report = f"""# Simulation Engine Design Spike

## Verdict

**{summary['verdict']}**

The current historical data is strong enough to build a transparent V1 team-strength profile for offense and pace. It is not strong enough to present defense or game-level win probability as definitive without additional game-level and stronger defensive data. The recommended next build should expose uncertainty clearly and keep the simulator labeled as an experimental historical translation.

## Reconstruction Dataset

- Team-season rows: {summary['dataset']['team_rows']}
- Seasons: {summary['dataset']['season_start']} through {summary['dataset']['season_end']}
- Team roster source: `player_seasons.csv` plus `player_team_splits.csv`
- Team target source: `team_seasons.csv`
- League context source: `league_seasons.csv`

Rows use actual historical player minutes for validation. Traded players use team-specific split rows for the teams they actually played for.

## Feature Set

Features are minutes-weighted, not equal-player averages:

- Per-75 production: points, assists, rebounds, turnovers, steals, blocks, offensive rebounds, defensive rebounds
- Efficiency and style: relative TS, TS%, 3PAr, FTr
- Creation/ball control: usage, AST%, TOV%, top-three usage pressure
- Rebounding and defense proxies: TRB%, defensive rebounds, steals, blocks, PER-based box proxy
- Rotation structure: top 1/3/5 minute share, minutes HHI, players above 500/1000 minutes
- Era context: league estimated ORtg, league pace, league TS%

No 0-100 ratings, chemistry, or arbitrary overalls were created.

## Split

- Train: 1979-80 through 2013-14
- Validation: 2014-15 through 2018-19
- Test: 2019-20 through 2025-26

This is temporal validation to reduce season leakage.

## Model Results

| Target | Model | Val MAE | Val RMSE | Val R2 | Test MAE | Test RMSE | Test R2 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(metrics_table)}

Recommended selected approaches:

- ORtg reconstruction: `{summary['selected_models']['actual_ortg']}`
- DRtg prototype baseline: `{summary['selected_models']['actual_drtg']}`
- Pace prototype baseline: `{summary['selected_models']['actual_pace']}`

Ridge is acceptable for historical ORtg reconstruction because it is stable on validation/test and remains interpretable. DRtg and Pace use formula baselines for the prototype recommendation because the learned models either underperform the temporal test split or extrapolate poorly for custom all-time rosters.

## Coefficient / Feature Sanity

```json
{json.dumps(summary['coefficient_sanity'], indent=2)}
```

Main interpretation:

- Offensive estimates are meaningfully informed by league environment, shooting efficiency, creation, turnover pressure, and spacing.
- Pace estimates are fairly stable because league pace provides a strong era anchor.
- Defensive estimates are materially weaker. Box-score defensive proxies have limited signal and should be treated as uncertainty-heavy.

## Famous-Team Sanity Checks

| Season | Team | Actual ORtg | Pred ORtg | Actual DRtg | Pred DRtg | Actual Pace | Pred Pace |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(famous_rows)}

These examples are sanity checks only. Models were trained and selected on broad temporal splits, not tuned to famous teams.

## Rotation Recommendation

Historical validation uses actual minutes.

For custom V1 rosters, use a transparent default:

1. Start from each selected PlayerSeason's historical MPG.
2. Starters have a floor near starter minutes and a cap around 40.
3. Bench players have lower floors and caps.
4. Normalize projected player minutes to total exactly 240.
5. Show the projected minutes and allow later user overrides before production simulation.

This avoids equal-player weighting while keeping assumptions visible.

## Usage Normalization

The spike uses `weighted_usage_top3` and `usage_pressure` as transparent diminishing-return signals. A future roster with Jordan, LeBron, and Curry cannot assume each keeps full historical usage. V1 should normalize expected usage so total lineup/team usage is coherent, then apply a modest penalty when too much usage is concentrated among high-usage creators. This is not chemistry; it is a usage accounting constraint.

## Custom All-Time Roster Prototype

Target environment: {custom['target_environment']}

| Slot | Player | Season | Historical Team | Historical MPG | Projected Minutes |
| --- | --- | --- | --- | ---: | ---: |
{chr(10).join(rotation_rows)}

Intermediate profile:

```json
{json.dumps(custom['profile_features'], indent=2)}
```

Projected profile:

```json
{json.dumps(custom['projections'], indent=2)}
```

Raw selected-model extrapolation, shown for audit only:

```json
{json.dumps(custom['selected_model_raw_projections'], indent=2)}
```

Unassigned bench minutes: {custom['unassigned_bench_minutes']}

Caveats:

{chr(10).join(f'- {c}' for c in custom['caveats'])}

## Cross-Era Methodology

Recommended V1 default: **user-selectable target era with neutral modern default**.

For the first simulator build, use a target league environment such as 2016-17 or a selectable season. Translate player/team profiles through era-relative features:

- Use relative TS and season-relative percentiles/z-scores for player quality signals.
- Use the target era's league ORtg, league pace, and league TS as anchors.
- Preserve style indicators like 3PAr rather than pretending a 1980 shot profile automatically becomes modern spacing.

Avoid modeling every rule/officiating change in V1.

## Game Simulation Design

Do not productionize yet, but the recommended structure is:

1. Estimate Team A offensive strength in target environment.
2. Estimate Team B defensive strength in target environment.
3. Blend expected points per 100 possessions from A offense vs B defense.
4. Repeat for B offense vs A defense.
5. Estimate game pace from both teams' pace profiles and target era.
6. Convert expected points per 100 possessions to expected score.
7. Sample game outcomes from calibrated score/efficiency variance.

The current processed historical season data does not contain game-level margins, home/away splits, or team-game pace. Therefore game variance and home-court should not be calibrated from the current dataset alone.

## Home Court

Current historical processed data is not sufficient to estimate home-court effect. Required data: historical team game logs with home/away, points for/against, possessions or pace proxy, date, and season. Likely sources include Basketball-Reference team game logs or other public game-log datasets. Codex can likely acquire Basketball-Reference game logs programmatically with the same caching discipline, subject to access reliability.

## Additional Data Needed

- Team game logs: calibrate score variance, home-court, game-to-game efficiency spread.
- Opponent-adjusted team/player context: reduce bias from teammate/team environment.
- Stronger individual defense data: DBPM/DRB/STL/BLK alone is not enough for confident defensive simulation.
- Lineup/minutes data if future versions model fit more precisely.

## Uncertainty

Biggest uncertainty sources:

- Individual defense
- Mixed-era translation
- Usage and role fit
- Bench strength/minute allocation
- Lack of lineup data
- Historical missing fields
- Rule/officiating differences
- No game-level variance calibration in current processed data

## Recommendation

**{summary['verdict']}**

Build the next iteration as a transparent prototype only after adding game-log data for variance/home-court and improving the defensive model story. Offense and pace are viable as V1 ingredients; defense and win probability need stronger support.
"""
    (REPORT_DIR / "SIMULATION_ENGINE_DESIGN.md").write_text(report, encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    players, splits, teams, leagues = load_inputs()
    roster = roster_rows(players, splits)
    team_rows = build_team_rows(roster, teams, leagues)
    team_rows = team_rows.dropna(subset=["actual_ortg", "actual_drtg", "actual_pace"]).copy()
    results, fitted = fit_models(team_rows)
    selected = select_recommended(results)
    sanity = coefficient_sanity(fitted)
    preds = predictions_table(team_rows, fitted, selected)
    famous = famous_team_checks(preds)
    custom = custom_team_profile(players, fitted, selected)

    summary = {
        "dataset": {
            "team_rows": int(len(team_rows)),
            "season_start": str(team_rows["season"].iloc[team_rows["season_end_year"].argmin()]),
            "season_end": str(team_rows["season"].iloc[team_rows["season_end_year"].argmax()]),
            "train_rows": int(len(temporal_split(team_rows)["train"])),
            "validation_rows": int(len(temporal_split(team_rows)["validation"])),
            "test_rows": int(len(temporal_split(team_rows)["test"])),
        },
        "feature_columns": FEATURES,
        "model_metrics": results,
        "selected_models": selected,
        "coefficient_sanity": sanity,
        "famous_team_checks": famous,
        "custom_team_profile": custom,
        "verdict": "NEEDS MORE DATA / MODELING",
    }
    team_rows.to_csv(OUT_DIR / "team_reconstruction_features.csv", index=False)
    preds.to_csv(OUT_DIR / "team_reconstruction_predictions.csv", index=False)
    (OUT_DIR / "simulation_engine_spike_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
