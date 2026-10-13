#!/usr/bin/env python3
"""Build full HoopsLab historical NBA player/team/league season datasets."""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

from build_player_season_prototype import (
    FIELD_PROVENANCE,
    METADATA_DIR,
    PROCESSED_DIR,
    RAW_DIR,
    SCHEMA_VERSION,
    SOURCE_BASE,
    TEAM_IDENTITY_BY_ABBR,
    ascii_fold,
    availability_for_player,
    clean_team_table,
    estimate_possessions,
    extract_player_ids,
    load_season,
    per_75,
    safe_div,
    safe_float,
    season_label,
    team_identity_from_abbr,
)


START_SEASON_END_YEAR = 1980
LATEST_COMPLETED_SEASON_END_YEAR = 2026
FULL_SCHEMA_VERSION = "historical_full_player_team_league_v1"
MIN_PERCENTILE_SAMPLE = 50
MIN_COMPARISON_MINUTES = 500
QUALIFIED_ROTATION_MINUTES = 1000

ERA_METRICS = [
    "points_per_75",
    "assists_per_75",
    "rebounds_per_75",
    "relative_ts",
    "usage_pct",
    "assist_pct",
    "total_rebound_pct",
    "steals_per_75",
    "blocks_per_75",
    "turnovers_per_75",
]


def team_identity_from_name(team_name: str, season_end_year: int) -> dict:
    if team_name == "Charlotte Bobcats":
        return {"team_abbr": "CHA", **TEAM_IDENTITY_BY_ABBR["CHA_2005_2014"]}
    for abbr, identity in TEAM_IDENTITY_BY_ABBR.items():
        if abbr == "CHA_2005_2014":
            continue
        if identity["display_name"] == team_name:
            if team_name == "Charlotte Hornets" and season_end_year < 2003:
                return {"team_abbr": "CHH", **TEAM_IDENTITY_BY_ABBR["CHH"]}
            if team_name == "Charlotte Hornets" and season_end_year >= 2015:
                return {"team_abbr": "CHO", **TEAM_IDENTITY_BY_ABBR["CHO"]}
            return {"team_abbr": abbr, **identity}
    raise ValueError(f"No team identity mapping for {team_name} in {season_end_year}.")


def player_id_for(name: object, ids_by_kind: dict[str, dict[str, str]]) -> str | None:
    key = ascii_fold(name).lower()
    return ids_by_kind.get("totals", {}).get(key) or ids_by_kind.get("advanced", {}).get(key)


def source_player_name_for(source_id: str | None, ids_by_kind: dict[str, dict[str, str]], fallback: str) -> str:
    if not source_id:
        return fallback
    # The display name from pandas is already Unicode-safe for cached UTF-8 pages.
    return fallback


def pick_advanced_row(advanced: pd.DataFrame, source_name: object, team: object) -> pd.Series | None:
    key = ascii_fold(source_name).lower()
    matches = advanced[advanced["Player"].map(lambda x: ascii_fold(x).lower() == key)]
    if matches.empty:
        return None
    team_matches = matches[matches["Team"] == team]
    if not team_matches.empty:
        return team_matches.iloc[0]
    total = matches[matches["Team"].map(is_multi_team_abbr)]
    if not total.empty:
        return total.iloc[0]
    return matches.sort_values("MP", ascending=False).iloc[0]


def is_multi_team_abbr(team_abbr: object) -> bool:
    value = str(team_abbr)
    return value == "TOT" or value.endswith("TM")


def make_player_row(
    totals_row: pd.Series,
    adv_row: pd.Series | None,
    season: dict,
    season_end_year: int,
    row_type: str,
) -> dict:
    source_name = str(totals_row.get("Player"))
    source_player_id = player_id_for(source_name, season["player_ids"])
    if not source_player_id:
        raise ValueError(f"Missing source player id for {source_name} {season_end_year}.")

    team_abbr = str(totals_row.get("Team"))
    multi_team = is_multi_team_abbr(team_abbr)
    if multi_team:
        team_identity = {
            "team_abbr": "TOT",
            "team_id": "team:multiple",
            "franchise_id": "franchise:multiple",
            "display_name": "Multiple Teams",
        }
        pace = season["league"]["league_pace"]
    else:
        team_identity = team_identity_from_abbr(team_abbr, season_end_year)
        team_ctx = season["team_by_name"].get(team_identity["display_name"], {})
        pace = team_ctx.get("pace") or season["league"]["league_pace"]

    est_poss = estimate_possessions(totals_row.get("MP"), pace)
    fga = safe_float(totals_row.get("FGA"))
    fta = safe_float(totals_row.get("FTA"))
    three_pa = safe_float(totals_row.get("3PA"))
    tov = safe_float(totals_row.get("TOV"))
    player_ts = safe_float(adv_row.get("TS%")) if adv_row is not None else None
    if player_ts is None:
        player_ts = safe_div(totals_row.get("PTS"), 2 * ((fga or 0) + 0.44 * (fta or 0)))
    league_ts = season["league"]["league_ts_pct"]

    return {
        "player_id": f"br:{source_player_id}",
        "source_player_id": source_player_id,
        "player_name": source_name,
        "source_player_name": source_name,
        "season": season_label(season_end_year),
        "season_end_year": season_end_year,
        "row_type": row_type,
        "is_multi_team_total": multi_team,
        "team": team_abbr,
        "team_id": team_identity["team_id"],
        "franchise_id": team_identity["franchise_id"],
        "team_name": team_identity["display_name"],
        "age": safe_float(totals_row.get("Age")),
        "position": totals_row.get("Pos"),
        "games": safe_float(totals_row.get("G")),
        "games_started": safe_float(totals_row.get("GS")),
        "minutes": safe_float(totals_row.get("MP")),
        "minutes_per_game": safe_div(totals_row.get("MP"), totals_row.get("G")),
        "points": safe_float(totals_row.get("PTS")),
        "field_goals": safe_float(totals_row.get("FG")),
        "field_goal_attempts": fga,
        "two_point_makes": safe_float(totals_row.get("2P")),
        "two_point_attempts": safe_float(totals_row.get("2PA")),
        "three_point_makes": safe_float(totals_row.get("3P")),
        "three_point_attempts": three_pa,
        "free_throws": safe_float(totals_row.get("FT")),
        "free_throw_attempts": fta,
        "offensive_rebounds": safe_float(totals_row.get("ORB")),
        "defensive_rebounds": safe_float(totals_row.get("DRB")),
        "total_rebounds": safe_float(totals_row.get("TRB")),
        "assists": safe_float(totals_row.get("AST")),
        "steals": safe_float(totals_row.get("STL")),
        "blocks": safe_float(totals_row.get("BLK")),
        "turnovers": tov,
        "personal_fouls": safe_float(totals_row.get("PF")),
        "true_shooting_pct": player_ts,
        "assist_pct": safe_float(adv_row.get("AST%")) if adv_row is not None else None,
        "turnover_pct": safe_float(adv_row.get("TOV%")) if adv_row is not None and "TOV%" in adv_row else None,
        "total_rebound_pct": safe_float(adv_row.get("TRB%")) if adv_row is not None else None,
        "usage_pct": safe_float(adv_row.get("USG%")) if adv_row is not None and "USG%" in adv_row else None,
        "player_efficiency_rating": safe_float(adv_row.get("PER")) if adv_row is not None else None,
        "win_shares": safe_float(adv_row.get("WS")) if adv_row is not None else None,
        "box_plus_minus": safe_float(adv_row.get("BPM")) if adv_row is not None else None,
        "team_pace": pace,
        "league_pace": season["league"]["league_pace"],
        "league_points": season["league"]["league_points"],
        "league_fga": season["league"]["league_fga"],
        "league_fta": season["league"]["league_fta"],
        "league_ts_pct": league_ts,
        "league_points_per_75": season["league"]["league_points_per_75"],
        "estimated_possessions_minutes_team_pace": est_poss,
        "possession_estimate_method": "minutes_team_pace",
        "points_per_75": per_75(totals_row.get("PTS"), est_poss),
        "assists_per_75": per_75(totals_row.get("AST"), est_poss),
        "rebounds_per_75": per_75(totals_row.get("TRB"), est_poss),
        "turnovers_per_75": per_75(tov, est_poss),
        "steals_per_75": per_75(totals_row.get("STL"), est_poss),
        "blocks_per_75": per_75(totals_row.get("BLK"), est_poss),
        "three_point_attempt_rate": safe_div(three_pa, fga),
        "free_throw_rate": safe_div(fta, fga),
        "relative_ts": player_ts - league_ts if player_ts is not None and league_ts is not None else None,
        "comparison_sample": comparison_sample(safe_float(totals_row.get("MP"))),
        "availability": json.dumps(availability_for_player(pd.concat([totals_row, adv_row]) if adv_row is not None else totals_row), ensure_ascii=False),
        "source_status": "loaded",
        "source": "Basketball-Reference season totals and advanced tables",
    }


def comparison_sample(minutes: float | None) -> str:
    if minutes is None:
        return "unknown"
    if minutes >= QUALIFIED_ROTATION_MINUTES:
        return "qualified_rotation"
    if minutes >= MIN_COMPARISON_MINUTES:
        return "comparison_population"
    return "low_sample"


def build_player_datasets(season_data: dict[int, dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    primary_rows: list[dict] = []
    split_rows: list[dict] = []
    for season_end_year, season in season_data.items():
        totals = season["player_totals"].copy()
        totals = totals[totals["Player"].map(lambda value: str(value) != "League Average")]
        season["player_advanced"] = season["player_advanced"][
            season["player_advanced"]["Player"].map(lambda value: str(value) != "League Average")
        ].copy()
        grouped = totals.groupby("Player", sort=False)
        for _, group in grouped:
            total_rows = group[group["Team"].map(is_multi_team_abbr)]
            if not total_rows.empty:
                primary_source = total_rows.iloc[0]
                split_source = group[~group["Team"].map(is_multi_team_abbr)]
            else:
                primary_source = group.sort_values("MP", ascending=False).iloc[0]
                split_source = pd.DataFrame(columns=group.columns)
            adv_row = pick_advanced_row(season["player_advanced"], primary_source["Player"], primary_source["Team"])
            primary_rows.append(make_player_row(primary_source, adv_row, season, season_end_year, "season_total"))
            for _, split in split_source.iterrows():
                split_adv = pick_advanced_row(season["player_advanced"], split["Player"], split["Team"])
                split_rows.append(make_player_row(split, split_adv, season, season_end_year, "team_split"))
    return pd.DataFrame(primary_rows), pd.DataFrame(split_rows)


def build_team_dataset(season_data: dict[int, dict]) -> pd.DataFrame:
    rows: list[dict] = []
    for season_end_year, season in season_data.items():
        team_misc = season["team_misc"].copy()
        team_misc = team_misc[team_misc["Team"].map(lambda value: str(value) != "League Average")]
        for _, row in team_misc.iterrows():
            name = str(row["Team"])
            identity = team_identity_from_name(name, season_end_year)
            per_game = season["team_by_name"].get(name, {})
            rows.append(
                {
                    "team_id": identity["team_id"],
                    "franchise_id": identity["franchise_id"],
                    "team_name": name,
                    "season": season_label(season_end_year),
                    "season_end_year": season_end_year,
                    "wins": safe_float(row.get("W")),
                    "losses": safe_float(row.get("L")),
                    "pace": safe_float(row.get("Pace")),
                    "off_rtg": safe_float(row.get("ORtg")),
                    "def_rtg": safe_float(row.get("DRtg")),
                    "net_rtg": safe_float(row.get("NRtg")),
                    "team_ts_pct": safe_float(row.get("TS%")),
                    "team_3par": safe_float(row.get("3PAr")),
                    "team_tov_pct": safe_float(row.get("Offense Four Factors_TOV%"))
                    if "Offense Four Factors_TOV%" in row
                    else safe_float(row.get("TOV%")),
                    "team_orb_pct": safe_float(row.get("Offense Four Factors_ORB%"))
                    if "Offense Four Factors_ORB%" in row
                    else safe_float(row.get("ORB%")),
                    "team_points_per_game": per_game.get("team_points_per_game"),
                    "team_assists_per_game": per_game.get("team_assists_per_game"),
                    "team_rebounds_per_game": per_game.get("team_rebounds_per_game"),
                    "source_status": "loaded",
                    "source": "Basketball-Reference league team per-game and misc tables",
                }
            )
    return pd.DataFrame(rows)


def build_league_dataset(season_data: dict[int, dict]) -> pd.DataFrame:
    rows: list[dict] = []
    for season_end_year, season in season_data.items():
        rows.append(
            {
                "season": season_label(season_end_year),
                "season_end_year": season_end_year,
                **season["league"],
            }
        )
    return pd.DataFrame(rows)


def add_era_context(players: pd.DataFrame) -> pd.DataFrame:
    out = players.copy()
    eligible = out["minutes"].fillna(0) >= MIN_COMPARISON_MINUTES
    for metric in ERA_METRICS:
        pct_col = f"{metric}_season_percentile"
        z_col = f"{metric}_season_z"
        out[pct_col] = pd.NA
        out[z_col] = pd.NA
        for season, group in out[eligible & out[metric].notna()].groupby("season"):
            if len(group) < MIN_PERCENTILE_SAMPLE:
                continue
            values = group[metric].astype(float)
            std = values.std(ddof=0)
            ranks = values.rank(method="average", pct=True) * 100
            out.loc[group.index, pct_col] = ranks
            if std and math.isfinite(std):
                out.loc[group.index, z_col] = (values - values.mean()) / std
    return out


def write_outputs(players: pd.DataFrame, splits: pd.DataFrame, teams: pd.DataFrame, leagues: pd.DataFrame, fetch_log: list[dict], started: float, refresh: bool) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    players.to_csv(PROCESSED_DIR / "player_seasons.csv", index=False, encoding="utf-8")
    splits.to_csv(PROCESSED_DIR / "player_team_splits.csv", index=False, encoding="utf-8")
    teams.to_csv(PROCESSED_DIR / "team_seasons.csv", index=False, encoding="utf-8")
    leagues.to_csv(PROCESSED_DIR / "league_seasons.csv", index=False, encoding="utf-8")

    sample_cols = [
        "player_id",
        "player_name",
        "season",
        "team",
        "points_per_75",
        "relative_ts",
        "usage_pct",
        "points_per_75_season_percentile",
        "relative_ts_season_percentile",
    ]
    players[sample_cols].head(200).to_csv(PROCESSED_DIR / "player_seasons_sample.csv", index=False, encoding="utf-8")
    teams.head(100).to_csv(PROCESSED_DIR / "team_seasons_sample.csv", index=False, encoding="utf-8")
    leagues.to_csv(PROCESSED_DIR / "league_seasons_sample.csv", index=False, encoding="utf-8")

    missingness = players.isna().mean().sort_values(ascending=False).to_dict()
    manifest = {
        "schema_version": FULL_SCHEMA_VERSION,
        "prototype_schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "season_range": {
            "start": season_label(START_SEASON_END_YEAR),
            "end": season_label(LATEST_COMPLETED_SEASON_END_YEAR),
            "start_end_year": START_SEASON_END_YEAR,
            "end_end_year": LATEST_COMPLETED_SEASON_END_YEAR,
        },
        "source": "Basketball-Reference",
        "source_base_url": SOURCE_BASE,
        "raw_cache_dir": str(RAW_DIR.relative_to(Path(__file__).resolve().parents[2])),
        "refresh": refresh,
        "row_counts": {
            "player_seasons": int(len(players)),
            "player_team_splits": int(len(splits)),
            "team_seasons": int(len(teams)),
            "league_seasons": int(len(leagues)),
            "unique_players": int(players["player_id"].nunique()),
        },
        "field_provenance": FIELD_PROVENANCE,
        "era_metric_rules": {
            "minimum_minutes": MIN_COMPARISON_MINUTES,
            "minimum_population": MIN_PERCENTILE_SAMPLE,
            "metrics": ERA_METRICS,
        },
        "source_fetches": fetch_log,
        "major_missingness": {k: v for k, v in missingness.items() if v >= 0.25},
        "runtime_seconds": round(time.perf_counter() - started, 2),
        "outputs": [
            "data/historical/processed/player_seasons.csv",
            "data/historical/processed/player_team_splits.csv",
            "data/historical/processed/team_seasons.csv",
            "data/historical/processed/league_seasons.csv",
        ],
    }
    (METADATA_DIR / "full_historical_ingestion_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (METADATA_DIR / "team_identity_map.json").write_text(
        json.dumps(TEAM_IDENTITY_BY_ABBR, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def build(refresh: bool) -> None:
    started = time.perf_counter()
    fetch_log: list[dict] = []
    season_data: dict[int, dict] = {}
    for year in range(START_SEASON_END_YEAR, LATEST_COMPLETED_SEASON_END_YEAR + 1):
        print(f"Loading {season_label(year)} ({year})...", flush=True)
        season_data[year] = load_season(year, refresh, fetch_log)
        print(f"Loaded {season_label(year)}", flush=True)
    print("Building player datasets...", flush=True)
    players, splits = build_player_datasets(season_data)
    print("Adding era context...", flush=True)
    players = add_era_context(players)
    print("Building team and league datasets...", flush=True)
    teams = build_team_dataset(season_data)
    leagues = build_league_dataset(season_data)
    print("Writing outputs...", flush=True)
    write_outputs(players, splits, teams, leagues, fetch_log, started, refresh)
    print(
        "Wrote full historical datasets: "
        f"{len(players)} player seasons, {len(splits)} team splits, "
        f"{len(teams)} team seasons, {len(leagues)} league seasons.",
        flush=True,
    )


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="Re-download cached source HTML.")
    args = parser.parse_args(list(argv) if argv is not None else None)
    build(refresh=args.refresh)


if __name__ == "__main__":
    main()
