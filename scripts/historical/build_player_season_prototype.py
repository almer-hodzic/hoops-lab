#!/usr/bin/env python3
"""Build a small historical NBA player/team season feasibility prototype.

The prototype intentionally uses a narrow set of representative seasons and
transparent formulas. Raw Basketball-Reference HTML is cached locally so the
pipeline can be rerun without repeatedly hitting the source.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.request import Request, urlopen

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "historical" / "raw" / "basketball-reference"
PROCESSED_DIR = ROOT / "data" / "historical" / "processed"
METADATA_DIR = ROOT / "data" / "historical" / "metadata"

SOURCE_BASE = "https://www.basketball-reference.com/leagues"
USER_AGENT = "HoopsLab historical feasibility research (contact: local portfolio project)"


@dataclass(frozen=True)
class PlayerTarget:
    player_name: str
    season_end_year: int
    expected_team: str | None = None


@dataclass(frozen=True)
class TeamTarget:
    team_name: str
    season_end_year: int


PLAYER_TARGETS = [
    PlayerTarget("Kareem Abdul-Jabbar", 1977, "LAL"),
    PlayerTarget("Larry Bird", 1986, "BOS"),
    PlayerTarget("Michael Jordan", 1991, "CHI"),
    PlayerTarget("Shaquille O'Neal", 2000, "LAL"),
    PlayerTarget("LeBron James", 2013, "MIA"),
    PlayerTarget("Stephen Curry", 2016, "GSW"),
    PlayerTarget("Nikola Jokic", 2023, "DEN"),
]

TEAM_TARGETS = [
    TeamTarget("Boston Celtics", 1986),
    TeamTarget("Chicago Bulls", 1996),
    TeamTarget("Los Angeles Lakers", 2001),
    TeamTarget("Miami Heat", 2013),
    TeamTarget("Golden State Warriors", 2017),
]

REQUIRED_SEASONS = sorted(
    {p.season_end_year for p in PLAYER_TARGETS} | {t.season_end_year for t in TEAM_TARGETS}
)


def season_label(end_year: int) -> str:
    return f"{end_year - 1}-{str(end_year)[-2:]}"


def clean_team_name(value: object) -> str:
    return str(value).replace("*", "").strip()


def ascii_fold(value: object) -> str:
    text = str(value)
    return "".join(
        char for char in unicodedata.normalize("NFKD", text) if not unicodedata.combining(char)
    )


def fix_text(value: object) -> object:
    if not isinstance(value, str):
        return value
    for encoding in ("latin1", "cp1252"):
        try:
            return value.encode(encoding).decode("utf-8")
        except UnicodeError:
            pass
    return value


def normalize_col(name: object) -> str:
    if isinstance(name, tuple):
        parts = [str(x) for x in name if not str(x).startswith("Unnamed")]
        name = "_".join(parts) if len(parts) > 1 else (parts[-1] if parts else str(name[-1]))
    return str(name).strip()


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [normalize_col(c) for c in out.columns]
    return out


def to_number(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def safe_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(x):
        return None
    return x


def safe_div(num: object, den: object) -> float | None:
    n = safe_float(num)
    d = safe_float(den)
    if n is None or d in (None, 0):
        return None
    return n / d


def value_from(row: pd.Series, *names: str) -> object:
    for name in names:
        if name in row:
            return row.get(name)
    return None


def fetch_url(url: str, dest: Path, refresh: bool) -> None:
    if dest.exists() and not refresh:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=45) as response:
        dest.write_bytes(response.read())
    time.sleep(3)


def source_url(year: int, kind: str) -> str:
    suffix = "" if kind == "league" else f"_{kind}"
    return f"{SOURCE_BASE}/NBA_{year}{suffix}.html"


def cache_path(year: int, kind: str) -> Path:
    return RAW_DIR / f"NBA_{year}_{kind}.html"


def read_tables(year: int, kind: str, refresh: bool) -> list[pd.DataFrame]:
    url = source_url(year, kind)
    path = cache_path(year, kind)
    fetch_url(url, path, refresh)
    return pd.read_html(path)


def clean_player_table(df: pd.DataFrame) -> pd.DataFrame:
    out = flatten_columns(df)
    out = out[out["Player"].astype(str) != "Player"].copy()
    out["Player"] = out["Player"].map(fix_text)
    for col in out.columns:
        if col not in {"Player", "Team", "Pos", "Awards"}:
            out[col] = to_number(out[col])
    return out


def clean_team_misc_table(df: pd.DataFrame) -> pd.DataFrame:
    out = flatten_columns(df)
    out = out[out["Team"].astype(str) != "Team"].copy()
    out["Team"] = out["Team"].map(clean_team_name)
    for col in out.columns:
        if col not in {"Team", "Arena"}:
            out[col] = to_number(out[col])
    return out


def clean_team_per_game_table(df: pd.DataFrame) -> pd.DataFrame:
    out = flatten_columns(df)
    out = out[out["Team"].astype(str) != "Team"].copy()
    out["Team"] = out["Team"].map(clean_team_name)
    for col in out.columns:
        if col != "Team":
            out[col] = to_number(out[col])
    return out


def find_table(tables: list[pd.DataFrame], required: set[str], preferred_absent: set[str] | None = None) -> pd.DataFrame:
    preferred_absent = preferred_absent or set()
    for table in tables:
        cols = set(flatten_columns(table).columns)
        if required.issubset(cols) and not (cols & preferred_absent):
            return table
    for table in tables:
        cols = set(flatten_columns(table).columns)
        if required.issubset(cols):
            return table
    raise ValueError(f"No table found with columns: {sorted(required)}")


def pick_player_row(df: pd.DataFrame, target: PlayerTarget) -> pd.Series | None:
    target_name = ascii_fold(target.player_name).lower()
    matches = df[df["Player"].map(lambda x: ascii_fold(x).lower() == target_name)]
    if matches.empty:
        return None
    if "Team" in matches.columns:
        exact_team = matches[matches["Team"] == target.expected_team]
        if not exact_team.empty:
            return exact_team.iloc[0]
        total = matches[matches["Team"] == "TOT"]
        if not total.empty:
            return total.iloc[0]
    return matches.sort_values("MP", ascending=False).iloc[0]


def team_lookup(misc: pd.DataFrame, per_game: pd.DataFrame) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for _, row in misc.iterrows():
        rows[str(row["Team"])] = {
            "team_name": row["Team"],
            "wins": safe_float(row.get("W")),
            "losses": safe_float(row.get("L")),
            "pace": safe_float(row.get("Pace")),
            "off_rtg": safe_float(row.get("ORtg")),
            "def_rtg": safe_float(row.get("DRtg")),
            "net_rtg": safe_float(row.get("NRtg")),
            "team_ts_pct": safe_float(row.get("TS%")),
            "team_3par": safe_float(row.get("3PAr")),
                "team_tov_pct": safe_float(value_from(row, "Offense Four Factors_TOV%", "TOV%")),
                "team_orb_pct": safe_float(value_from(row, "Offense Four Factors_ORB%", "ORB%")),
        }
    for _, row in per_game.iterrows():
        name = str(row["Team"])
        rows.setdefault(name, {"team_name": name})
        rows[name].update(
            {
                "team_points_per_game": safe_float(row.get("PTS")),
                "team_assists_per_game": safe_float(row.get("AST")),
                "team_rebounds_per_game": safe_float(row.get("TRB")),
            }
        )
    return rows


def league_context(totals: pd.DataFrame, team_misc: pd.DataFrame) -> dict:
    totals_numeric = totals.copy()
    for col in ["PTS", "FGA", "FTA", "AST", "TRB", "TOV"]:
        if col not in totals_numeric.columns:
            totals_numeric[col] = pd.NA
    pts = totals_numeric["PTS"].sum(skipna=True)
    fga = totals_numeric["FGA"].sum(skipna=True)
    fta = totals_numeric["FTA"].sum(skipna=True)
    league_ts = safe_div(pts, 2 * (fga + 0.44 * fta))
    return {
        "league_ts_pct": league_ts,
        "league_pace": safe_float(team_misc["Pace"].mean(skipna=True)) if "Pace" in team_misc else None,
    }


def estimate_possessions(minutes: object, pace: object) -> float | None:
    mp = safe_float(minutes)
    p = safe_float(pace)
    if mp is None or p is None:
        return None
    return (mp / 48.0) * p


def per_75(stat: object, est_possessions: object) -> float | None:
    return safe_div(stat, est_possessions) * 75 if safe_div(stat, est_possessions) is not None else None


def build_player_rows(seasons: dict[int, dict]) -> list[dict]:
    rows: list[dict] = []
    for target in PLAYER_TARGETS:
        season = seasons[target.season_end_year]
        totals_row = pick_player_row(season["player_totals"], target)
        adv_row = pick_player_row(season["player_advanced"], target)
        if totals_row is None:
            rows.append(
                {
                    "player_name": target.player_name,
                    "season": season_label(target.season_end_year),
                    "source_status": "missing_player_totals",
                }
            )
            continue

        team_abbr = totals_row.get("Team")
        team_name = season["team_abbr_to_name"].get(team_abbr)
        team_ctx = season["team_by_name"].get(team_name or "", {})
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

        rows.append(
            {
                "player_id": None,
                "player_name": target.player_name,
                "source_player_name": totals_row.get("Player"),
                "season": season_label(target.season_end_year),
                "season_end_year": target.season_end_year,
                "team": team_abbr,
                "team_name": team_name,
                "age": safe_float(totals_row.get("Age")),
                "position": totals_row.get("Pos"),
                "games": safe_float(totals_row.get("G")),
                "games_started": safe_float(totals_row.get("GS")),
                "minutes": safe_float(totals_row.get("MP")),
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
                "estimated_on_court_possessions": est_poss,
                "league_ts_pct": league_ts,
                "points_per_75": per_75(totals_row.get("PTS"), est_poss),
                "assists_per_75": per_75(totals_row.get("AST"), est_poss),
                "rebounds_per_75": per_75(totals_row.get("TRB"), est_poss),
                "turnovers_per_75": per_75(tov, est_poss),
                "steals_per_75": per_75(totals_row.get("STL"), est_poss),
                "blocks_per_75": per_75(totals_row.get("BLK"), est_poss),
                "three_point_attempt_rate": safe_div(three_pa, fga),
                "free_throw_rate": safe_div(fta, fga),
                "relative_ts": player_ts - league_ts if player_ts is not None and league_ts is not None else None,
                "league_relative_scoring_per_75": None,
                "source_status": "loaded",
                "source": "Basketball-Reference season totals and advanced tables",
            }
        )
    points = [r["points_per_75"] for r in rows if r.get("points_per_75") is not None]
    mean_points = sum(points) / len(points) if points else None
    for row in rows:
        if row.get("points_per_75") is not None and mean_points:
            row["league_relative_scoring_per_75"] = row["points_per_75"] - mean_points
    return rows


def build_team_rows(seasons: dict[int, dict]) -> list[dict]:
    rows: list[dict] = []
    for target in TEAM_TARGETS:
        season = seasons[target.season_end_year]
        team = season["team_by_name"].get(target.team_name)
        if not team:
            rows.append(
                {
                    "team_name": target.team_name,
                    "season": season_label(target.season_end_year),
                    "source_status": "missing_team",
                }
            )
            continue
        rows.append(
            {
                "team_name": target.team_name,
                "season": season_label(target.season_end_year),
                "season_end_year": target.season_end_year,
                "wins": team.get("wins"),
                "losses": team.get("losses"),
                "pace": team.get("pace"),
                "off_rtg": team.get("off_rtg"),
                "def_rtg": team.get("def_rtg"),
                "net_rtg": team.get("net_rtg"),
                "team_ts_pct": team.get("team_ts_pct"),
                "team_3par": team.get("team_3par"),
                "team_tov_pct": team.get("team_tov_pct"),
                "team_orb_pct": team.get("team_orb_pct"),
                "team_points_per_game": team.get("team_points_per_game"),
                "team_assists_per_game": team.get("team_assists_per_game"),
                "team_rebounds_per_game": team.get("team_rebounds_per_game"),
                "source_status": "loaded",
                "source": "Basketball-Reference league team per-game and misc tables",
            }
        )
    return rows


def load_season(year: int, refresh: bool) -> dict:
    totals = clean_player_table(read_tables(year, "totals", refresh)[0])
    advanced = clean_player_table(read_tables(year, "advanced", refresh)[0])
    league_tables = read_tables(year, "league", refresh)
    team_per_game = clean_team_per_game_table(
        find_table(league_tables, {"Team", "G", "MP", "FG", "FGA", "PTS"}, {"PW", "ORtg", "Pace"})
    )
    team_misc = clean_team_misc_table(find_table(league_tables, {"Team", "W", "L", "ORtg", "DRtg", "Pace"}))
    team_by_name = team_lookup(team_misc, team_per_game)
    abbr_to_name = {
        "ATL": "Atlanta Hawks",
        "BOS": "Boston Celtics",
        "BRK": "Brooklyn Nets",
        "BKN": "Brooklyn Nets",
        "CHA": "Charlotte Hornets",
        "CHO": "Charlotte Hornets",
        "CHH": "Charlotte Hornets",
        "CHI": "Chicago Bulls",
        "CLE": "Cleveland Cavaliers",
        "DAL": "Dallas Mavericks",
        "DEN": "Denver Nuggets",
        "DET": "Detroit Pistons",
        "GSW": "Golden State Warriors",
        "HOU": "Houston Rockets",
        "IND": "Indiana Pacers",
        "LAC": "Los Angeles Clippers",
        "LAL": "Los Angeles Lakers",
        "MEM": "Memphis Grizzlies",
        "MIA": "Miami Heat",
        "MIL": "Milwaukee Bucks",
        "MIN": "Minnesota Timberwolves",
        "NJN": "New Jersey Nets",
        "NOH": "New Orleans Hornets",
        "NOP": "New Orleans Pelicans",
        "NYK": "New York Knicks",
        "OKC": "Oklahoma City Thunder",
        "ORL": "Orlando Magic",
        "PHI": "Philadelphia 76ers",
        "PHO": "Phoenix Suns",
        "POR": "Portland Trail Blazers",
        "SAC": "Sacramento Kings",
        "SAS": "San Antonio Spurs",
        "SEA": "Seattle SuperSonics",
        "TOR": "Toronto Raptors",
        "UTA": "Utah Jazz",
        "VAN": "Vancouver Grizzlies",
        "WAS": "Washington Wizards",
        "WSB": "Washington Bullets",
    }
    return {
        "player_totals": totals,
        "player_advanced": advanced,
        "team_per_game": team_per_game,
        "team_misc": team_misc,
        "team_by_name": team_by_name,
        "team_abbr_to_name": abbr_to_name,
        "league": league_context(totals, team_misc),
    }


def validate_players(df: pd.DataFrame) -> list[str]:
    issues: list[str] = []
    loaded = df[df["source_status"] == "loaded"]
    if len(loaded) != len(PLAYER_TARGETS):
        issues.append(f"Expected {len(PLAYER_TARGETS)} loaded player rows, found {len(loaded)}.")
    if loaded.duplicated(["player_name", "season"]).any():
        issues.append("Duplicate player-season rows detected.")
    for _, row in loaded.iterrows():
        if row["games"] <= 0 or row["minutes"] <= 0:
            issues.append(f"Impossible games/minutes for {row['player_name']} {row['season']}.")
        if row["true_shooting_pct"] is not None and not (0.0 < row["true_shooting_pct"] < 1.0):
            issues.append(f"Impossible TS% for {row['player_name']} {row['season']}.")
        if row["points_per_75"] is not None and row["points_per_75"] <= 0:
            issues.append(f"Impossible points_per_75 for {row['player_name']} {row['season']}.")
    return issues


def write_outputs(player_rows: list[dict], team_rows: list[dict], refresh: bool) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    players = pd.DataFrame(player_rows)
    teams = pd.DataFrame(team_rows)
    players.to_csv(PROCESSED_DIR / "player_season_prototype.csv", index=False)
    teams.to_csv(PROCESSED_DIR / "team_season_prototype.csv", index=False)
    comparison_cols = [
        "player_name",
        "season",
        "team",
        "minutes",
        "points_per_75",
        "true_shooting_pct",
        "league_ts_pct",
        "relative_ts",
        "usage_pct",
        "assist_pct",
        "turnover_pct",
        "total_rebound_pct",
        "three_point_attempt_rate",
        "free_throw_rate",
        "team_pace",
        "source_status",
    ]
    players[comparison_cols].to_csv(PROCESSED_DIR / "player_season_comparison.csv", index=False)
    issues = validate_players(players)
    manifest = {
        "source": "Basketball-Reference",
        "source_base_url": SOURCE_BASE,
        "raw_cache_dir": str(RAW_DIR.relative_to(ROOT)),
        "processed_dir": str(PROCESSED_DIR.relative_to(ROOT)),
        "seasons_fetched": REQUIRED_SEASONS,
        "refresh": refresh,
        "validation_issues": issues,
        "outputs": [
            "data/historical/processed/player_season_prototype.csv",
            "data/historical/processed/player_season_comparison.csv",
            "data/historical/processed/team_season_prototype.csv",
        ],
    }
    (METADATA_DIR / "historical_player_data_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    if issues:
        raise SystemExit("Validation failed:\n" + "\n".join(issues))


def build(refresh: bool) -> None:
    seasons = {year: load_season(year, refresh) for year in REQUIRED_SEASONS}
    write_outputs(build_player_rows(seasons), build_team_rows(seasons), refresh)


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="Re-download cached source HTML.")
    args = parser.parse_args(list(argv) if argv is not None else None)
    build(refresh=args.refresh)


if __name__ == "__main__":
    main()
