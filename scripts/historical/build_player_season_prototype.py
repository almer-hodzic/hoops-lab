#!/usr/bin/env python3
"""Build a small historical NBA player/team season feasibility prototype."""

from __future__ import annotations

import argparse
import io
import json
import math
import re
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "historical" / "raw" / "basketball-reference"
PROCESSED_DIR = ROOT / "data" / "historical" / "processed"
METADATA_DIR = ROOT / "data" / "historical" / "metadata"

SOURCE_BASE = "https://www.basketball-reference.com/leagues"
USER_AGENT = "HoopsLab historical feasibility research (contact: local portfolio project)"
SCHEMA_VERSION = "historical_player_season_prototype_v2"
REQUEST_TIMEOUT_SECONDS = 45
MAX_DOWNLOAD_ATTEMPTS = 3
REQUEST_DELAY_SECONDS = 3


@dataclass(frozen=True)
class PlayerTarget:
    player_name: str
    season_end_year: int
    expected_team: str | None = None


@dataclass(frozen=True)
class TeamTarget:
    team_name: str
    season_end_year: int


@dataclass(frozen=True)
class SourceFetch:
    year: int
    kind: str
    url: str
    cache_path: str
    cache_status: str
    attempts: int


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

TEAM_IDENTITY_BY_ABBR = {
    "ATL": {"team_id": "team:atlanta-hawks", "franchise_id": "franchise:atlanta-hawks", "display_name": "Atlanta Hawks"},
    "BOS": {"team_id": "team:boston-celtics", "franchise_id": "franchise:boston-celtics", "display_name": "Boston Celtics"},
    "BRK": {"team_id": "team:brooklyn-nets", "franchise_id": "franchise:nets", "display_name": "Brooklyn Nets"},
    "BKN": {"team_id": "team:brooklyn-nets", "franchise_id": "franchise:nets", "display_name": "Brooklyn Nets"},
    "NJN": {"team_id": "team:new-jersey-nets", "franchise_id": "franchise:nets", "display_name": "New Jersey Nets"},
    "CHA": {"team_id": "team:charlotte-hornets-current", "franchise_id": "franchise:charlotte-hornets-current", "display_name": "Charlotte Hornets"},
    "CHO": {"team_id": "team:charlotte-hornets-current", "franchise_id": "franchise:charlotte-hornets-current", "display_name": "Charlotte Hornets"},
    "CHH": {"team_id": "team:charlotte-hornets-original", "franchise_id": "franchise:new-orleans-pelicans", "display_name": "Charlotte Hornets"},
    "CHA_2005_2014": {"team_id": "team:charlotte-bobcats", "franchise_id": "franchise:charlotte-hornets-current", "display_name": "Charlotte Bobcats"},
    "CHI": {"team_id": "team:chicago-bulls", "franchise_id": "franchise:chicago-bulls", "display_name": "Chicago Bulls"},
    "CLE": {"team_id": "team:cleveland-cavaliers", "franchise_id": "franchise:cleveland-cavaliers", "display_name": "Cleveland Cavaliers"},
    "DAL": {"team_id": "team:dallas-mavericks", "franchise_id": "franchise:dallas-mavericks", "display_name": "Dallas Mavericks"},
    "DEN": {"team_id": "team:denver-nuggets", "franchise_id": "franchise:denver-nuggets", "display_name": "Denver Nuggets"},
    "DET": {"team_id": "team:detroit-pistons", "franchise_id": "franchise:detroit-pistons", "display_name": "Detroit Pistons"},
    "GSW": {"team_id": "team:golden-state-warriors", "franchise_id": "franchise:warriors", "display_name": "Golden State Warriors"},
    "HOU": {"team_id": "team:houston-rockets", "franchise_id": "franchise:rockets", "display_name": "Houston Rockets"},
    "IND": {"team_id": "team:indiana-pacers", "franchise_id": "franchise:indiana-pacers", "display_name": "Indiana Pacers"},
    "LAC": {"team_id": "team:los-angeles-clippers", "franchise_id": "franchise:clippers", "display_name": "Los Angeles Clippers"},
    "LAL": {"team_id": "team:los-angeles-lakers", "franchise_id": "franchise:lakers", "display_name": "Los Angeles Lakers"},
    "MEM": {"team_id": "team:memphis-grizzlies", "franchise_id": "franchise:grizzlies", "display_name": "Memphis Grizzlies"},
    "VAN": {"team_id": "team:vancouver-grizzlies", "franchise_id": "franchise:grizzlies", "display_name": "Vancouver Grizzlies"},
    "MIA": {"team_id": "team:miami-heat", "franchise_id": "franchise:miami-heat", "display_name": "Miami Heat"},
    "MIL": {"team_id": "team:milwaukee-bucks", "franchise_id": "franchise:milwaukee-bucks", "display_name": "Milwaukee Bucks"},
    "MIN": {"team_id": "team:minnesota-timberwolves", "franchise_id": "franchise:minnesota-timberwolves", "display_name": "Minnesota Timberwolves"},
    "NOH": {"team_id": "team:new-orleans-hornets", "franchise_id": "franchise:new-orleans-pelicans", "display_name": "New Orleans Hornets"},
    "NOK": {"team_id": "team:new-orleans-oklahoma-city-hornets", "franchise_id": "franchise:new-orleans-pelicans", "display_name": "New Orleans/Oklahoma City Hornets"},
    "NOP": {"team_id": "team:new-orleans-pelicans", "franchise_id": "franchise:new-orleans-pelicans", "display_name": "New Orleans Pelicans"},
    "NYK": {"team_id": "team:new-york-knicks", "franchise_id": "franchise:new-york-knicks", "display_name": "New York Knicks"},
    "OKC": {"team_id": "team:oklahoma-city-thunder", "franchise_id": "franchise:thunder", "display_name": "Oklahoma City Thunder"},
    "SEA": {"team_id": "team:seattle-supersonics", "franchise_id": "franchise:thunder", "display_name": "Seattle SuperSonics"},
    "ORL": {"team_id": "team:orlando-magic", "franchise_id": "franchise:orlando-magic", "display_name": "Orlando Magic"},
    "PHI": {"team_id": "team:philadelphia-76ers", "franchise_id": "franchise:76ers", "display_name": "Philadelphia 76ers"},
    "PHO": {"team_id": "team:phoenix-suns", "franchise_id": "franchise:phoenix-suns", "display_name": "Phoenix Suns"},
    "POR": {"team_id": "team:portland-trail-blazers", "franchise_id": "franchise:portland-trail-blazers", "display_name": "Portland Trail Blazers"},
    "SAC": {"team_id": "team:sacramento-kings", "franchise_id": "franchise:kings", "display_name": "Sacramento Kings"},
    "SAS": {"team_id": "team:san-antonio-spurs", "franchise_id": "franchise:san-antonio-spurs", "display_name": "San Antonio Spurs"},
    "TOR": {"team_id": "team:toronto-raptors", "franchise_id": "franchise:toronto-raptors", "display_name": "Toronto Raptors"},
    "UTA": {"team_id": "team:utah-jazz", "franchise_id": "franchise:jazz", "display_name": "Utah Jazz"},
    "WAS": {"team_id": "team:washington-wizards", "franchise_id": "franchise:wizards", "display_name": "Washington Wizards"},
    "WSB": {"team_id": "team:washington-bullets", "franchise_id": "franchise:wizards", "display_name": "Washington Bullets"},
}

FIELD_PROVENANCE = {
    "player_efficiency_rating": "source_derived_basketball_reference",
    "win_shares": "source_derived_basketball_reference",
    "box_plus_minus": "source_derived_basketball_reference",
    "usage_pct": "source_derived_basketball_reference_or_unavailable",
    "assist_pct": "source_derived_basketball_reference",
    "turnover_pct": "source_derived_basketball_reference_or_unavailable",
    "total_rebound_pct": "source_derived_basketball_reference",
    "true_shooting_pct": "directly_sourced_from_basketball_reference_advanced_or_reconstructed_fallback",
    "team_pace": "directly_sourced_from_basketball_reference_team_misc",
    "league_pace": "reconstructed_from_team_games_and_team_pace",
    "league_ts_pct": "reconstructed_from_basketball_reference_team_totals",
    "estimated_possessions_minutes_team_pace": "estimated_minutes_team_pace",
    "points_per_75": "derived_from_points_and_estimated_possessions",
    "assists_per_75": "derived_from_assists_and_estimated_possessions",
    "rebounds_per_75": "derived_from_rebounds_and_estimated_possessions",
    "turnovers_per_75": "derived_or_unavailable",
    "three_point_attempt_rate": "derived_or_not_tracked_in_pre_3pt_era",
    "free_throw_rate": "derived_from_fta_and_fga",
}


def season_label(end_year: int) -> str:
    return f"{end_year - 1}-{str(end_year)[-2:]}"


def ascii_fold(value: object) -> str:
    return "".join(
        char for char in unicodedata.normalize("NFKD", str(value)) if not unicodedata.combining(char)
    )


def clean_team_name(value: object) -> str:
    return str(value).replace("*", "").strip()


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


def source_url(year: int, kind: str) -> str:
    suffix = "" if kind == "league" else f"_{kind}"
    return f"{SOURCE_BASE}/NBA_{year}{suffix}.html"


def cache_path(year: int, kind: str) -> Path:
    return RAW_DIR / f"NBA_{year}_{kind}.html"


def read_cached_html(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def fetch_url(url: str, dest: Path, refresh: bool) -> SourceFetch:
    if dest.exists() and not refresh:
        return SourceFetch(0, "", url, str(dest.relative_to(ROOT)), "cache_hit", 0)
    dest.parent.mkdir(parents=True, exist_ok=True)
    delay = REQUEST_DELAY_SECONDS
    last_error: Exception | None = None
    for attempt in range(1, MAX_DOWNLOAD_ATTEMPTS + 1):
        try:
            request = Request(url, headers={"User-Agent": USER_AGENT})
            with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                dest.write_bytes(response.read())
            time.sleep(REQUEST_DELAY_SECONDS)
            return SourceFetch(0, "", url, str(dest.relative_to(ROOT)), "downloaded", attempt)
        except (HTTPError, URLError, TimeoutError) as exc:
            last_error = exc
            if attempt < MAX_DOWNLOAD_ATTEMPTS:
                time.sleep(delay)
                delay *= 2
    raise RuntimeError(f"Failed to fetch {url} after {MAX_DOWNLOAD_ATTEMPTS} attempts: {last_error}")


def read_tables(year: int, kind: str, refresh: bool, fetch_log: list[dict]) -> list[pd.DataFrame]:
    url = source_url(year, kind)
    path = cache_path(year, kind)
    result = fetch_url(url, path, refresh)
    fetch_log.append(
        {
            **result.__dict__,
            "year": year,
            "kind": kind,
            "source_url": url,
            "cache_path": str(path.relative_to(ROOT)),
        }
    )
    return pd.read_html(io.StringIO(read_cached_html(path)))


def extract_player_ids(year: int, kind: str) -> dict[str, str]:
    html = read_cached_html(cache_path(year, kind))
    ids: dict[str, str] = {}
    pattern = re.compile(r'<a href="/players/[a-z]/([a-z0-9]+)\.html">([^<]+)</a>')
    for source_id, name in pattern.findall(html):
        ids[ascii_fold(name).lower()] = source_id
    return ids


def clean_player_table(df: pd.DataFrame) -> pd.DataFrame:
    out = flatten_columns(df)
    out = out[out["Player"].astype(str) != "Player"].copy()
    for col in out.columns:
        if col not in {"Player", "Team", "Pos", "Awards"}:
            out[col] = to_number(out[col])
    return out


def clean_team_table(df: pd.DataFrame) -> pd.DataFrame:
    out = flatten_columns(df)
    out = out[out["Team"].astype(str) != "Team"].copy()
    out["Team"] = out["Team"].map(clean_team_name)
    for col in out.columns:
        if col not in {"Team", "Arena"}:
            out[col] = to_number(out[col])
    return out


def find_table(tables: list[pd.DataFrame], required: set[str], selector=None) -> pd.DataFrame:
    candidates: list[pd.DataFrame] = []
    for table in tables:
        flat = flatten_columns(table)
        if required.issubset(set(flat.columns)):
            candidates.append(flat)
    if selector:
        selected = [table for table in candidates if selector(table)]
        if selected:
            return selected[0]
    if candidates:
        return candidates[0]
    raise ValueError(f"No table found with columns: {sorted(required)}")


def is_team_totals_table(df: pd.DataFrame) -> bool:
    mp = pd.to_numeric(df.get("MP"), errors="coerce")
    return bool(mp.max(skipna=True) and mp.max(skipna=True) > 10000)


def is_team_per_game_table(df: pd.DataFrame) -> bool:
    mp = pd.to_numeric(df.get("MP"), errors="coerce")
    return bool(mp.max(skipna=True) and 100 < mp.max(skipna=True) < 400)


def team_identity_from_abbr(abbr: str | None, season_end_year: int) -> dict:
    key = abbr or ""
    if key == "CHA" and 2005 <= season_end_year <= 2014:
        key = "CHA_2005_2014"
    identity = TEAM_IDENTITY_BY_ABBR.get(key)
    if not identity:
        raise ValueError(f"No team identity mapping for {abbr} in {season_end_year}.")
    return {"team_abbr": abbr, **identity}


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


def league_context(team_totals: pd.DataFrame, team_misc: pd.DataFrame) -> dict:
    totals = clean_team_table(team_totals)
    misc = clean_team_table(team_misc)
    league_points = totals["PTS"].sum(skipna=True)
    league_fga = totals["FGA"].sum(skipna=True)
    league_fta = totals["FTA"].sum(skipna=True)
    league_ts = safe_div(league_points, 2 * (league_fga + 0.44 * league_fta))
    pace_context = totals[["Team", "G"]].merge(misc[["Team", "Pace"]], on="Team", how="inner")
    pace_weight_den = pace_context["G"].sum(skipna=True)
    league_pace = safe_div((pace_context["Pace"] * pace_context["G"]).sum(skipna=True), pace_weight_den)
    league_est_possessions = (pace_context["Pace"] * pace_context["G"]).sum(skipna=True)
    return {
        "league_points": safe_float(league_points),
        "league_fga": safe_float(league_fga),
        "league_fta": safe_float(league_fta),
        "league_ts_pct": league_ts,
        "league_pace": league_pace,
        "league_estimated_team_possessions": safe_float(league_est_possessions),
        "league_points_per_75": safe_div(league_points, league_est_possessions) * 75
        if safe_div(league_points, league_est_possessions) is not None
        else None,
        "league_context_provenance": "team_totals_and_team_misc_no_player_rows",
    }


def estimate_possessions(minutes: object, pace: object) -> float | None:
    mp = safe_float(minutes)
    p = safe_float(pace)
    if mp is None or p is None:
        return None
    return (mp / 48.0) * p


def per_75(stat: object, est_possessions: object) -> float | None:
    ratio = safe_div(stat, est_possessions)
    return ratio * 75 if ratio is not None else None


def pick_player_row(df: pd.DataFrame, target: PlayerTarget) -> pd.Series | None:
    target_name = ascii_fold(target.player_name).lower()
    matches = df[df["Player"].map(lambda x: ascii_fold(x).lower() == target_name)]
    if matches.empty:
        return None
    exact_team = matches[matches["Team"] == target.expected_team]
    if not exact_team.empty:
        return exact_team.iloc[0]
    total = matches[matches["Team"] == "TOT"]
    if not total.empty:
        return total.iloc[0]
    return matches.sort_values("MP", ascending=False).iloc[0]


def availability_for_player(row: pd.Series) -> dict:
    return {
        "three_point_stats": "recorded" if safe_float(row.get("3PA")) is not None else "not_tracked_in_era",
        "turnovers": "recorded" if safe_float(row.get("TOV")) is not None else "source_unavailable_or_not_tracked",
        "usage_pct": "recorded" if "USG%" in row and safe_float(row.get("USG%")) is not None else "source_unavailable_or_not_tracked",
    }


def player_id_for(target: PlayerTarget, ids_by_kind: dict[str, dict[str, str]]) -> str | None:
    key = ascii_fold(target.player_name).lower()
    return ids_by_kind.get("totals", {}).get(key) or ids_by_kind.get("advanced", {}).get(key)


def build_player_rows(seasons: dict[int, dict]) -> list[dict]:
    rows: list[dict] = []
    for target in PLAYER_TARGETS:
        season = seasons[target.season_end_year]
        totals_row = pick_player_row(season["player_totals"], target)
        adv_row = pick_player_row(season["player_advanced"], target)
        if totals_row is None:
            rows.append({"player_name": target.player_name, "season": season_label(target.season_end_year), "source_status": "missing_player_totals"})
            continue

        source_player_id = player_id_for(target, season["player_ids"])
        if not source_player_id:
            raise ValueError(f"Missing Basketball-Reference source player id for {target.player_name}.")

        team_abbr = str(totals_row.get("Team"))
        team_identity = team_identity_from_abbr(team_abbr, target.season_end_year)
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
        league_points_per_75 = season["league"]["league_points_per_75"]
        points_per_75 = per_75(totals_row.get("PTS"), est_poss)

        rows.append(
            {
                "player_id": f"br:{source_player_id}",
                "source_player_id": source_player_id,
                "player_name": target.player_name,
                "source_player_name": adv_row.get("Player") if adv_row is not None else totals_row.get("Player"),
                "season": season_label(target.season_end_year),
                "season_end_year": target.season_end_year,
                "team": team_abbr,
                "team_id": team_identity["team_id"],
                "franchise_id": team_identity["franchise_id"],
                "team_name": team_identity["display_name"],
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
                "league_points": season["league"]["league_points"],
                "league_fga": season["league"]["league_fga"],
                "league_fta": season["league"]["league_fta"],
                "league_ts_pct": league_ts,
                "league_points_per_75": league_points_per_75,
                "estimated_possessions_minutes_team_pace": est_poss,
                "possession_estimate_method": "minutes_team_pace",
                "points_per_75": points_per_75,
                "assists_per_75": per_75(totals_row.get("AST"), est_poss),
                "rebounds_per_75": per_75(totals_row.get("TRB"), est_poss),
                "turnovers_per_75": per_75(tov, est_poss),
                "steals_per_75": per_75(totals_row.get("STL"), est_poss),
                "blocks_per_75": per_75(totals_row.get("BLK"), est_poss),
                "three_point_attempt_rate": safe_div(three_pa, fga),
                "free_throw_rate": safe_div(fta, fga),
                "relative_ts": player_ts - league_ts if player_ts is not None and league_ts is not None else None,
                "availability": json.dumps(availability_for_player(pd.concat([totals_row, adv_row]) if adv_row is not None else totals_row), ensure_ascii=False),
                "source_status": "loaded",
                "source": "Basketball-Reference season totals and advanced tables",
            }
        )
    return rows


def build_team_rows(seasons: dict[int, dict]) -> list[dict]:
    rows: list[dict] = []
    for target in TEAM_TARGETS:
        season = seasons[target.season_end_year]
        team = season["team_by_name"].get(target.team_name)
        if not team:
            rows.append({"team_name": target.team_name, "season": season_label(target.season_end_year), "source_status": "missing_team"})
            continue
        team_identity = next(
            identity for identity in TEAM_IDENTITY_BY_ABBR.values() if identity["display_name"] == target.team_name
        )
        rows.append(
            {
                "team_id": team_identity["team_id"],
                "franchise_id": team_identity["franchise_id"],
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


def load_season(year: int, refresh: bool, fetch_log: list[dict]) -> dict:
    totals = clean_player_table(read_tables(year, "totals", refresh, fetch_log)[0])
    advanced = clean_player_table(read_tables(year, "advanced", refresh, fetch_log)[0])
    player_ids = {"totals": extract_player_ids(year, "totals"), "advanced": extract_player_ids(year, "advanced")}
    league_tables = read_tables(year, "league", refresh, fetch_log)
    team_per_game = clean_team_table(find_table(league_tables, {"Team", "G", "MP", "FG", "FGA", "PTS"}, is_team_per_game_table))
    team_totals = clean_team_table(find_table(league_tables, {"Team", "G", "MP", "FG", "FGA", "FTA", "PTS"}, is_team_totals_table))
    team_misc = clean_team_table(find_table(league_tables, {"Team", "W", "L", "ORtg", "DRtg", "Pace"}))
    return {
        "player_totals": totals,
        "player_advanced": advanced,
        "player_ids": player_ids,
        "team_per_game": team_per_game,
        "team_totals": team_totals,
        "team_misc": team_misc,
        "team_by_name": team_lookup(team_misc, team_per_game),
        "league": league_context(team_totals, team_misc),
    }


def validate_players(df: pd.DataFrame) -> list[str]:
    issues: list[str] = []
    loaded = df[df["source_status"] == "loaded"]
    if len(loaded) != len(PLAYER_TARGETS):
        issues.append(f"Expected {len(PLAYER_TARGETS)} loaded player rows, found {len(loaded)}.")
    if loaded.duplicated(["player_id", "season"]).any():
        issues.append("Duplicate player-season rows detected.")
    if loaded["player_id"].isna().any() or loaded["source_player_id"].isna().any():
        issues.append("Missing stable Basketball-Reference player identity.")
    if loaded["source_player_name"].astype(str).str.contains("Ä", regex=False).any():
        issues.append("Unicode source names contain mojibake.")
    if "league_relative_scoring_per_75" in loaded.columns:
        issues.append("Invalid sample-relative scoring field still present.")
    kareem = loaded[loaded["player_name"] == "Kareem Abdul-Jabbar"].iloc[0]
    if not pd.isna(kareem["three_point_attempt_rate"]) or not pd.isna(kareem["turnovers_per_75"]):
        issues.append("Kareem 1976-77 historical unavailable stats should remain null.")
    for _, row in loaded.iterrows():
        if row["games"] <= 0 or row["minutes"] <= 0:
            issues.append(f"Impossible games/minutes for {row['player_name']} {row['season']}.")
        if not (0.0 < row["true_shooting_pct"] < 1.0):
            issues.append(f"Impossible TS% for {row['player_name']} {row['season']}.")
        if row["points_per_75"] <= 0:
            issues.append(f"Impossible points_per_75 for {row['player_name']} {row['season']}.")
        expected_poss = (row["minutes"] / 48.0) * row["team_pace"]
        if abs(row["estimated_possessions_minutes_team_pace"] - expected_poss) > 1e-9:
            issues.append(f"Possession estimate mismatch for {row['player_name']} {row['season']}.")
    return issues


def write_outputs(player_rows: list[dict], team_rows: list[dict], seasons: dict[int, dict], refresh: bool, fetch_log: list[dict]) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    players = pd.DataFrame(player_rows)
    teams = pd.DataFrame(team_rows)
    players.to_csv(PROCESSED_DIR / "player_season_prototype.csv", index=False, encoding="utf-8")
    teams.to_csv(PROCESSED_DIR / "team_season_prototype.csv", index=False, encoding="utf-8")
    comparison_cols = [
        "player_id",
        "player_name",
        "season",
        "team",
        "team_id",
        "franchise_id",
        "minutes",
        "points_per_75",
        "league_points_per_75",
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
        "league_pace",
        "possession_estimate_method",
        "source_status",
    ]
    players[comparison_cols].to_csv(PROCESSED_DIR / "player_season_comparison.csv", index=False, encoding="utf-8")
    issues = validate_players(players)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "Basketball-Reference",
        "source_base_url": SOURCE_BASE,
        "raw_cache_dir": str(RAW_DIR.relative_to(ROOT)),
        "processed_dir": str(PROCESSED_DIR.relative_to(ROOT)),
        "seasons_fetched": REQUIRED_SEASONS,
        "refresh": refresh,
        "request_timeout_seconds": REQUEST_TIMEOUT_SECONDS,
        "request_delay_seconds": REQUEST_DELAY_SECONDS,
        "download_attempts": MAX_DOWNLOAD_ATTEMPTS,
        "source_fetches": fetch_log,
        "field_provenance": FIELD_PROVENANCE,
        "team_identity_source": "explicit_audited_prototype_mapping",
        "league_context": {str(year): season["league"] for year, season in seasons.items()},
        "validation_issues": issues,
        "outputs": [
            "data/historical/processed/player_season_prototype.csv",
            "data/historical/processed/player_season_comparison.csv",
            "data/historical/processed/team_season_prototype.csv",
        ],
    }
    (METADATA_DIR / "historical_player_data_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    team_map = {
        abbr: identity for abbr, identity in sorted(TEAM_IDENTITY_BY_ABBR.items()) if abbr != "CHA_2005_2014"
    }
    (METADATA_DIR / "team_identity_map.json").write_text(
        json.dumps(team_map, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if issues:
        raise SystemExit("Validation failed:\n" + "\n".join(issues))


def build(refresh: bool) -> None:
    fetch_log: list[dict] = []
    seasons = {year: load_season(year, refresh, fetch_log) for year in REQUIRED_SEASONS}
    write_outputs(build_player_rows(seasons), build_team_rows(seasons), seasons, refresh, fetch_log)


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="Re-download cached source HTML.")
    args = parser.parse_args(list(argv) if argv is not None else None)
    build(refresh=args.refresh)


if __name__ == "__main__":
    main()
