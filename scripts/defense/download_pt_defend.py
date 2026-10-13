"""Download NBA Stats LeagueDashPtDefend tables.

This downloader is intentionally conservative and resumable. It writes one CSV
per season, season type, and defense category plus a JSON status manifest.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = ROOT / "data" / "raw" / "defense" / "nba_pt_defend"
STATUS_PATH = ROOT / "data" / "raw" / "defense" / "metadata" / "pt_defend_download_status.json"

CATEGORIES = [
    "Overall",
    "2 Pointers",
    "3 Pointers",
    "Less Than 6Ft",
    "Less Than 10Ft",
    "Greater Than 15Ft",
]
SEASON_TYPES = ["Regular Season", "Playoffs"]
SEASONS = [f"{year}-{str(year + 1)[-2:]}" for year in range(2014, 2026)]

HEADERS = {
    "Host": "stats.nba.com",
    "Connection": "keep-alive",
    "Accept": "application/json, text/plain, */*",
    "x-nba-stats-token": "true",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    ),
    "x-nba-stats-origin": "stats",
    "Origin": "https://www.nba.com",
    "Referer": "https://www.nba.com/",
    "Accept-Language": "en-US,en;q=0.9",
}


def slug(value: str) -> str:
    return (
        value.lower()
        .replace(" ", "_")
        .replace("+", "plus")
        .replace("-", "_")
        .replace("than", "than")
    )


def season_type_dir(value: str) -> str:
    return "regular_season" if value == "Regular Season" else "playoffs"


def endpoint_params(season: str, season_type: str, category: str) -> dict[str, str]:
    params = {
        "College": "",
        "Conference": "",
        "Country": "",
        "DateFrom": "",
        "DateTo": "",
        "DefenseCategory": category,
        "Division": "",
        "DraftPick": "",
        "DraftYear": "",
        "GameSegment": "",
        "Height": "",
        "LastNGames": "",
        "LeagueID": "00",
        "Location": "",
        "Month": "",
        "OpponentTeamID": "",
        "Outcome": "",
        "PORound": "",
        "PerMode": "Totals",
        "Period": "",
        "PlayerExperience": "",
        "PlayerID": "",
        "PlayerPosition": "",
        "Season": season,
        "SeasonSegment": "",
        "SeasonType": season_type,
        "StarterBench": "",
        "TeamID": "",
        "VsConference": "",
        "VsDivision": "",
        "Weight": "",
    }
    return params


def parse_table(payload: dict[str, Any]) -> pd.DataFrame:
    result_sets = payload.get("resultSets") or payload.get("resultSet") or []
    if isinstance(result_sets, dict):
        result_sets = [result_sets]
    for result in result_sets:
        name = str(result.get("name", "")).lower()
        headers = result.get("headers")
        rows = result.get("rowSet")
        if headers is not None and rows is not None and ("defend" in name or rows):
            return pd.DataFrame(rows, columns=headers)
    return pd.DataFrame()


def read_status() -> list[dict[str, Any]]:
    if STATUS_PATH.exists():
        return json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    return []


def write_status(records: list[dict[str, Any]]) -> None:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")


def fetch_one(
    season: str,
    season_type: str,
    category: str,
    timeout: int,
    retries: int,
    backoff: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    url = "https://stats.nba.com/stats/leaguedashptdefend"
    params = endpoint_params(season, season_type, category)
    last_error = ""
    for attempt in range(retries + 1):
        try:
            full_url = f"{url}?{urlencode(params)}"
            request = Request(full_url, headers=HEADERS)
            with urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed NBA Stats endpoint.
                status = getattr(response, "status", 200)
                body = response.read().decode("utf-8", errors="replace")
            if status != 200:
                last_error = f"HTTP {status}: {body[:300]}"
                raise RuntimeError(last_error)
            payload = json.loads(body)
            table = parse_table(payload)
            return table, {
                "success": not table.empty,
                "empty": table.empty,
                "failed": False,
                "error": None,
                "request_url": full_url,
            }
        except Exception as exc:  # noqa: BLE001 - every request should be recorded.
            last_error = repr(exc)
            if attempt < retries:
                time.sleep(backoff * (2**attempt))
    return pd.DataFrame(), {
        "success": False,
        "empty": False,
        "failed": True,
        "error": last_error,
        "request_url": f"{url}?{urlencode(params)}",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Re-download files that already exist.")
    parser.add_argument("--delay", type=float, default=3.0, help="Seconds to sleep between requests.")
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--backoff", type=float, default=2.0)
    parser.add_argument("--season", action="append", choices=SEASONS)
    args = parser.parse_args()

    seasons = args.season or SEASONS
    status = read_status()
    status_by_key = {
        (r["season"], r["season_type"], r["defense_category"]): r for r in status
    }
    for season in seasons:
        for season_type in SEASON_TYPES:
            for category in CATEGORIES:
                rel_dir = RAW_ROOT / season_type_dir(season_type) / season
                rel_dir.mkdir(parents=True, exist_ok=True)
                out_path = rel_dir / f"{slug(category)}.csv"
                key = (season, season_type, category)
                if out_path.exists() and out_path.stat().st_size > 0 and not args.force:
                    record = {
                        "season": season,
                        "season_type": season_type,
                        "defense_category": category,
                        "success": True,
                        "empty": False,
                        "failed": False,
                        "rows": int(pd.read_csv(out_path, nrows=0).shape[0]),
                        "path": str(out_path.relative_to(ROOT)),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "error": None,
                        "notes": "Skipped existing file; use --force to refresh.",
                    }
                    status_by_key[key] = record
                    write_status(list(status_by_key.values()))
                    continue

                table, result = fetch_one(
                    season,
                    season_type,
                    category,
                    timeout=args.timeout,
                    retries=args.retries,
                    backoff=args.backoff,
                )
                if not table.empty:
                    table.to_csv(out_path, index=False)
                record = {
                    "season": season,
                    "season_type": season_type,
                    "defense_category": category,
                    **result,
                    "rows": int(len(table)),
                    "path": str(out_path.relative_to(ROOT)) if out_path.exists() else None,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                status_by_key[key] = record
                write_status(list(status_by_key.values()))
                time.sleep(args.delay)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
