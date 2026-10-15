"""Offline advanced-field ingestion extension; never fetches or rewrites canonical data."""
from __future__ import annotations

import pandas as pd
from lxml import html

from build_player_season_prototype import (
    cache_path, read_cached_html, safe_float,
)

DEFENSIVE_FIELDS = {
    "DBPM": "defensive_box_plus_minus",
    "DWS": "defensive_win_shares",
    "STL%": "steal_pct",
    "BLK%": "block_pct",
    "DRB%": "defensive_rebound_pct",
}


def export_cached_profiles(years):
    """Read row-level source IDs (names are not unique); keep missing values null."""
    records = []
    stats = dict(zip(DEFENSIVE_FIELDS, ["dbpm", "dws", "stl_pct", "blk_pct", "drb_pct"]))
    for year in sorted(set(years)):
        path = cache_path(int(year), "advanced")
        if not path.exists():
            raise FileNotFoundError(f"Offline audit requires existing cache: {path}")
        table = html.fromstring(read_cached_html(path)).xpath("//table")[0]
        for tr in table.xpath(".//tr"):
            cells = {td.get("data-stat"): td for td in tr.xpath("./td")}
            identity = next((td.get("data-append-csv") for td in cells.values()
                             if td.get("data-append-csv")), None)
            if not identity:
                continue
            team = cells.get("team_name_abbr")
            if team is None:
                raise ValueError(f"No source team cell in {path}")
            row = dict(player_id="br:" + identity, season_end_year=int(year), team=team.text_content())
            row["source_minutes"] = safe_float(cells["mp"].text_content()) if "mp" in cells else None
            for source, destination in DEFENSIVE_FIELDS.items():
                cell = cells.get(stats[source])
                row[destination] = safe_float(cell.text_content()) if cell is not None else None
            records.append(row)
    result = pd.DataFrame(records).drop_duplicates()
    keys = ["player_id", "season_end_year", "team"]
    if result.duplicated(keys).any():
        raise ValueError("Ambiguous source defensive rows: " + str(result[result.duplicated(keys, keep=False)].head(12).to_dict("records")))
    return result
