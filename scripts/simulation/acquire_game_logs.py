"""Acquire public, version-pinned team game logs; cache first, never authenticate.

Run once online, then use --offline to validate the same cached bytes.
Only the three small team/game tables are downloaded, not player or PBP data.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/simulation/raw/game_logs"
MANIFEST = ROOT / "data/simulation/game_log_source_manifest.json"
DATASET = "eoinamoore/historical-nba-data-and-player-box-scores"
VERSION = 515
API = "https://www.kaggle.com/api/v1/datasets/"
SOURCE = "https://www.kaggle.com/datasets/" + DATASET
FILES = {"Games.csv": "gameId", "TeamStatistics.csv": "gameId",
         "TeamStatisticsExtended.csv": "gameId"}


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def acquire(offline=False):
    RAW.mkdir(parents=True, exist_ok=True)
    existing = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else None
    if existing and existing["version"] != VERSION:
        raise RuntimeError("Cache belongs to another version; do not mix snapshots")
    records, requests = [], 0
    for filename, first_column in FILES.items():
        path = RAW / filename
        url = API + "download/" + DATASET + "/" + filename + f"?datasetVersionNumber={VERSION}"
        if not path.exists():
            if offline:
                raise FileNotFoundError(f"Download {filename} from {SOURCE}/versions/{VERSION} into {RAW}")
            print(f"Fetching public {filename} (version {VERSION})", flush=True)
            temporary = path.with_suffix(".partial")
            try:
                with urlopen(url, timeout=90) as response, temporary.open("wb") as stream:
                    content_type = response.headers.get("Content-Type", "")
                    if "html" in content_type.lower():
                        raise RuntimeError("Source returned HTML, not a public CSV; no bypass attempted")
                    size = 0
                    while chunk := response.read(1024 * 1024):
                        size += len(chunk)
                        if size > 100_000_000:
                            raise RuntimeError("Unexpected oversized team table")
                        stream.write(chunk)
                with temporary.open(encoding="utf-8-sig", newline="") as stream:
                    if next(csv.reader(stream))[0] != first_column:
                        raise RuntimeError("Unexpected source schema; cache not accepted")
                temporary.replace(path)
                requests += 1
            except HTTPError as exc:
                raise RuntimeError(f"Public access failed ({exc.code}). Manually download {filename} "
                                   f"from {SOURCE}/versions/{VERSION} and place it in {RAW}. "
                                   "The pipeline will validate, map identities, and build canonical games.") from exc
            finally:
                if temporary.exists():
                    temporary.unlink()
        digest = fingerprint(path)
        if existing:
            expected = next(record for record in existing["files"] if record["file"] == filename)
            if digest != expected["sha256"]:
                raise RuntimeError("Raw cached bytes changed: " + filename)
        records.append(dict(file=filename, url=url, sha256=digest, bytes=path.stat().st_size))
    if not existing:
        if offline:
            raise RuntimeError("Offline raw cache has no provenance manifest")
        with urlopen(API + "view/" + DATASET + f"?datasetVersionNumber={VERSION}", timeout=45) as response:
            metadata = json.load(response)
        requests += 1
        if metadata["currentVersionNumber"] != VERSION:
            raise RuntimeError("Dataset version changed; inspect pinned version metadata before proceeding")
        (RAW / "dataset_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        record = dict(source=SOURCE, version=VERSION, upstream="NBA.com, per dataset author",
                      license=metadata["licenseName"], source_updated=metadata["lastUpdated"],
                      acquired_utc=datetime.now(timezone.utc).isoformat(),
                      method="Anonymous public Kaggle file endpoint, pinned datasetVersionNumber; no scraping/authentication",
                      raw_directory=str(RAW.relative_to(ROOT)).replace("\\", "/"),
                      initial_pipeline_requests=requests, files=records,
                      probes="Basketball-Reference reachable; obsolete 538 forecast CSV redirected to HTML and was rejected",
                      cache_policy="Existing bytes must match SHA256; offline reruns make zero requests")
        MANIFEST.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(status="cached", network_requests_this_run=requests,
                          files=[r["file"] for r in records])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true")
    acquire(parser.parse_args().offline)
