#!/usr/bin/env python3
"""Smoke-test the HoopsLab historical player API against the real Web host."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "nba-clutch-agent" / "AiAgents.NbaClutchAgent.Web" / "AiAgents.NbaClutchAgent.Web.csproj"
BASE_URL = os.environ.get("HOOPSLAB_TEST_BASE_URL", "http://127.0.0.1:5299")


def get_json(path: str, expected_status: int = 200):
    try:
        with urlopen(f"{BASE_URL}{path}", timeout=20) as response:
            body = response.read().decode("utf-8")
            if response.status != expected_status:
                raise AssertionError(f"{path}: expected {expected_status}, got {response.status}")
            return json.loads(body) if body else None
    except HTTPError as exc:
        if exc.code != expected_status:
            body = exc.read().decode("utf-8", errors="replace")
            raise AssertionError(f"{path}: expected {expected_status}, got {exc.code}: {body}") from exc
        return json.loads(exc.read().decode("utf-8"))


def wait_for_server(process: subprocess.Popen[str]) -> None:
    deadline = time.time() + 45
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Web host exited early with code {process.returncode}")
        try:
            get_json("/ping")
            return
        except Exception:
            time.sleep(0.5)
    raise TimeoutError("Timed out waiting for Web host.")


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    env = os.environ.copy()
    env["ASPNETCORE_URLS"] = BASE_URL
    process = subprocess.Popen(
        ["dotnet", "run", "--no-build", "--project", str(PROJECT), "--urls", BASE_URL],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    try:
        wait_for_server(process)

        jordan_search = get_json("/api/historical/players?search=jordan&limit=5")
        jordan = next((p for p in jordan_search["items"] if p["playerName"] == "Michael Jordan"), None)
        assert_true(jordan is not None, "search should return Michael Jordan")
        assert_true(jordan["playerId"].startswith("br:"), "search should return stable player id")

        jokic_search = get_json("/api/historical/players?search=Jokic&limit=5")
        assert_true(any(p["playerName"] == "Nikola Jokić" for p in jokic_search["items"]), "ASCII Jokic search should find Nikola Jokić")
        jokic_unicode = get_json("/api/historical/players?search=Joki%C4%87&limit=5")
        assert_true(any(p["playerName"] == "Nikola Jokić" for p in jokic_unicode["items"]), "Unicode Jokić search should find Nikola Jokić")

        profile = get_json(f"/api/historical/players/{jordan['playerId']}")
        seasons = [s["season"] for s in profile["seasons"]]
        assert_true(seasons == sorted(seasons, key=lambda s: int("20" + s[-2:]) if s[-2:] < "50" else int("19" + s[-2:])), "season list should be chronological")
        assert_true("1990-91" in seasons, "Jordan profile should include 1990-91")

        jordan_1991_before = get_json(f"/api/historical/players/{jordan['playerId']}/seasons/1990-91")
        assert_true(jordan_1991_before["identity"]["playerName"] == "Michael Jordan", "Jordan 1990-91 retrieval failed")
        assert_true(jordan_1991_before["identity"]["team"] == "CHI", "Jordan 1990-91 team should be CHI")

        curry = next(p for p in get_json("/api/historical/players?search=curry&limit=8")["items"] if p["playerName"] == "Stephen Curry")
        curry_2016 = get_json(f"/api/historical/players/{curry['playerId']}/seasons/2015-16")
        assert_true(curry_2016["identity"]["team"] == "GSW", "Curry 2015-16 team should be GSW")

        get_json("/api/historical/players/br:not-a-player", expected_status=404)
        get_json(f"/api/historical/players/{jordan['playerId']}/seasons/1970-71", expected_status=404)

        kareem = next(p for p in get_json("/api/historical/players?search=kareem&limit=5")["items"] if p["playerName"] == "Kareem Abdul-Jabbar")
        kareem_1980 = get_json(f"/api/historical/players/{kareem['playerId']}/seasons/1979-80")
        assert_true(kareem_1980["identity"]["gamesStarted"] is None, "missing games_started should serialize as null")
        assert_true(kareem_1980["advanced"]["winSharesPer48"] is None, "unavailable WS/48 should serialize as null")

        comparison = get_json(
            f"/api/historical/compare?playerA={jordan['playerId']}&seasonA=1990-91&playerB={curry['playerId']}&seasonB=2015-16"
        )
        assert_true(comparison["playerA"]["identity"]["playerName"] == "Michael Jordan", "comparison player A mismatch")
        assert_true(comparison["playerB"]["identity"]["playerName"] == "Stephen Curry", "comparison player B mismatch")
        jordan_1991_after = get_json(f"/api/historical/players/{jordan['playerId']}/seasons/1990-91")
        assert_true(jordan_1991_before == jordan_1991_after, "comparison should not mutate season data")

        print(json.dumps({"status": "pass", "baseUrl": BASE_URL}, indent=2))
        return 0
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


if __name__ == "__main__":
    raise SystemExit(main())
