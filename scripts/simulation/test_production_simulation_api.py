"""Exercise the real .NET proxy and Python simulator, then benchmark warm requests."""
from __future__ import annotations

import copy
import ctypes
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def free_port(exclude=()):
    # Avoid Windows ephemeral-port reservations when launching child servers.
    for port in range(18012, 18112):
        if port in exclude:
            continue
        with socket.socket() as sock:
            try:
                sock.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError("No available integration-test port")


def rss(pid):
    if os.name != "nt":
        return None
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [(name, ctypes.c_size_t) for name in
            ["PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage",
             "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage"]]
    kernel = ctypes.windll.kernel32
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    get_memory = ctypes.windll.psapi.GetProcessMemoryInfo
    get_memory.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    handle = kernel.OpenProcess(0x410, False, pid)
    if not handle:
        return None
    try:
        counters = Counters(); counters.cb = ctypes.sizeof(counters)
        if not get_memory(handle, ctypes.byref(counters), counters.cb):
            return None
        return round(counters.WorkingSetSize/1024**2, 3)
    finally:
        kernel.CloseHandle(handle)


def call(base, path, payload=None, expected=200, content_type="application/json"):
    data = json.dumps(payload).encode() if payload is not None else None
    request = Request(base+path, data=data, headers={"Content-Type": content_type})
    try:
        with urlopen(request, timeout=40) as response:
            assert response.status == expected, (response.status, expected)
            return json.load(response)
    except HTTPError as exc:
        body = json.load(exc)
        assert exc.code == expected, (exc.code, expected, body)
        return body


def wait(process, base):
    deadline = time.monotonic()+90
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Server exited: {process.returncode}")
        try:
            call(base, "/ping")
            return
        except Exception:
            time.sleep(.25)
    raise TimeoutError(base)


def main():
    references = json.loads((ROOT / "data/simulation/production_v1/reference_cases.json").read_text())
    original = references["cases"][0]["request"]
    sim_port = free_port()
    web_port = free_port((sim_port,))
    sim_url, web_url = f"http://127.0.0.1:{sim_port}", f"http://127.0.0.1:{web_port}"
    env = dict(os.environ, Simulation__BaseUrl=sim_url, ASPNETCORE_URLS=web_url,
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    processes = []
    started = time.perf_counter()
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as sim_log, tempfile.TemporaryFile(mode="w+", encoding="utf-8") as web_log:
        try:
            simulator = subprocess.Popen([sys.executable, "scripts/simulation/simulation_service.py", "--port", str(sim_port)],
                                         cwd=ROOT, env=env, stdout=sim_log, stderr=subprocess.STDOUT)
            processes.append(simulator); wait(simulator, sim_url)
            cold_seconds = time.perf_counter()-started
            dll = ROOT / "nba-clutch-agent/AiAgents.NbaClutchAgent.Web/bin/Debug/net9.0/AiAgents.NbaClutchAgent.Web.dll"
            web = subprocess.Popen(["dotnet", str(dll), "--urls", web_url],
                                   cwd=ROOT / "nba-clutch-agent/AiAgents.NbaClutchAgent.Web", env=env,
                                   stdout=web_log, stderr=subprocess.STDOUT)
            processes.append(web); wait(web, web_url)
            info = call(web_url, "/api/simulation/info")
            assert info["modelVersion"] == "hoopslab-single-game-v1.0.0"
            assert len(info["supportedTargetSeasons"]) == 47
            checked = []
            for case in references["cases"]:
                req = case["request"]
                response = call(web_url, "/api/simulation/game", req)
                assert response == call(web_url, "/api/simulation/game", req)
                assert response["game"]["finalTies"] == 0
                assert response["teamA"]["projectedMinutes"] == 240.
                assert response["teamB"]["projectedMinutes"] == 240.
                checked.append(case["case"])
            invalid = []
            def bad(label, mutate):
                request = copy.deepcopy(original); mutate(request)
                call(web_url, "/api/simulation/game", request, 400)
                invalid.append(label)
            bad("duplicate player", lambda r: r["teamA"]["bench"].__setitem__(0, r["teamA"]["starters"][0]))
            bad("unknown player", lambda r: r["teamA"]["starters"][0].__setitem__("playerId", "br:no-player"))
            bad("unavailable season", lambda r: r["teamA"]["starters"][0].__setitem__("season", "1970-71"))
            bad("invalid target era", lambda r: r.__setitem__("targetSeason", "1970-71"))
            bad("five without ironman", lambda r: r["teamA"].__setitem__("bench", []))
            bad("seven players", lambda r: r["teamA"].__setitem__("bench", r["teamA"]["bench"][:2]))
            bad("thirteen players", lambda r: r["teamA"]["bench"].append(dict(playerId="br:looneke01", season="2015-16")))
            bad("malformed slots", lambda r: r["teamA"]["starters"][0].__setitem__("slot", "X"))
            bad("stats override", lambda r: r["teamA"]["starters"][0].__setitem__("points", 999))
            for value in [999, 100001, True]:
                bad(f"count {value}", lambda r, v=value: r.__setitem__("simulationCount", v))
            call(web_url, "/api/simulation/game", original, 415, "text/plain")
            call(web_url, "/api/simulation/game", dict(padding="x"*65537), 413)

            # Existing routes still work with the added proxy routes registered.
            search = call(web_url, "/api/historical/players?search=curry&limit=8")
            assert any(row["playerName"] == "Stephen Curry" for row in search["items"])
            assert call(web_url, "/api/agent/status") is not None
            durations = []
            memory = [rss(simulator.pid)]
            for _ in range(30):
                start = time.perf_counter(); call(web_url, "/api/simulation/game", original)
                durations.append((time.perf_counter()-start)*1000)
                memory.append(rss(simulator.pid))
            maximum = dict(original, simulationCount=100000)
            start = time.perf_counter(); call(web_url, "/api/simulation/game", maximum)
            max_ms = (time.perf_counter()-start)*1000
            memory.append(rss(simulator.pid))
            results = dict(status="pass", referenceCases=checked, invalidRequestsRejected=invalid,
                draws=10000, repetitions=30, medianRequestMs=float(np.median(durations)), p95RequestMs=float(np.quantile(durations, .95)),
                maxDrawRequestMs=max_ms, pythonColdStartupSeconds=cold_seconds, memoryUnit="MiB working set",
                pythonMemoryBefore=memory[0], pythonMemoryAfter30=memory[-2], pythonMemoryAfterMax=memory[-1],
                pythonMemoryMin=min(v for v in memory if v is not None) if any(v is not None for v in memory) else None,
                pythonMemoryMax=max(v for v in memory if v is not None) if any(v is not None for v in memory) else None,
                dataLoadBehavior="Once at service startup; no raw cache or joblib access; per-request parse guarded by unit test")
            simulator.terminate(); simulator.wait(timeout=15)
            call(web_url, "/api/simulation/info", expected=503)
            output = ROOT / "data/simulation/production_v1/verification.json"
            output.write_text(json.dumps(results, indent=2)+"\n", encoding="utf-8")
            print(json.dumps(results, indent=2))
        except Exception:
            for log in [sim_log, web_log]:
                log.seek(0); print(log.read(), file=sys.stderr)
            raise
        finally:
            for process in reversed(processes):
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        process.kill(); process.wait(timeout=10)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
