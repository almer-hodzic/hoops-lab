"""Local functional/lifecycle check; creates only the named V1 review container."""
import http.client
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
NAME = "hoopslab-v1-review"
PORT = 18080
BASE = f"http://127.0.0.1:{PORT}"
results = {"url": BASE, "image": "hoopslab-v1:local", "checks": {}}


def docker(*args, **kwargs):
    return subprocess.check_output(["docker", *args], text=True, encoding="utf-8", **kwargs).strip()


def call(path, payload=None, status=200, method=None, raw=None):
    connection = http.client.HTTPConnection("127.0.0.1", PORT, timeout=40)
    data = raw if raw is not None else json.dumps(payload).encode() if payload is not None else None
    try:
        connection.request(method or ("POST" if data is not None else "GET"), path, data,
                           {"Content-Type": "application/json"} if data is not None else {})
        response = connection.getresponse(); body = response.read()
        assert response.status == status, (path, response.status, status, body[:300])
        return json.loads(body) if "application/json" in response.getheader("Content-Type", "") and body else body
    finally:
        connection.close()


def ready():
    deadline = time.monotonic()+210
    while time.monotonic() < deadline:
        try:
            assert call("/ready")["ready"] is True
            return
        except (OSError, AssertionError, http.client.HTTPException):
            if docker("inspect", "--format", "{{.State.Running}}", NAME) != "true":
                raise RuntimeError(docker("logs", NAME))
            time.sleep(.25)
    raise TimeoutError(docker("logs", NAME))


def memory():
    return int(docker("exec", NAME, "python", "-c", "print(open('/sys/fs/cgroup/memory.current').read())"))/1024**2


def measured(path, payload):
    # Sample cgroup total memory during a single request (20 ms intervals, 3 s).
    sampler = subprocess.Popen(["docker", "exec", NAME, "python", "-c",
        "import time,json; from pathlib import Path; values=[]; end=time.monotonic()+3; "
        "exec(\"while time.monotonic()<end:\\n values.append(int(Path('/sys/fs/cgroup/memory.current').read_text())); time.sleep(.02)\"); print(json.dumps(values))"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    time.sleep(.25)
    started = time.perf_counter(); result = call(path, payload); elapsed = time.perf_counter()-started
    stdout, stderr = sampler.communicate(timeout=15)
    assert sampler.returncode == 0, stderr
    samples = json.loads(stdout)
    return result, {"peakMiB": round(max(samples)/1024**2, 2), "afterMiB": round(memory(), 2), "requestSeconds": round(elapsed, 3)}


def main():
    # Never remove/reuse an unrelated existing container.
    existing = docker("ps", "-a", "--filter", f"name=^/{NAME}$", "--format", "{{.Names}}")
    assert not existing, f"Container {NAME} already exists; inspect it before a rerun"
    started = time.perf_counter()
    docker("run", "-d", "--name", NAME, "--memory", "2g", "--cpus", "1", "-p", f"127.0.0.1:{PORT}:8080", "-e", "PORT=8080", "hoopslab-v1:local")
    ready(); results["coldStartupSeconds"] = round(time.perf_counter()-started, 3)
    assert call("/ping")["ok"]
    assert b"HoopsLab" in call("/")
    for asset in ["/app.js", "/app.css", "/teamBuilder.js"]:
        assert call(asset)
    results["checks"]["homepageAndStaticAssets"] = "pass"
    search = call("/api/historical/players?search=curry&limit=8")
    curry = next(p for p in search["items"] if p["playerName"] == "Stephen Curry")
    jordan = next(p for p in call("/api/historical/players?search=jordan&limit=8")["items"] if p["playerName"] == "Michael Jordan")
    assert call(f"/api/historical/players/{curry['playerId']}")["seasons"]
    assert call(f"/api/historical/players/{curry['playerId']}/seasons/2015-16")["identity"]["team"] == "GSW"
    compare = call(f"/api/historical/compare?playerA={jordan['playerId']}&seasonA=1990-91&playerB={curry['playerId']}&seasonB=2015-16")
    assert compare["playerA"]["identity"]["playerName"] == "Michael Jordan"
    assert compare["playerB"]["identity"]["playerName"] == "Stephen Curry"
    call("/api/historical/players/br:not-a-player", status=404)
    results["checks"]["historicalExplorerAndCompare"] = "pass"
    players = call("/api/players?q=Stephen&limit=5")["items"]
    player = next(p for p in players if "Curry" in p["player_name"])
    assert call(f"/api/players/{player['player_id']}/profile")
    assert call("/api/model")
    analysis = call("/api/analysis/decisions?page=1&pageSize=3")
    assert len(analysis["items"]) == 3
    results["checks"]["shotLabProfilesAndAnalysis"] = "pass"
    reference = json.loads((ROOT/"data/simulation/production_v1/reference_cases.json").read_text())["cases"][0]["request"]
    shot = dict(playerId=str(player["player_id"]), locX=80, locY=250, shotDistance=26, shotValue=3,
                shotType="3PT Field Goal", actionType="Jump Shot", period=4, secondsRemaining=240,
                scoreDiff=-3, isHome=True, isOvertime=False, gameState="trailing")
    assert call("/api/decision/analyze", shot)["prediction"]["makeProbability"] >= 0
    assert call("/api/simulation/info")["modelVersion"] == "hoopslab-single-game-v1.0.0"
    assert call("/api/simulation/game", reference)["game"]["finalTies"] == 0
    time.sleep(1)
    results["idleMiB"] = round(memory(), 2)
    prediction, results["inference"] = measured("/api/decision/analyze", shot)
    simulation, results["simulation"] = measured("/api/simulation/game", reference)
    assert simulation["teamA"]["projectedMinutes"] == 240
    results["checks"]["singleGameSimulation"] = "pass"
    for path in ["/api/run", "/api/agent/start", "/api/agent/stop", "/api/agent/status", "/api/baseline"]:
        for method in ["GET", "POST"]: call(path, status=404, method=method)
    results["checks"]["legacyBlocked"] = "pass"
    call("/api/decision/analyze", raw=b"{" ,status=400)
    call("/api/decision/analyze", raw=b"x"*65537, status=413)
    error = call("/api/players?limit=not-a-number", status=500)
    assert "not-a-number" not in json.dumps(error) and "error" in error
    results["checks"]["bodyLimitAndErrorSanitizing"] = "pass"
    # Low-cost invalid payloads verify the rate-limit boundary; no simulation work.
    statuses = []
    for _ in range(21):
        conn = http.client.HTTPConnection("127.0.0.1", PORT, timeout=10)
        conn.request("POST", "/api/simulation/game", b"{}", {"Content-Type":"application/json"})
        response=conn.getresponse(); statuses.append(response.status); response.read(); conn.close()
        if statuses[-1] == 429: break
    assert 429 in statuses
    results["checks"]["expensiveRateLimit"] = "pass"
    network = json.loads(docker("inspect", NAME))[0]
    assert set(network["HostConfig"]["PortBindings"]) == {"8080/tcp"}
    private = docker("exec", NAME, "python", "-c",
        "from pathlib import Path; rows=Path('/proc/net/tcp').read_text().splitlines()[1:]; "
        "ports={int(r.split()[1].split(':')[1],16):r.split()[1].split(':')[0] for r in rows if r.split()[3]=='0A'}; "
        "assert ports[8011]==ports[8012]=='0100007F'; print(ports)")
    results["checks"]["privateLoopbackBindings"] = private
    # Required-child failure must fail PID 1 and the whole container.
    docker("exec", NAME, "python", "-c",
        "import os,signal; from pathlib import Path; "
        "pids=Path('/proc/1/task/1/children').read_text().split(); "
        "pid=next(int(p) for p in pids if b'simulation_service.py' in Path('/proc/'+p+'/cmdline').read_bytes()); os.kill(pid,signal.SIGTERM)")
    deadline=time.monotonic()+20
    while docker("inspect", "--format", "{{.State.Running}}", NAME)=="true" and time.monotonic()<deadline: time.sleep(.25)
    assert docker("inspect", "--format", "{{.State.ExitCode}}", NAME)=="1"
    results["checks"]["childFailureFailsContainer"] = "pass"
    docker("start",NAME); ready()
    docker("stop", "--time", "20", NAME)
    assert docker("inspect", "--format", "{{.State.ExitCode}}", NAME)=="0"
    results["checks"]["gracefulTermination"] = "pass"
    started=time.perf_counter(); docker("start",NAME); ready()
    results["restartStartupSeconds"]=round(time.perf_counter()-started,3)
    results["checks"]["cleanRestart"]="pass"
    results["containerStillRunning"]=docker("inspect","--format","{{.State.Running}}",NAME)=="true"
    results["measurementNotes"]="Linux cgroup memory.current, including charged cache and sampler overhead; one CPU / 2 GiB container; one warm inference and one 10,000-draw simulation; 20 ms sampling for 3 s. Cold startup excludes image download/build."
    results["visualQA"]="Separate browser review required: this script performs HTTP/API and lifecycle checks; Team Builder browser/localStorage interactions are not tested here."
    (ROOT/"deployment/validation-results.json").write_text(json.dumps(results,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(results,indent=2))


if __name__ == "__main__":
    try: main()
    except Exception:
        print(json.dumps(results,indent=2))
        raise
