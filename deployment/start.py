"""PID 1: validate frozen artifacts, supervise all three required child processes."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
stop = threading.Event()
children = []


def healthy(port):
    try:
        with urlopen(f"http://127.0.0.1:{port}/ping", timeout=2) as response:
            return response.status == 200 and json.load(response).get("ok") is True
    except Exception:
        return False


def check_children():
    for child in children:
        if child.poll() is not None:
            raise RuntimeError(f"Required process exited with code {child.returncode}")


def start(command, cwd, env):
    child = subprocess.Popen(command, cwd=cwd, env=env, start_new_session=True)
    children.append(child)
    return child


def shutdown():
    for child in children:
        if child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 15
    for child in children:
        try:
            child.wait(timeout=max(.1, deadline-time.monotonic()))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait()


def main():
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    port = int(os.environ.get("PORT", "8080"))
    if not 0 < port <= 65535 or port in (8011, 8012):
        raise ValueError("Invalid public PORT")
    for key, value in {"ProductionInference__BaseUrl": "http://127.0.0.1:8011",
                       "Simulation__BaseUrl": "http://127.0.0.1:8012"}.items():
        if os.environ.get(key, value) != value:
            raise ValueError(f"{key} must remain loopback-only")
    manifest = json.loads((ROOT/"deployment/runtime-artifacts.json").read_text())
    for item in manifest["files"]:
        path = ROOT/item["path"]
        if "text_sha256" in item:
            digest = hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            expected = item["text_sha256"]
        else:
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            expected = item["sha256"]
        if digest != expected:
            raise RuntimeError(f"Runtime artifact checksum mismatch: {item['path']}")
    env = dict(os.environ, APP_ROOT=str(ROOT), ASPNETCORE_URLS=f"http://0.0.0.0:{port}",
               ASPNETCORE_ENVIRONMENT=os.environ.get("ASPNETCORE_ENVIRONMENT", "Production"),
               PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    start([sys.executable, "scripts/inference/inference_service.py", "--host", "127.0.0.1", "--port", "8011"], ROOT, env)
    start([sys.executable, "scripts/simulation/simulation_service.py", "--port", "8012"], ROOT, env)
    deadline = time.monotonic()+180
    while not stop.is_set():
        check_children()
        if healthy(8011) and healthy(8012):
            break
        if time.monotonic() > deadline:
            raise TimeoutError("Internal services did not become ready")
        stop.wait(.25)
    if stop.is_set():
        return
    start(["dotnet", "AiAgents.NbaClutchAgent.Web.dll"], ROOT/"web", env)
    while not stop.wait(.25):
        check_children()


if __name__ == "__main__":
    code = 0
    try:
        main()
    except Exception as exc:
        print(f"Startup/supervision failure: {exc}", file=sys.stderr, flush=True)
        code = 1
    finally:
        shutdown()
    sys.exit(code)
