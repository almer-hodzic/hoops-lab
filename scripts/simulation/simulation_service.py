"""Loopback-only production simulation service; no Shot Quality dependency."""
from __future__ import annotations

import argparse
import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from threadpoolctl import threadpool_limits

from production_simulator import ProductionSimulator, SimulationError

MAX_BODY = 65536


class SimulationServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, engine):
        super().__init__(address, Handler)
        self.engine = engine
        self.capacity = threading.BoundedSemaphore(4)


class Handler(BaseHTTPRequestHandler):
    def setup(self):
        super().setup()
        self.connection.settimeout(10)

    def reply(self, status, payload):
        body = json.dumps(payload, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/simulation/info":
            self.reply(200, self.server.engine.info())
        elif self.path == "/ping":
            self.reply(200, dict(ok=True, modelVersion=self.server.engine.config["modelVersion"]))
        else:
            self.reply(404, dict(error="Not found"))

    def do_POST(self):
        if self.path != "/api/simulation/game":
            self.reply(404, dict(error="Not found"))
            return
        if not self.server.capacity.acquire(blocking=False):
            self.reply(503, dict(error="Simulator busy; retry later"))
            return
        try:
            if self.headers.get_content_type() != "application/json":
                raise SimulationError("Content-Type must be application/json", 415)
            if self.headers.get("Transfer-Encoding"):
                raise SimulationError("Chunked requests are not supported", 400)
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError as exc:
                raise SimulationError("Invalid Content-Length") from exc
            if not 0 < length <= MAX_BODY:
                raise SimulationError("Request body must contain 1-65536 bytes", 413)
            raw = self.rfile.read(length)
            if len(raw) != length:
                raise SimulationError("Incomplete request body")
            try:
                payload = json.loads(raw, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Non-finite JSON")))
            except (ValueError, UnicodeDecodeError) as exc:
                raise SimulationError("Invalid JSON") from exc
            self.reply(200, self.server.engine.simulate(payload))
        except SimulationError as exc:
            self.reply(exc.status, dict(error=str(exc)))
        except Exception:
            logging.exception("Simulation request failed")
            self.reply(500, dict(error="Simulation failed; inspect service logs"))
        finally:
            self.server.capacity.release()

    def log_message(self, *_):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8012)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        server = SimulationServer(("127.0.0.1", args.port), ProductionSimulator())
        print(json.dumps(dict(ok=True, url=f"http://127.0.0.1:{args.port}", service="hoopslab-simulation")), flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
