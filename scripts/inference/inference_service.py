#!/usr/bin/env python3
"""Small local HTTP inference service using only the Python standard library."""

from __future__ import annotations

import argparse
import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from inference_core import InferenceError, ProductionInferenceEngine


class Handler(BaseHTTPRequestHandler):
    engine: ProductionInferenceEngine

    def setup(self):
        super().setup()
        self.connection.settimeout(10)

    def _json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        if self.headers.get("Transfer-Encoding"):
            raise InferenceError(400, "Chunked requests are not supported")
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise InferenceError(400, "Invalid Content-Length") from exc
        if not 0 < length <= 65536:
            raise InferenceError(413, "Request body must contain 1-65536 bytes")
        raw = self.rfile.read(length) if length else b"{}"
        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise InferenceError(400, f"Invalid JSON: {exc}") from exc

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        query = {k: v[-1] for k, v in parse_qs(parsed.query).items()}
        try:
            if parsed.path == "/ping":
                self._json(200, {"ok": True})
            elif parsed.path == "/api/model":
                self._json(200, self.engine.model_metadata())
            elif parsed.path == "/api/players":
                self._json(200, self.engine.players(query.get("q", ""), int(query.get("limit", 25))))
            elif parsed.path.startswith("/api/players/") and parsed.path.endswith("/profile"):
                player_id = parsed.path.split("/")[3]
                self._json(200, self.engine.player_profile(player_id))
            elif parsed.path == "/api/analysis/decisions":
                self._json(200, self.engine.historical_decisions(query))
            else:
                self._json(404, {"error": "Not found"})
        except InferenceError as exc:
            self._json(exc.status_code, {"error": exc.message})
        except Exception as exc:  # pragma: no cover - defensive HTTP boundary
            logging.exception("Inference request failed")
            self._json(500, {"error": "Inference unavailable"})

    def do_POST(self) -> None:  # noqa: N802
        try:
            if self.path == "/api/decision/analyze":
                self._json(200, self.engine.analyze(self._read_json()))
                return
            if self.path != "/api/decision/analyze":
                self._json(404, {"error": "Not found"})
                return
        except InferenceError as exc:
            self._json(exc.status_code, {"error": exc.message})
        except Exception as exc:  # pragma: no cover - defensive HTTP boundary
            logging.exception("Inference request failed")
            self._json(500, {"error": "Inference unavailable"})

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8011)
    args = parser.parse_args()
    Handler.engine = ProductionInferenceEngine()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(json.dumps({"ok": True, "url": f"http://{args.host}:{args.port}", "service": "nba-clutch-inference"}), flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
