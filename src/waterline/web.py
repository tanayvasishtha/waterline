"""The demo web app: a read-only window into what the agent did.

Stdlib only: judges can run `waterline web` with zero extra installs. Serves the
dashboard plus JSON views of readings, traces, flags and evidence images that
the poll/eval pipelines already wrote to the data dir.
"""

import csv
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from waterline import sites as site_lib

DATA = Path("data")
RESULTS = Path("results")
STATIC = Path(__file__).parent / "static"


def _readings(site: str) -> list[dict]:
    """Series rows: eval CSV if present, else the poll log."""
    rows = []
    for csv_path in sorted(RESULTS.glob(f"*{site}*.csv")):
        with csv_path.open() as fh:
            for r in csv.DictReader(fh):
                if r.get("gage_ft"):
                    rows.append(
                        {
                            "time": r["time"],
                            "gage_ft": float(r["gage_ft"]),
                            "stage_ft": float(r["stage_ft"]) if r["stage_ft"] else None,
                            "status": r.get("agent_status") or "read",
                            "file": r["file"],
                            "is_night": r.get("is_night") == "True",
                        }
                    )
    if not rows:
        log = DATA / "readings.jsonl"
        if log.exists():
            for line in log.read_text().splitlines():
                d = json.loads(line)
                if d.get("site_id") == site:
                    rows.append(
                        {
                            "time": d["time"],
                            "gage_ft": None,
                            "stage_ft": d.get("stage_ft"),
                            "status": d.get("status"),
                            "file": "",
                            "is_night": False,
                        }
                    )
    rows.sort(key=lambda r: r["time"])
    return rows


def _flags(site: str | None) -> list[dict]:
    out = []
    events = DATA / "events"
    if events.exists():
        for path in events.rglob("flag.json"):
            d = json.loads(path.read_text())
            if site is None or d.get("site_id") == site:
                out.append(d)
    return sorted(out, key=lambda f: f["opened_at"], reverse=True)


def _traces(site: str | None) -> list[dict]:
    out = []
    tdir = DATA / "traces"
    if tdir.exists():
        for path in sorted(tdir.glob("*.json"))[-50:]:
            d = json.loads(path.read_text())
            if site is None or d.get("site_id") == site:
                out.append(d)
    return out


def api(path: str, query: dict) -> tuple[int, str, bytes]:
    """Route an API request to a JSON response."""
    site = (query.get("site") or [None])[0]
    if path == "/api/sites":
        return 200, "application/json", json.dumps(site_lib.list_sites()).encode()
    if path == "/api/readings":
        return 200, "application/json", json.dumps(_readings(site or "")).encode()
    if path == "/api/flags":
        return 200, "application/json", json.dumps(_flags(site)).encode()
    if path == "/api/traces":
        return 200, "application/json", json.dumps(_traces(site)).encode()
    if path == "/api/site":
        cfg = site_lib.load_site(site or "")
        return 200, "application/json", cfg.model_dump_json().encode()
    return 404, "application/json", b'{"error":"unknown endpoint"}'


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: object) -> None:
        pass

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/"):
            status, ctype, body = api(parsed.path, parse_qs(parsed.query))
            self._reply(status, ctype, body)
            return
        if parsed.path.startswith("/media/"):
            rel = Path(parsed.path[len("/media/") :]).name
            for base in (DATA / "evidence", DATA / "cache"):
                for cand in base.rglob(rel):
                    if cand.is_file():
                        self._reply(200, "image/jpeg", cand.read_bytes())
                        return
            self._reply(404, "text/plain", b"not found")
            return
        name = "index.html" if parsed.path in ("/", "") else Path(parsed.path).name
        file = STATIC / name
        if file.is_file():
            ctype = {"html": "text/html", "css": "text/css", "js": "text/javascript"}.get(
                file.suffix[1:], "application/octet-stream"
            )
            self._reply(200, ctype, file.read_bytes())
            return
        self._reply(404, "text/plain", b"not found")

    def _reply(self, status: int, ctype: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def serve(port: int = 8000) -> None:
    print(f"waterline dashboard: http://localhost:{port}")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
