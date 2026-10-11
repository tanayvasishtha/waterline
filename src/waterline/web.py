"""The demo web app: a read-only window into what the agent did.

Serves the dashboard plus JSON views of readings, summaries, traces, flags and
evidence images that the poll/eval pipelines already wrote to the data dir.
"""

import csv
import json
from functools import cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np

from waterline import sites as site_lib

DATA = Path("data")
RESULTS = Path("results")
STATIC = Path(__file__).parent / "static"

# Sites shown in the dashboard. Other site configs stay on disk but are not demoed.
DEMO_SITES = ("hancock", "in_overland", "oh_wetland")


@cache
def _media_index() -> dict[str, Path]:
    """Lazy filename -> path index over evidence frames, built on first request."""
    index: dict[str, Path] = {}
    for base in (DATA / "evidence", DATA / "cache"):
        if base.exists():
            for path in base.rglob("*.jpg"):
                index.setdefault(path.name, path)
    return index


def _eval_csv(site: str) -> Path | None:
    """The site's held-out eval CSV: eval_{site}_{start}_{end}.csv."""
    for path in sorted(RESULTS.glob("eval_*.csv")):
        parts = path.stem.rsplit("_", 2)
        if len(parts) == 3 and parts[0] == f"eval_{site}":
            return path
    return None


def _status(r: dict[str, str]) -> str:
    if r.get("reason") == "below_plate":
        return "below_range"
    if r.get("gate_pass") != "True":
        return "abstained"
    return "read" if r.get("stage_ft") else "abstained"


def _readings(site: str) -> list[dict[str, Any]]:
    """Series rows from the site's held-out eval CSV only."""
    rows = []
    path = _eval_csv(site)
    if path is not None:
        with path.open() as fh:
            for r in csv.DictReader(fh):
                if r.get("gage_ft"):
                    rows.append(
                        {
                            "time": r["time"],
                            "gage_ft": float(r["gage_ft"]),
                            "stage_ft": float(r["stage_ft"]) if r["stage_ft"] else None,
                            "status": _status(r),
                            "file": r["file"],
                            "is_night": r.get("is_night") == "True",
                        }
                    )
    rows.sort(key=lambda r: str(r["time"]))
    return rows


def _summary(site: str) -> dict[str, Any]:
    """Held-out event metrics, computed with the same slice semantics as eval.py:
    rows failing the triage gate count as abstained and are excluded from errors.
    """
    out: dict[str, Any] = {
        "site": site,
        "window": None,
        "n_frames": 0,
        "n_reads": 0,
        "n_below_range": 0,
        "n_abstained": 0,
        "mae_ft": None,
        "median_ft": None,
        "p90_ft": None,
    }
    path = _eval_csv(site)
    if path is None:
        return out
    parts = path.stem.rsplit("_", 2)
    out["window"] = [parts[1], parts[2]]
    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    out["n_frames"] = len(rows)
    out["n_below_range"] = sum(1 for r in rows if _status(r) == "below_range")
    out["n_abstained"] = sum(1 for r in rows if _status(r) == "abstained")
    gated = [r for r in rows if r.get("gate_pass") == "True"]
    errors = np.array(
        [
            float(r["stage_ft"]) - float(r["gage_ft"])
            for r in gated
            if r["stage_ft"] and r["gage_ft"]
        ]
    )
    if errors.size:
        ae = np.abs(errors)
        out["n_reads"] = int(ae.size)
        out["mae_ft"] = float(ae.mean())
        out["median_ft"] = float(np.median(ae))
        out["p90_ft"] = float(np.percentile(ae, 90))
    return out


def _flags(site: str | None) -> list[dict[str, Any]]:
    out = []
    events = DATA / "events"
    if events.exists():
        for path in events.rglob("flag.json"):
            d = json.loads(path.read_text())
            if site is None or d.get("site_id") == site:
                out.append(d)
    return sorted(out, key=lambda f: str(f["opened_at"]), reverse=True)


def _traces(site: str | None) -> list[dict[str, Any]]:
    out = []
    tdir = DATA / "traces"
    if tdir.exists():
        pattern = f"{site}-*.json" if site else "*.json"
        for path in sorted(tdir.glob(pattern))[-50:]:
            out.append(json.loads(path.read_text()))
    return out


def api(path: str, query: dict[str, list[str]]) -> tuple[int, str, bytes]:
    """Route an API request to a JSON response."""
    site = (query.get("site") or [""])[0] or None
    body: Any
    if path == "/api/sites":
        body = [s for s in DEMO_SITES if s in site_lib.list_sites()]
    elif path == "/api/flags":
        body = _flags(site)
    elif path == "/api/traces":
        body = _traces(site)
    elif path in ("/api/site", "/api/summary", "/api/readings"):
        if site is None or site not in site_lib.list_sites():
            return 404, "application/json", b'{"error":"unknown site"}'
        if path == "/api/site":
            cfg = site_lib.load_site(site)
            return 200, "application/json", cfg.model_dump_json().encode()
        body = _summary(site) if path == "/api/summary" else _readings(site)
    else:
        return 404, "application/json", b'{"error":"unknown endpoint"}'
    return 200, "application/json", json.dumps(body).encode()


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
            name = Path(parsed.path[len("/media/") :]).name
            cand = _media_index().get(name)
            if cand is not None and cand.is_file():
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
