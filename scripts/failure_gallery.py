"""Pick failure and recovery cases from the final run by fixed rules.

Seven categories, each filled from the final agent CSVs plus the replayed
traces under results/final_traces/. "first 2" always means the two earliest
cases by row timestamp; "largest agent error" means the two biggest absolute
agent errors among accepted reads. Nothing is picked by eye.

For each case it writes results/gallery/{category}_{n}.jpg: the frame scaled
to 960 px wide, the detected waterline drawn by the pipeline's own overlay,
and the rectified plate strip beside it when the frame registered. Stage and
gage numbers go on a caption bar under the image. results/gallery/index.json
carries the same numbers as text for the dashboard.

Only cached frames are used; a missing file is logged, and the next candidate
takes its place.

Run from the repo root: uv run python scripts/failure_gallery.py
"""

import csv
import json
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from waterline import eval as ev
from waterline import sites
from waterline.pipeline import Perception, overlay, read_frame
from waterline.settings import Settings

RESULTS = Path("results")
CACHE = Path("data/cache")
OUT = RESULTS / "gallery"

FRAME_W = 960
STRIP_W = 240

_ctx: dict[str, Perception] = {}


def _perception(site: str) -> Perception:
    if site not in _ctx:
        _ctx[site] = ev.make_perception(sites.load_site(site), Settings())
    return _ctx[site]


def _file_stamp(name: str) -> str:
    stamp = name.rsplit("___", 1)[1].rsplit(".", 1)[0]
    day, clock = stamp.split("T")
    parts = clock.rstrip("Z").split("-")
    return f"{day.replace('-', '')}T{parts[0]}{parts[1]}Z"


def _trace_name(site: str, row: dict[str, str]) -> str:
    src = row["agent_frame_file"] or row["file"]
    return f"{site}-{_file_stamp(src)}.json"


def _rows() -> list[dict[str, Any]]:
    """Every final agent row across both sites, with its site attached."""
    out: list[dict[str, Any]] = []
    for site in ("hancock", "in_overland"):
        for path in sorted(RESULTS.glob(f"final_eval_agent_{site}_*.csv")):
            with path.open() as fh:
                for r in csv.DictReader(fh):
                    r["site"] = site
                    out.append(r)
    out.sort(key=lambda r: str(r["time"]))
    return out


def _traces() -> dict[str, dict[str, Any]]:
    out = {}
    for site in ("hancock", "in_overland"):
        tdir = RESULTS / "final_traces" / site
        if not tdir.exists():
            continue
        for path in sorted(tdir.glob("*.json")):
            out[f"{site}/{path.name}"] = json.loads(path.read_text())
    return out


def _decisions(trace: dict[str, Any] | None) -> list[str]:
    if not trace:
        return []
    return [str(s.get("decision") or "") for s in trace.get("steps", [])]


def _trace_for(row: dict[str, Any], traces: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    return traces.get(f"{row['site']}/{_trace_name(str(row['site']), row)}")


def _err(row: dict[str, Any]) -> float | None:
    if row["agent_status"] != "accepted" or not row["agent_stage_ft"]:
        return None
    gage = row["agent_gage_ft"] or row["gage_ft"]
    if not gage:
        return None
    return abs(float(str(row["agent_stage_ft"])) - float(str(gage)))


Rule = Callable[[dict[str, Any], dict[str, Any] | None], bool]

RULES: list[tuple[str, str, Rule]] = [
    (
        "registration_failed",
        "the frame never registered against a reference",
        lambda row, tr: any("register" in d or "registration failed" in d for d in _decisions(tr)),
    ),
    (
        "night_abstain",
        "night or IR frame the agent could not read",
        lambda row, tr: (
            any(d.startswith("night") for d in _decisions(tr))
            or "night" in str((tr or {}).get("reason") or "")
        ),
    ),
    (
        "below_plate",
        "water below the staff plate",
        lambda row, tr: row["agent_status"] == "below_range",
    ),
    (
        "repair_rejected",
        "a repair step whose result was discarded",
        lambda row, tr: any(d.startswith("repair rejected") for d in _decisions(tr)),
    ),
    (
        "pending",
        "sent to a person",
        lambda row, tr: row["agent_status"] == "pending",
    ),
    (
        "substitution",
        "reading taken from a daylight neighbour",
        lambda row, tr: bool(row["agent_frame_file"]) and row["agent_frame_file"] != row["file"],
    ),
]


def _find_cached(name: str) -> Path | None:
    for cam_dir in CACHE.iterdir():
        cand = cam_dir / "overlay" / name
        if cand.exists():
            return cand
    return None


def _f2(v: float | None) -> str:
    return f"{v:.2f}" if v is not None else "-"


def _parse_time(filename: str) -> datetime:
    stamp = filename.rsplit("___", 1)[1].rsplit(".", 1)[0]
    return datetime.strptime(stamp, "%Y-%m-%dT%H-%M-%SZ")


def _compose(site: str, frame_path: Path, caption: str) -> np.ndarray | None:
    image = cv2.imread(str(frame_path))
    if image is None:
        return None
    result = read_frame(_perception(site), image, _parse_time(frame_path.name), seed=0)
    base = overlay(image, result)
    scale = FRAME_W / base.shape[1]
    base = cv2.resize(base, (FRAME_W, int(base.shape[0] * scale)))
    parts = [base]
    if result.strip is not None:
        strip = result.strip.image.copy()
        if result.detection is not None and result.detection.row_px is not None:
            row = int(result.detection.row_px)
            cv2.line(strip, (0, row), (strip.shape[1], row), (0, 255, 0), 2)
        sscale = base.shape[0] / strip.shape[0]
        sw = min(int(strip.shape[1] * sscale), STRIP_W)
        strip = cv2.resize(strip, (sw, base.shape[0]))
        parts.append(strip)
    out = np.hstack(parts)
    bar = np.full((30, out.shape[1], 3), 245, np.uint8)
    cv2.putText(bar, caption, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1)
    return np.vstack([out, bar])


def _entry(cat: str, n: int, r: dict[str, Any], img_name: str) -> dict[str, Any]:
    return {
        "category": cat,
        "site": r["site"],
        "image": f"{cat}_{n}.jpg",
        "file": img_name,
        "time": r["time"],
        "gage_ft": float(r["gage_ft"]) if r["gage_ft"] else None,
        "baseline_ft": float(r["stage_ft"]) if r["stage_ft"] else None,
        "agent_ft": float(r["agent_stage_ft"]) if r["agent_stage_ft"] else None,
        "agent_gage_ft": float(r["agent_gage_ft"]) if r["agent_gage_ft"] else None,
        "status": r["agent_status"],
    }


def main() -> int:
    rows = _rows()
    traces = _traces()
    OUT.mkdir(parents=True, exist_ok=True)
    index: list[dict[str, Any]] = []

    # Candidates per category in selection order; first two with a cached
    # frame are rendered, the rest are backfill if an earlier frame is missing.
    pending: list[tuple[str, dict[str, Any]]] = []
    for cat, _desc, rule in RULES:
        for r in rows:
            if rule(r, _trace_for(r, traces)):
                pending.append((cat, r))
    scored = [(e, r) for r in rows if (e := _err(r)) is not None]
    scored.sort(key=lambda t: (-t[0], str(t[1]["time"])))
    for _e, r in scored[:6]:
        pending.append(("largest_agent_error", r))

    per_cat: dict[str, int] = {}
    counts: dict[str, int] = {}
    for cat, r in pending:
        n = per_cat.get(cat, 0)
        if n >= 2:
            continue
        img_name = r["file"]
        if cat in ("substitution", "largest_agent_error") and r["agent_frame_file"]:
            img_name = r["agent_frame_file"]
        path = _find_cached(str(img_name))
        if path is None:
            print(f"MISSING {cat} {img_name}")
            continue
        tr = _trace_for(r, traces)
        reason = (tr or {}).get("reason") or (r["reason"] if r["reason"] else None)
        entry = _entry(cat, n + 1, r, str(img_name))
        entry["reason"] = reason
        entry["trace_id"] = (tr or {}).get("trace_id")
        cap = (
            f"{str(r['time'])[:16]}  gage {_f2(entry['gage_ft'])} ft  "
            f"baseline {_f2(entry['baseline_ft'])}  agent {_f2(entry['agent_ft'])}  "
            f"{r['agent_status']}"
        )
        img = _compose(str(r["site"]), path, cap)
        if img is None:
            print(f"UNREADABLE {cat} {img_name}")
            continue
        per_cat[cat] = n + 1
        cv2.imwrite(str(OUT / str(entry["image"])), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        index.append(entry)
        counts[cat] = counts.get(cat, 0) + 1
        print(f"WROTE {entry['image']} <- {img_name}")
    (OUT / "index.json").write_text(json.dumps(index, indent=2))
    print("counts:", json.dumps(counts, sort_keys=True))
    print(f"{len(index)} gallery entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
