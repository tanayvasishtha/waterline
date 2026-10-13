"""Deterministically replay the frozen agent over the final test windows.

Reruns run_frame per frame in each window's final order with the same frozen
code path the eval used, writes every Trace to results/final_traces/{site}/,
and compares each replayed reading (status, stage to 1e-9, frame_file) against
the committed final agent CSV. Zero mismatches is required to publish: a
difference would mean the agent's output is not reproducible from the code.

Each window runs in a fresh subprocess so per-window memory is returned. A
window whose traces are all on disk is compared and skipped without camera
work; a partially traced window replays from its first frame because the
plausibility history depends on earlier readings. Frames in
results/final_missing_frames.txt are skipped, matching the resumed final run.

Scope: all hancock windows, plus in_overland windows starting in 2026 to bound
runtime.

Run from the repo root: uv run python scripts/final_traces.py
"""

import csv
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from waterline import eval as ev
from waterline import sites
from waterline.agent.history import CameraHistory
from waterline.agent.loop import run_frame
from waterline.agent.tools import FrameNavigator
from waterline.settings import Settings
from waterline.usgs import nims
from waterline.usgs.http import UsgsError, make_client

RESULTS = Path("results")
CACHE = Path("data/cache")
TRACE_DIR = RESULTS / "final_traces"

# in_overland: replay only windows starting in this year
IO_YEAR = "2026"
SITES = ("hancock", "in_overland")


def _patient(fn: Any) -> Any:
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        for i in range(200):
            try:
                return fn(*args, **kwargs)
            except UsgsError as e:
                m = str(e)
                if (
                    "rate limited" not in m
                    and "ConnectError" not in m
                    and "failed after retries" not in m
                ):
                    raise
                print(f"    retry 65s (#{i + 1})", flush=True)
                time.sleep(65)
            except Exception as e:
                if "Connect" not in type(e).__name__ and "Timeout" not in type(e).__name__:
                    raise
                print(f"    conn retry 65s (#{i + 1})", flush=True)
                time.sleep(65)
        raise UsgsError("limit stuck")

    return wrapped


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as fh:
        return list(csv.DictReader(fh))


def _file_stamp(name: str) -> str:
    """...___2026-09-01T04-00-09Z.jpg -> 20260901T0400Z."""
    stamp = name.rsplit("___", 1)[1].rsplit(".", 1)[0]
    day, clock = stamp.split("T")
    parts = clock.rstrip("Z").split("-")
    return f"{day.replace('-', '')}T{parts[0]}{parts[1]}Z"


def _trace_name(site: str, row: dict[str, str]) -> str:
    """Trace file a CSV row produces: named for the frame the reading came
    from (the substitute, when the agent swapped frames)."""
    src = row["agent_frame_file"] or row["file"]
    return f"{site}-{_file_stamp(src)}.json"


def _label(start: str, end: str) -> str:
    fmt = "%Y-%m-%d"
    s = datetime.fromisoformat(start.replace("Z", "+00:00")).strftime(fmt)
    e = datetime.fromisoformat(end.replace("Z", "+00:00")).strftime(fmt)
    return f"{s}_{e}"


def _close(a: float | None, b: float | None) -> bool:
    if a is None or b is None:
        return a is b
    return abs(a - b) <= 1e-9


def _check(row: dict[str, str], status: str, stage: float | None, ffile: str | None) -> list[str]:
    bad = []
    if status != (row["agent_status"] or None):
        bad.append(f"status {status} vs {row['agent_status']}")
    exp_stage = float(row["agent_stage_ft"]) if row["agent_stage_ft"] else None
    if not _close(stage, exp_stage):
        bad.append(f"stage {stage} vs {exp_stage}")
    if ffile != (row["agent_frame_file"] or row["file"]):
        bad.append(f"frame_file {ffile} vs {row['agent_frame_file']}")
    return bad


def _compare_saved(site: str, rows: list[dict[str, str]], missing: set[str], outdir: Path) -> int:
    """Compare already-written traces against the CSV without camera work."""
    mismatches = 0
    for row in rows:
        if row["file"] in missing:
            continue
        d = json.loads((outdir / _trace_name(site, row)).read_text())
        bad = _check(row, str(d["final_status"]), d.get("final_stage_ft"), str(d["frame_file"]))
        if bad:
            mismatches += 1
            print(f"  MISMATCH {row['file']}: {'; '.join(bad)}", flush=True)
    return mismatches


def worker(site: str, start: str, end: str, every: int, exclude: frozenset[str]) -> int:
    """Replay one window. Returns the mismatch count."""
    label = _label(start, end)
    csv_path = RESULTS / f"final_eval_agent_{site}_{label}.csv"
    rows = _csv_rows(csv_path)
    missing = set()
    miss_path = RESULTS / "final_missing_frames.txt"
    if miss_path.exists():
        missing = {ln.split()[0] for ln in miss_path.read_text().splitlines() if ln.strip()}
    outdir = TRACE_DIR / site
    outdir.mkdir(parents=True, exist_ok=True)
    all_done = all(
        row["file"] in missing or (outdir / _trace_name(site, row)).exists() for row in rows
    )
    if all_done:
        mismatches = _compare_saved(site, rows, missing, outdir)
        print(f"  {site} {label}: {len(rows)} frames traced, {mismatches} mismatches")
        print(f"RESULT {mismatches}", flush=True)
        return 0 if mismatches == 0 else 1
    cfg = sites.load_site(site)
    ctx = ev.make_perception(cfg, Settings())
    s = datetime.fromisoformat(start.replace("Z", "+00:00")).astimezone(UTC)
    e = datetime.fromisoformat(end.replace("Z", "+00:00")).astimezone(UTC)
    refs = {f.filename: f for f in ev.frame_list(cfg, s, e, every, exclude)}
    client = make_client()
    camera = {c.cam_id: c for c in nims.list_cameras(client)}[cfg.cam_id]
    nav = FrameNavigator(camera, CACHE)
    history = CameraHistory()
    mismatches = skipped = 0
    for row in rows:
        if row["file"] in missing:
            skipped += 1
            continue
        ref = refs.get(row["file"])
        if ref is None:
            print(f"  WARN {row['file']} not in frame list", flush=True)
            continue
        image = nims.fetch_image(client, ref, CACHE)
        if image is None:
            skipped += 1
            continue
        reading, trace = run_frame(ctx, image, ref, nav=nav, history=history)
        history.update(reading)
        (outdir / f"{trace.trace_id}.json").write_text(trace.model_dump_json(indent=2))
        bad = _check(row, reading.status, reading.stage_ft, reading.frame_file)
        if bad:
            mismatches += 1
            print(f"  MISMATCH {row['file']}: {'; '.join(bad)}", flush=True)
    print(
        f"  {site} {label}: {len(rows) - skipped} frames, {mismatches} mismatches",
        flush=True,
    )
    print(f"RESULT {mismatches}", flush=True)
    return 0 if mismatches == 0 else 1


def _windows_for(site: str) -> list[tuple[str, str, int]]:
    data = json.loads((RESULTS / "test_windows.json").read_text())
    out = []
    for w in data.get(site, []):
        if not isinstance(w, dict):
            continue
        if site == "in_overland" and not str(w["start"]).startswith(IO_YEAR):
            continue
        out.append((str(w["start"]), str(w["end"]), int(w["every"])))
    return out


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        nims.list_frames = _patient(nims.list_frames)
        nims.list_cameras = _patient(nims.list_cameras)
        nims.fetch_image = _patient(nims.fetch_image)
        data = json.loads((RESULTS / "test_windows.json").read_text())
        exclude = frozenset(str(f) for f in data.get("exclude_files", []))
        _site, start, end, every = (
            sys.argv[2],
            sys.argv[3],
            sys.argv[4],
            int(sys.argv[5]),
        )
        return worker(_site, start, end, every, exclude)
    total = 0
    for site in SITES:
        for start, end, every in _windows_for(site):
            print(f"REPLAY {site} {start}..{end} every={every}", flush=True)
            t0 = time.perf_counter()
            r = subprocess.run(
                [sys.executable, __file__, "--worker", site, start, end, str(every)],
                check=False,
                capture_output=True,
                text=True,
            )
            sys.stdout.write(r.stdout)
            sys.stderr.write(r.stderr)
            result_line = [ln for ln in r.stdout.splitlines() if ln.startswith("RESULT ")]
            n = int(result_line[-1].split()[1]) if result_line else -1
            print(f"  exit={r.returncode} {time.perf_counter() - t0:.0f}s", flush=True)
            if n < 0:
                print(f"  worker produced no result for {site} {start}; stopping")
                return 2
            total += n
    print(f"TOTAL mismatches: {total}", flush=True)
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
