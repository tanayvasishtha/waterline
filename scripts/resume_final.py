"""Resume the --final run one leg per subprocess.

The frozen eval code stays untouched: each remaining (site, window, mode) leg
runs in a fresh `python -c` child that calls ev.run_eval(..., final=True,
out_prefix='final_'), so the parent process holds nothing between legs. A leg
whose final_ CSV and summary already exist is skipped. A child that dies on
an out-of-memory signature is retried once; a second failure stops the run.
After every leg exists, pooled per-site summaries are rebuilt from the CSVs
and the injected-fault harness runs over them.
"""

import csv
import ctypes
import json
import subprocess
import sys
import time
from ctypes import wintypes
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from waterline import eval as ev
from waterline import faults
from waterline.eval import FrameRow

RESULTS = Path("results")
TEST_WINDOWS = RESULTS / "test_windows.json"
LOG = RESULTS / "final_run_log.txt"

CHILD = """
import time
from waterline.usgs import nims
from waterline.usgs.http import UsgsError
def _p(fn):
    def w(*a, **k):
        for i in range(200):
            try:
                return fn(*a, **k)
            except UsgsError as e:
                m = str(e)
                if ('rate limited' not in m and 'ConnectError' not in m
                        and 'failed after retries' not in m):
                    raise
                print('    retry 65s (#%d)' % (i + 1), flush=True)
                time.sleep(65)
            except Exception as e:
                if 'Connect' not in type(e).__name__ and 'Timeout' not in type(e).__name__:
                    raise
                print('    conn retry 65s (#%d)' % (i + 1), flush=True)
                time.sleep(65)
        raise UsgsError('limit stuck')
    return w
nims.list_frames = _p(nims.list_frames)
nims.list_cameras = _p(nims.list_cameras)
nims.fetch_image = _p(nims.fetch_image)
_orig_fetch = nims.fetch_image
def _fetch_skip_missing(client, ref, cache):
    try:
        return _orig_fetch(client, ref, cache)
    except UsgsError as e:
        if '404' not in str(e):
            raise
        with open('results/final_missing_frames.txt', 'a') as fh:
            fh.write('{site} {start} {end} {agent} ' + ref.filename + chr(10))
        print('MISSING ' + ref.filename, flush=True)
        return None
nims.fetch_image = _fetch_skip_missing
import waterline.eval as ev
ev.run_eval({site!r}, {start!r}, {end!r}, {every}, agent={agent}, final=True, out_prefix='final_')
"""

OOM_MARKERS = ("Insufficient memory", "OutOfMemory", "MemoryError")


class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", wintypes.DWORD),
        ("dwMemoryLoad", wintypes.DWORD),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def available_mb() -> int:
    stat = MEMORYSTATUSEX()
    stat.dwLength = ctypes.sizeof(stat)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
    return int(stat.ullAvailPhys / 1024 / 1024)


class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def peak_rss_mb(pid: int) -> int | None:
    handle = ctypes.windll.kernel32.OpenProcess(0x0400 | 0x0010, False, pid)
    if not handle:
        return None
    pmc = PROCESS_MEMORY_COUNTERS()
    pmc.cb = ctypes.sizeof(pmc)
    ok = ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(pmc), pmc.cb)
    ctypes.windll.kernel32.CloseHandle(handle)
    return int(pmc.PeakWorkingSetSize / 1024 / 1024) if ok else None


def _label(start: str, end: str) -> str:
    s = datetime.fromisoformat(start.replace("Z", "+00:00"))
    e = datetime.fromisoformat(end.replace("Z", "+00:00"))
    return f"{s.date()}_{e.date()}"


def _artifacts(site: str, label: str, agent: bool) -> list[Path]:
    stem = f"eval{'_agent' if agent else ''}_{site}_{label}"
    return [RESULTS / f"final_{stem}.csv", RESULTS / f"summary_final_{stem}.json"]


def _marker_pids(marker: str) -> list[int]:
    """All python processes whose command line carries the leg marker.

    The venv shim and the real interpreter it spawns share the same -c
    payload, so both match; the caller takes the max peak RSS across
    matches, which always lands on the real interpreter (the ~5 MB shim
    loses every max()).
    """
    try:
        out = subprocess.check_output(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_Process -Filter \"Name like 'python%'\""
                f" | Where-Object {{$_.CommandLine -match '{marker}'}}"
                " | Select-Object -Expand ProcessId",
            ],
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, OSError):
        return []
    return [int(x) for x in out.split() if x.isdigit()]


def _run_leg(site: str, start: str, end: str, every: int, agent: bool) -> tuple[int, int, str]:
    """One leg in a fresh subprocess; returns (wall_s, peak_rss_mb, output)."""
    marker = f"WLLEG_{site}_{_label(start, end)}_{'a' if agent else 'b'}"
    payload = CHILD.format(site=site, start=start, end=end, every=every, agent=agent)
    payload += f"\nMARKER = {marker!r}\n"
    t0 = time.time()
    proc = subprocess.Popen(
        [sys.executable, "-c", payload],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(Path.cwd()),
    )
    peak = 0
    while proc.poll() is None:
        for pid in _marker_pids(marker):
            got = peak_rss_mb(pid)
            if got is not None:
                peak = max(peak, got)
        time.sleep(2)
    for pid in _marker_pids(marker):
        got = peak_rss_mb(pid)
        if got is not None:
            peak = max(peak, got)
    out = proc.stdout.read() if proc.stdout else ""
    return int(time.time() - t0), peak, out


def _log(line: str) -> None:
    print(line, flush=True)
    with LOG.open("a") as fh:
        fh.write(line + "\n")


def _to_frame_row(r: dict[str, str]) -> dict[str, Any]:
    num = (
        "gage_ft",
        "stage_ft",
        "sigma_ft",
        "method_a_ft",
        "method_b_ft",
        "blur_var",
        "glare_frac",
        "mean_v",
        "inlier_ratio",
        "shift_px",
        "plate_visibility",
        "agent_stage_ft",
        "agent_sigma_ft",
        "agent_gage_ft",
    )
    row: dict[str, Any] = {}
    for k, v in r.items():
        if k in ("is_night", "gate_pass"):
            row[k] = v == "True"
        elif k in ("inliers", "agent_steps"):
            row[k] = int(v) if v else None
        elif k in num:
            row[k] = float(v) if v else None
        else:
            row[k] = v or None
    return row


def _load_final_rows(path: Path) -> list[FrameRow]:
    with path.open() as fh:
        return [cast(FrameRow, _to_frame_row(r)) for r in csv.DictReader(fh)]


def _legs(data: dict[str, Any]) -> list[tuple[str, str, str, int, bool]]:
    legs: list[tuple[str, str, str, int, bool]] = []
    for site_id, windows in data.items():
        if not isinstance(windows, list):
            continue
        for w in windows:
            if not isinstance(w, dict):
                continue
            for agent in (False, True):
                legs.append((site_id, str(w["start"]), str(w["end"]), int(w["every"]), agent))
    return legs


def _pool_and_faults(data: dict[str, Any]) -> None:
    for site_id, windows in data.items():
        if not isinstance(windows, list):
            continue
        window_rows: list[tuple[str, list[FrameRow]]] = []
        complete = True
        for w in windows:
            if not isinstance(w, dict):
                continue
            label = _label(str(w["start"]), str(w["end"]))
            csv_path = RESULTS / f"final_eval_agent_{site_id}_{label}.csv"
            if not csv_path.exists():
                _log(f"POOL-SKIP {site_id}: missing {csv_path.name}")
                complete = False
                break
            window_rows.append((label, _load_final_rows(csv_path)))
        if complete and window_rows:
            path = ev._write_final_summary(site_id, window_rows)
            _log(f"POOLED {site_id} -> {path}")
    outs = faults.run_final_faults(RESULTS)
    for site_id, path in outs.items():
        _log(f"FAULTS {site_id} -> {path}")


def _leg(site_id: str, start: str, end: str, every: int, agent: bool) -> int:
    label = _label(start, end)
    mode = "agent" if agent else "baseline"
    if all(p.exists() for p in _artifacts(site_id, label, agent)):
        _log(f"SKIP {site_id} {label} {mode}: artifacts exist")
        return 0
    tag = f"{site_id} {label} {mode}"
    for attempt in (1, 2):
        _log(f"LEG {tag} attempt={attempt} avail_before_mb={available_mb()}")
        wall, peak, out = _run_leg(site_id, start, end, every, agent)
        ok = all(p.exists() for p in _artifacts(site_id, label, agent))
        _log(f"DONE {tag} attempt={attempt} wall_s={wall} peak_rss_mb={peak} ok={ok}")
        if ok:
            return 0
        tail = out.strip().splitlines()[-5:] if out.strip() else ["<no output>"]
        for line in tail:
            _log(f"  | {line}")
        if not (any(m in out for m in OOM_MARKERS) and attempt == 1):
            _log(f"STOP {tag}: failed attempt={attempt}")
            return 1
        _log(f"RETRY {tag}: OOM signature, retrying once")
    return 1


def main() -> int:
    data = json.loads(TEST_WINDOWS.read_text())
    for site_id, start, end, every, agent in _legs(data):
        if _leg(site_id, start, end, every, agent):
            return 1
    _pool_and_faults(data)
    _log("RESUME COMPLETE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
