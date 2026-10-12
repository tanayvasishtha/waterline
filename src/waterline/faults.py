"""Injected-fault harness: replay an agent eval CSV through the fault detector.

For each fault type the gage series gets one synthetic fault at a seeded onset
and the detector's detection rate, type match and delay are recorded. The clean
run reports how often the detector cries wolf on real data.
"""

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np

from waterline.agent.flag import FlagEngine
from waterline.models import Flag, Reading, Trace
from waterline.settings import Settings

SEED = 20261026
TRIALS_PER_TYPE = 20
DETECTION_WINDOW_ROWS = 24

FaultType = Literal["drift", "stuck", "step", "spike"]
FAULT_TYPES: tuple[FaultType, ...] = ("drift", "stuck", "step", "spike")


@dataclass
class Row:
    """One eval_agent row reduced to what the detector sees."""

    time: datetime
    file: str
    gage_ft: float
    camera_ft: float | None
    sigma_ft: float | None


def _f(value: str) -> float | None:
    return float(value) if value else None


def load_rows(path: Path) -> list[Row]:
    """eval_agent_{site}_{start}_{end}.csv rows that carry a gage truth, in order."""
    rows = []
    with path.open() as fh:
        for r in csv.DictReader(fh):
            if not r["gage_ft"]:
                continue
            accepted = r.get("agent_status") == "accepted"
            rows.append(
                Row(
                    time=datetime.fromisoformat(r["time"]),
                    file=r["file"],
                    gage_ft=float(r["gage_ft"]),
                    camera_ft=_f(r["agent_stage_ft"]) if accepted else None,
                    sigma_ft=_f(r["agent_sigma_ft"]),
                )
            )
    return rows


def inject(rows: list[Row], kind: FaultType, i0: int) -> list[Row]:
    """Return a copy of rows with one fault injected into the gage, from i0 on."""
    out = [Row(r.time, r.file, r.gage_ft, r.camera_ft, r.sigma_ft) for r in rows]
    base = rows[i0]
    if kind == "drift":
        for i in range(i0, len(out)):
            minutes = (out[i].time - base.time).total_seconds() / 60
            out[i].gage_ft += 0.005 * minutes / 15
    elif kind == "stuck":
        for i in range(i0, len(out)):
            out[i].gage_ft = base.gage_ft
    elif kind == "step":
        for i in range(i0, len(out)):
            out[i].gage_ft += 0.8
    elif kind == "spike":
        out[i0].gage_ft += 3.0
    return out


def _reading(site_id: str, row: Row) -> Reading:
    return Reading(
        site_id=site_id,
        time=row.time,
        stage_ft=row.camera_ft,
        sigma_ft=row.sigma_ft,
        status="accepted" if row.camera_ft is not None else "abstained",
    )


def _trace(site_id: str, row: Row) -> Trace:
    return Trace(
        trace_id=f"{site_id}-{row.time:%Y%m%dT%H%MZ}",
        site_id=site_id,
        frame_time=row.time,
        frame_file=row.file,
        started_at=row.time,
        finished_at=row.time,
        steps=[],
        final_status="accepted" if row.camera_ft is not None else "abstained",
        final_stage_ft=row.camera_ft,
        final_sigma_ft=row.sigma_ft,
        reason=None,
        policy="deterministic",
    )


def collect(site_id: str, rows: list[Row], settings: Settings) -> list[tuple[int, Flag]]:
    """Feed a row series to a fresh engine; return (row index, flag) pairs."""
    engine = FlagEngine(site_id)
    flags = []
    for i, row in enumerate(rows):
        flag = engine.feed(
            _reading(site_id, row), row.gage_ft, _trace(site_id, row), settings, None
        )
        if flag is not None:
            flags.append((i, flag))
    return flags


def run_faults(
    site_id: str, rows: list[Row], settings: Settings | None = None
) -> dict[str, object]:
    """Seeded injection trials plus a clean pass; returns the report dict."""
    settings = settings or Settings()
    n = len(rows)
    rng = np.random.default_rng(SEED)
    flags = collect(site_id, rows, settings)
    clean = {(i, f.event_type) for i, f in flags}
    faults: dict[str, object] = {}
    for kind in FAULT_TYPES:
        detected = 0
        detected_any = 0
        matched = 0
        delays: list[float] = []
        pre_onset = 0
        for _ in range(TRIALS_PER_TYPE):
            i0 = int(rng.integers(n // 4, 3 * n // 4))
            trial = collect(site_id, inject(rows, kind, i0), settings)
            trial = [(i, f) for i, f in trial if (i, f.event_type) not in clean]
            pre_onset += sum(1 for i, _ in trial if i < i0)
            post = [(i, f) for i, f in trial if i >= i0]
            if post:
                detected_any += 1
            windowed = [(i, f) for i, f in post if i <= i0 + DETECTION_WINDOW_ROWS]
            if windowed:
                i, flag = windowed[0]
                detected += 1
                matched += int(flag.event_type == kind)
                delays.append((rows[i].time - rows[i0].time).total_seconds() / 3600)
        faults[kind] = {
            "recall": detected / TRIALS_PER_TYPE,
            "recall_any": detected_any / TRIALS_PER_TYPE,
            "type_match": (matched / detected) if detected else None,
            "median_delay_h": float(np.median(delays)) if delays else None,
            "pre_onset_flags": pre_onset,
            "n_detected": detected,
            "n_detected_any": detected_any,
            "n_type_matched": matched,
            "delays_h": delays,
        }
    by_type = {k: sum(1 for _, f in flags if f.event_type == k) for k in FAULT_TYPES}
    span_days = (rows[-1].time - rows[0].time).total_seconds() / 86400
    return {
        "injected": True,
        "seed": SEED,
        "trials_per_type": TRIALS_PER_TYPE,
        "n_frames": n,
        "detection_window_rows": DETECTION_WINDOW_ROWS,
        "clean": {
            "flags": len(flags),
            "by_type": by_type,
            "false_flags_per_site_week": len(flags) / max(span_days / 7, 1e-9),
        },
        "faults": faults,
    }


def run_faults_file(
    site_id: str,
    start: str,
    end: str,
    results: Path = Path("results"),
    prefix: str = "",
) -> Path:
    """Run the harness on a committed agent eval CSV and write its JSON report."""
    rows = load_rows(results / f"{prefix}eval_agent_{site_id}_{start}_{end}.csv")
    report = run_faults(site_id, rows)
    out = results / f"{prefix}faults_{site_id}_{start}_{end}.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return out


def _parse_ts(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _pool(
    site_id: str,
    per_window: list[dict[str, Any]],
    skipped: list[str],
) -> dict[str, object]:
    """Pool per-window fault reports into the site's final_faults report."""
    n_trials = sum(int(w["report"]["trials_per_type"]) for w in per_window)
    pooled: dict[str, object] = {}
    for kind in FAULT_TYPES:
        det = sum(int(w["report"]["faults"][kind]["n_detected"]) for w in per_window)
        det_any = sum(int(w["report"]["faults"][kind]["n_detected_any"]) for w in per_window)
        matched = sum(int(w["report"]["faults"][kind]["n_type_matched"]) for w in per_window)
        delays = [d for w in per_window for d in w["report"]["faults"][kind]["delays_h"]]
        pre = sum(int(w["report"]["faults"][kind]["pre_onset_flags"]) for w in per_window)
        pooled[kind] = {
            "recall": det / n_trials if n_trials else None,
            "recall_any": det_any / n_trials if n_trials else None,
            "type_match": matched / det if det else None,
            "median_delay_h": float(np.median(delays)) if delays else None,
            "pre_onset_flags": pre,
            "n_detected": det,
            "n_detected_any": det_any,
            "n_type_matched": matched,
            "trials": n_trials,
        }
    clean_flags = sum(int(w["report"]["clean"]["flags"]) for w in per_window)
    clean_by_type = {
        k: sum(int(w["report"]["clean"]["by_type"][k]) for w in per_window) for k in FAULT_TYPES
    }
    span_days = sum(float(w["span_days"]) for w in per_window)
    return {
        "injected": True,
        "site": site_id,
        "seed": SEED,
        "trials_per_type": TRIALS_PER_TYPE,
        "detection_window_rows": DETECTION_WINDOW_ROWS,
        "n_windows": len(per_window),
        "skipped": skipped,
        "clean": {
            "flags": clean_flags,
            "by_type": clean_by_type,
            "false_flags_per_site_week": clean_flags / max(span_days / 7, 1e-9),
        },
        "faults": pooled,
        "windows": per_window,
    }


def run_final_faults(results: Path = Path("results")) -> dict[str, Path]:
    """Pooled injected-fault scoring over each site's final agent windows.

    Every frozen test window's final_eval_agent CSV is replayed; windows with
    fewer than 8 gage-bearing rows are skipped and reported. Per-window
    reports land in final_faults_{site}_{start}_{end}.json and the pooled
    site report in final_faults_{site}.json.
    """
    data = json.loads((results / "test_windows.json").read_text())
    outs: dict[str, Path] = {}
    for site_id, windows in data.items():
        if not isinstance(windows, list):
            continue
        per_window: list[dict[str, Any]] = []
        skipped: list[str] = []
        for w in windows:
            if not isinstance(w, dict):
                continue
            start, end = str(w["start"]), str(w["end"])
            s, e = _parse_ts(start), _parse_ts(end)
            label = f"{s.date()}_{e.date()}"
            csv_path = results / f"final_eval_agent_{site_id}_{label}.csv"
            if not csv_path.exists():
                skipped.append(f"{label}: no final_eval_agent CSV")
                continue
            rows = load_rows(csv_path)
            if len(rows) < 8:
                skipped.append(f"{label}: only {len(rows)} gage rows")
                continue
            rep = run_faults(site_id, rows)
            per = results / f"final_faults_{site_id}_{label}.json"
            per.write_text(json.dumps(rep, indent=2))
            per_window.append(
                {
                    "window": [start, end],
                    "span_days": (e - s).total_seconds() / 86400,
                    "report": rep,
                }
            )
        pooled = _pool(site_id, per_window, skipped)
        out = results / f"final_faults_{site_id}.json"
        out.write_text(json.dumps(pooled, indent=2))
        print(json.dumps(pooled, indent=2))
        outs[site_id] = out
    return outs
