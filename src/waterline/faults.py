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
from typing import Literal

import numpy as np

from waterline.agent.flag import FlagEngine
from waterline.models import Flag, Reading, Trace
from waterline.settings import Settings

SEED = 20261026
TRIALS_PER_TYPE = 20

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
    faults: dict[str, object] = {}
    for kind in FAULT_TYPES:
        detected = 0
        matched = 0
        delays: list[float] = []
        pre_onset = 0
        for _ in range(TRIALS_PER_TYPE):
            i0 = int(rng.integers(n // 4, 3 * n // 4))
            flags = collect(site_id, inject(rows, kind, i0), settings)
            pre_onset += sum(1 for i, _ in flags if i < i0)
            post = [(i, f) for i, f in flags if i >= i0]
            if post:
                i, flag = post[0]
                detected += 1
                matched += int(flag.event_type == kind)
                delays.append((rows[i].time - rows[i0].time).total_seconds() / 3600)
        faults[kind] = {
            "recall": detected / TRIALS_PER_TYPE,
            "type_match": (matched / detected) if detected else None,
            "median_delay_h": float(np.median(delays)) if delays else None,
            "pre_onset_flags": pre_onset,
        }
    flags = collect(site_id, rows, settings)
    by_type = {k: sum(1 for _, f in flags if f.event_type == k) for k in FAULT_TYPES}
    span_days = (rows[-1].time - rows[0].time).total_seconds() / 86400
    return {
        "injected": True,
        "seed": SEED,
        "trials_per_type": TRIALS_PER_TYPE,
        "n_frames": n,
        "clean": {
            "flags": len(flags),
            "by_type": by_type,
            "false_flags_per_site_week": len(flags) / max(span_days / 7, 1e-9),
        },
        "faults": faults,
    }


def run_faults_file(site_id: str, start: str, end: str, results: Path = Path("results")) -> Path:
    """Run the harness on a committed agent eval CSV and write its JSON report."""
    rows = load_rows(results / f"eval_agent_{site_id}_{start}_{end}.csv")
    report = run_faults(site_id, rows)
    out = results / f"faults_{site_id}_{start}_{end}.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return out
