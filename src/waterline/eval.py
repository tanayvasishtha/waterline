"""Calibration and evaluation over archived frames.

Two honest numbers come out of this: the datum offset learned on the calibration
window (which absorbs plate anchor error plus any true sensor-vs-plate offset), and
the held-out event metrics. Nothing here may tune on test windows.
"""

import csv
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TypedDict

import cv2
import numpy as np

from waterline import sites
from waterline.models import FrameRef, SiteConfig
from waterline.pipeline import Perception, read_frame
from waterline.plate import rectify
from waterline.quality import quality_ok
from waterline.register import build_reference
from waterline.settings import Settings
from waterline.usgs import gage, nims
from waterline.usgs.http import make_client


class FrameRow(TypedDict):
    file: str
    time: str
    gage_ft: float | None
    stage_ft: float | None
    sigma_ft: float | None
    method_a_ft: float | None
    method_b_ft: float | None
    reason: str | None
    blur_var: float
    glare_frac: float
    mean_v: float
    inliers: int
    inlier_ratio: float
    shift_px: float
    plate_visibility: float
    is_night: bool
    gate_pass: bool


def frame_list(
    site: SiteConfig,
    start: datetime,
    end: datetime,
    every_n: int,
) -> list[FrameRef]:
    """Every nth archived frame in the window."""
    client = make_client()
    camera = {c.cam_id: c for c in nims.list_cameras(client)}[site.cam_id]
    frames = nims.list_frames(client, camera, after=start, before=end, limit=10000)
    return frames[::every_n]


def run_window(
    ctx: Perception,
    frames: list[FrameRef],
    cache: Path,
) -> list[FrameRow]:
    """Read each frame, attach the nearest gage truth, return raw per-frame rows."""
    client = make_client()
    site = ctx.site
    truth = gage.continuous(
        client, site.nwis_id, min(f.time for f in frames), max(f.time for f in frames)
    )
    rows: list[FrameRow] = []
    for i, ref in enumerate(frames):
        image = nims.fetch_image(client, ref, cache)
        if image is None:
            continue
        gage_ft = gage.nearest(truth, ref.time, max_gap=timedelta(minutes=10))
        result = read_frame(ctx, image, ref.time, seed=i)
        rows.append(
            {
                "file": ref.filename,
                "time": ref.time.isoformat(),
                "gage_ft": None if gage_ft is None else gage_ft.value_ft,
                "stage_ft": result.reading.stage_ft,
                "sigma_ft": result.reading.sigma_ft,
                "method_a_ft": result.reading.method_a_ft,
                "method_b_ft": result.reading.method_b_ft,
                "reason": result.reading.reason,
                "blur_var": round(result.quality.blur_var, 1),
                "glare_frac": round(result.quality.glare_frac, 3),
                "mean_v": round(result.quality.mean_v, 1),
                "inliers": result.quality.registration_inliers,
                "inlier_ratio": round(result.quality.inlier_ratio, 3),
                "shift_px": round(result.quality.shift_px, 1),
                "plate_visibility": round(result.quality.plate_visibility, 1),
                "is_night": result.quality.is_night,
                "gate_pass": quality_ok(result.quality, ctx.settings),
            }
        )
    return rows


def fit_offset(rows: list[FrameRow]) -> dict[str, object]:
    """Median camera-minus-gage bias over accepted, in-range readings."""
    diffs = [
        r["stage_ft"] - r["gage_ft"]
        for r in rows
        if r["stage_ft"] is not None and r["gage_ft"] is not None
    ]
    return {"offset_ft": float(np.median(diffs)) if diffs else 0.0, "n": len(diffs)}


def _summarize_slice(rows: list[FrameRow]) -> dict[str, object]:
    """Single-pass baseline: rows failing the triage gate count as abstained."""
    gated = [r for r in rows if r["gate_pass"]]
    read = [
        (r["stage_ft"], r["gage_ft"], r["sigma_ft"])
        for r in gated
        if r["stage_ft"] is not None and r["gage_ft"] is not None
    ]
    below = sum(1 for r in gated if r["reason"] == "below_plate")
    errors = np.array([a - g for a, g, _ in read if a is not None and g is not None])
    within = (
        float(np.mean(np.abs(errors) <= 2 * np.array([s for _, _, s in read]))) if read else 0.0
    )
    return {
        "frames": len(rows),
        "gate_pass": len(gated),
        "read": len(read),
        "below_plate": below,
        "coverage": len(read) / max(len(rows), 1),
        "mae_ft": float(np.abs(errors).mean()) if read else None,
        "med_ae_ft": float(np.median(np.abs(errors))) if read else None,
        "p90_ae_ft": float(np.percentile(np.abs(errors), 90)) if read else None,
        "within_2sigma": within,
    }


def summarize(rows: list[FrameRow]) -> dict[str, object]:
    day = [r for r in rows if not r["is_night"]]
    night = [r for r in rows if r["is_night"]]
    return {
        "all": _summarize_slice(rows),
        "day": _summarize_slice(day),
        "night": _summarize_slice(night),
    }


def write_rows(rows: list[FrameRow], path: Path) -> None:
    if not rows:
        raise ValueError("no frame rows produced")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_perception(site: SiteConfig, settings: Settings) -> Perception:
    """Bank + rectified reference strip (the baseline for glare detection)."""
    bank = build_reference(site, Path("sites"), settings)
    first = next(iter(site.reference_frames.values()))
    ref_image = cv2.imread(str(sites.site_dir(site.site_id) / first))
    ref_strip = None if ref_image is None else rectify(ref_image, site, settings).image
    return Perception(site, settings, bank, ref_strip)


def _run(site_id: str, start: str, end: str, every_n: int, label: str) -> dict[str, object]:
    settings = Settings()
    site = sites.load_site(site_id)
    ctx = make_perception(site, settings)
    start_dt = datetime.fromisoformat(start).replace(tzinfo=UTC)
    end_dt = datetime.fromisoformat(end).replace(tzinfo=UTC)
    frames = frame_list(site, start_dt, end_dt, every_n)
    rows = run_window(ctx, frames, Path("data/cache"))
    write_rows(rows, Path("results") / f"{label}_{site_id}_{start_dt.date()}_{end_dt.date()}.csv")
    report: dict[str, object] = {
        "site": site_id,
        "window": [start, end],
        "rows": len(rows),
        "offset": fit_offset(rows),
        "summary": summarize(rows),
    }
    print(json.dumps(report, indent=2))
    return report


def run_calibrate(site_id: str, start: str, end: str, every_n: int = 6) -> dict[str, object]:
    return _run(site_id, start, end, every_n, "calib")


def run_eval(site_id: str, start: str, end: str, every_n: int = 1) -> dict[str, object]:
    return _run(site_id, start, end, every_n, "eval")
