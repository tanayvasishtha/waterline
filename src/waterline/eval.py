"""Calibration and evaluation over archived frames.

Two honest numbers come out of this: the datum offset learned on the calibration
window (which absorbs plate anchor error plus any true sensor-vs-plate offset), and
the held-out event metrics. Nothing here may tune on test windows.
"""

import csv
import json
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

from waterline import sites
from waterline.models import SiteConfig
from waterline.pipeline import Perception, read_frame
from waterline.register import build_reference
from waterline.settings import Settings
from waterline.usgs import gage, nims
from waterline.usgs.http import make_client


def frame_list(
    site: SiteConfig,
    start: datetime,
    end: datetime,
    every_n: int,
    settings: Settings,
) -> list[nims.FrameRef]:
    """Every nth archived frame in the window."""
    client = make_client()
    camera = {c.cam_id: c for c in nims.list_cameras(client)}[site.cam_id]
    frames = nims.list_frames(client, camera, after=start, before=end, limit=10000)
    return frames[::every_n]


def run_window(
    ctx: Perception,
    frames: list[nims.FrameRef],
    cache: Path,
) -> list[dict]:
    """Read each frame, attach the nearest gage truth, return raw per-frame rows."""
    settings = ctx.settings
    client = make_client()
    site = ctx.site
    truth = gage.continuous(
        client, site.nwis_id, min(f.time for f in frames), max(f.time for f in frames)
    )
    rows = []
    for i, ref in enumerate(frames):
        image = cv2.imread(str(nims.fetch_image(client, ref, cache)))
        if image is None:
            continue
        gage_ft = gage.nearest(truth, ref.time, max_gap_min=10)
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
            }
        )
    return rows


def fit_offset(rows: list[dict]) -> dict:
    """Median camera-minus-gage bias over accepted, in-range readings."""
    diffs = [
        r["stage_ft"] - r["gage_ft"]
        for r in rows
        if r["stage_ft"] is not None and r["gage_ft"] is not None
    ]
    return {"offset_ft": float(np.median(diffs)) if diffs else 0.0, "n": len(diffs)}


def summarize(rows: list[dict]) -> dict:
    read = [
        (r["stage_ft"], r["gage_ft"], r["sigma_ft"])
        for r in rows
        if r["stage_ft"] is not None and r["gage_ft"] is not None
    ]
    below = sum(1 for r in rows if r["reason"] == "below_plate")
    errors = np.array([a - g for a, g, _ in read])
    within = (
        float(np.mean(np.abs(errors) <= 2 * np.array([s for _, _, s in read])))
        if read
        else 0.0
    )
    return {
        "frames": len(rows),
        "read": len(read),
        "below_plate": below,
        "coverage": len(read) / max(len(rows), 1),
        "mae_ft": float(np.abs(errors).mean()) if read else None,
        "med_ae_ft": float(np.median(np.abs(errors))) if read else None,
        "p90_ae_ft": float(np.percentile(np.abs(errors), 90)) if read else None,
        "within_2sigma": within,
    }


def write_rows(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_calibrate(
    site_id: str, start: str, end: str, every_n: int = 6, out: str = "results"
) -> dict:
    settings = Settings()
    site = sites.load_site(site_id)
    bank = build_reference(site, Path("sites"), settings)
    ctx = Perception(site, settings, bank)
    start_dt = datetime.fromisoformat(start).replace(tzinfo=UTC)
    end_dt = datetime.fromisoformat(end).replace(tzinfo=UTC)
    frames = frame_list(site, start_dt, end_dt, every_n, settings)
    rows = run_window(ctx, frames, Path("data/cache"))
    write_rows(rows, Path(out) / f"calib_{site_id}_{start_dt.date()}.csv")
    report = {
        "site": site_id,
        "window": [start, end],
        "rows": len(rows),
        "offset": fit_offset(rows),
        "summary": summarize(rows),
    }
    print(json.dumps(report, indent=2))
    return report


def run_eval(
    site_id: str, start: str, end: str, every_n: int = 1, out: str = "results"
) -> dict:
    settings = Settings()
    site = sites.load_site(site_id)
    bank = build_reference(site, Path("sites"), settings)
    ctx = Perception(site, settings, bank)
    start_dt = datetime.fromisoformat(start).replace(tzinfo=UTC)
    end_dt = datetime.fromisoformat(end).replace(tzinfo=UTC)
    frames = frame_list(site, start_dt, end_dt, every_n, settings)
    rows = run_window(ctx, frames, Path("data/cache"))
    write_rows(rows, Path(out) / f"eval_{site_id}_{start_dt.date()}_{end_dt.date()}.csv")
    report = {"site": site_id, "window": [start, end], "summary": summarize(rows)}
    print(json.dumps(report, indent=2))
    return report
