"""Calibration and evaluation over archived frames.

Two honest numbers come out of this: the datum offset learned on the calibration
window (which absorbs plate anchor error plus any true sensor-vs-plate offset), and
the held-out event metrics. Nothing here may tune on test windows.
"""

import csv
import hashlib
import itertools
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TypedDict, cast

import cv2
import numpy as np

from waterline import sites
from waterline.agent.history import CameraHistory
from waterline.agent.loop import run_frame
from waterline.agent.tools import FrameNavigator
from waterline.models import FrameRef, GageReading, SiteConfig
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
    agent_stage_ft: float | None
    agent_sigma_ft: float | None
    agent_gage_ft: float | None
    agent_status: str | None
    agent_steps: int | None


TEST_WINDOWS_PATH = Path("results/test_windows.json")
PROTOCOL_PATH = Path("eval/PROTOCOL.md")
PROTOCOL_HASH_PATH = Path("results/protocol_hash.txt")


class TestWindowError(RuntimeError):
    """A requested window touches a frozen test window, or --final integrity fails."""


def _overlaps(a0: datetime, a1: datetime, b0: datetime, b1: datetime) -> bool:
    """Half-open [start, end) interval overlap."""
    return a0 < b1 and b0 < a1


def _check_window(site_id: str, start_dt: datetime, end_dt: datetime, final: bool) -> None:
    """Refuse frozen test windows unless final, and refuse final on a hash mismatch."""
    if TEST_WINDOWS_PATH.exists():
        data = json.loads(TEST_WINDOWS_PATH.read_text())
        windows = data.get(site_id, [])
        hit = any(
            _overlaps(
                start_dt,
                end_dt,
                datetime.fromisoformat(str(w["start"]).replace("Z", "+00:00")),
                datetime.fromisoformat(str(w["end"]).replace("Z", "+00:00")),
            )
            for w in windows
            if isinstance(w, dict)
        )
        if hit and not final:
            raise TestWindowError(
                f"{site_id} {start_dt.date()}..{end_dt.date()} overlaps a frozen test "
                "window from results/test_windows.json; rerun with --final"
            )
    if final:
        expected = (
            PROTOCOL_HASH_PATH.read_text().strip() if PROTOCOL_HASH_PATH.exists() else "<missing>"
        )
        actual = hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest()
        if actual != expected:
            raise TestWindowError(
                "eval/PROTOCOL.md changed since the protocol hash was frozen; "
                "--final refuses to run until results/protocol_hash.txt is refreshed"
            )


def _select(frames: list[FrameRef], every_n: int, exclude: frozenset[str]) -> list[FrameRef]:
    """Every nth frame, minus excluded filenames (reference frames, by rule)."""
    return [f for i, f in enumerate(frames) if i % every_n == 0 and f.filename not in exclude]


def frame_list(
    site: SiteConfig,
    start: datetime,
    end: datetime,
    every_n: int,
    exclude: frozenset[str] = frozenset(),
) -> list[FrameRef]:
    """Every nth archived frame in the window."""
    client = make_client()
    camera = {c.cam_id: c for c in nims.list_cameras(client)}[site.cam_id]
    frames = nims.list_frames(client, camera, after=start, before=end, limit=10000)
    return _select(frames, every_n, exclude)


def run_window(
    ctx: Perception,
    frames: list[FrameRef],
    cache: Path,
    nav: FrameNavigator | None = None,
) -> list[FrameRow]:
    """Read each frame, attach the nearest gage truth, return raw per-frame rows."""
    client = make_client()
    site = ctx.site
    truth = gage.continuous(
        client, site.nwis_id, min(f.time for f in frames), max(f.time for f in frames)
    )
    rows: list[FrameRow] = []
    history = CameraHistory()
    for i, ref in enumerate(frames):
        image = nims.fetch_image(client, ref, cache)
        if image is None:
            continue
        gage_ft = gage.nearest(truth, ref.time, max_gap=timedelta(minutes=10))
        if nav is not None:
            agent_reading, trace = run_frame(ctx, image, ref, nav, history=history)
            history.update(agent_reading)
            result = read_frame(ctx, image, ref.time, seed=i)
            agent_status = agent_reading.status
            agent_stage = agent_reading.stage_ft
            agent_sigma = agent_reading.sigma_ft
            agent_gage = gage.nearest(truth, agent_reading.time, max_gap=timedelta(minutes=10))
        else:
            result = read_frame(ctx, image, ref.time, seed=i)
            agent_status = None
            agent_stage = None
            agent_sigma = None
            agent_gage = None
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
                "agent_stage_ft": agent_stage,
                "agent_sigma_ft": agent_sigma,
                "agent_gage_ft": None if agent_gage is None else agent_gage.value_ft,
                "agent_status": agent_status,
                "agent_steps": None if nav is None else len(trace.steps),
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
    agent = _agent_summary(rows)
    out = _slice(rows)
    if agent:
        out["agent"] = agent
    return out


def _agent_summary(rows: list[FrameRow]) -> dict[str, object] | None:
    ar = [r for r in rows if r["agent_status"] is not None]
    if not ar:
        return None
    read = [
        (r["agent_stage_ft"], r["agent_gage_ft"], r["sigma_ft"])
        for r in ar
        if r["agent_status"] == "accepted"
        and r["agent_stage_ft"] is not None
        and r["agent_gage_ft"] is not None
    ]
    errors = np.array([a - g for a, g, _ in read if a is not None and g is not None])
    return {
        "accepted": sum(1 for r in ar if r["agent_status"] == "accepted"),
        "below_range": sum(1 for r in ar if r["agent_status"] == "below_range"),
        "abstained": sum(1 for r in ar if r["agent_status"] == "abstained"),
        "pending": sum(1 for r in ar if r["agent_status"] == "pending"),
        "read": len(read),
        "mae_ft": float(np.abs(errors).mean()) if read else None,
        "med_ae_ft": float(np.median(np.abs(errors))) if read else None,
        "p90_ae_ft": float(np.percentile(np.abs(errors), 90)) if read else None,
        "avg_steps": float(np.mean([r["agent_steps"] or 0 for r in ar])),
    }


def _slice(rows: list[FrameRow]) -> dict[str, object]:
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
    first_name = next(iter(site.reference_frames))
    first = site.reference_frames[first_name]
    ref_image = cv2.imread(str(sites.site_dir(site.site_id) / first))
    ref_strip = None if ref_image is None else rectify(ref_image, site, settings, first_name).image
    return Perception(site, settings, bank, ref_strip)


def _run(
    site_id: str,
    start: str,
    end: str,
    every_n: int,
    *,
    label: str,
    agent: bool,
    final: bool = False,
    exclude: frozenset[str] = frozenset(),
    out_prefix: str = "",
) -> dict[str, object]:
    settings = Settings()
    site = sites.load_site(site_id)
    ctx = make_perception(site, settings)
    start_dt = datetime.fromisoformat(start).replace(tzinfo=UTC)
    end_dt = datetime.fromisoformat(end).replace(tzinfo=UTC)
    _check_window(site_id, start_dt, end_dt, final)
    frames = frame_list(site, start_dt, end_dt, every_n, exclude)
    nav = None
    if agent:
        client = make_client()
        camera = {c.cam_id: c for c in nims.list_cameras(client)}[site.cam_id]
        nav = FrameNavigator(camera, Path("data/cache"))
    rows = run_window(ctx, frames, Path("data/cache"), nav)
    out_label = f"{label}_agent" if agent else label
    stamp = f"{out_prefix}{out_label}_{site_id}_{start_dt.date()}_{end_dt.date()}"
    write_rows(rows, Path("results") / f"{stamp}.csv")
    summary = summarize(rows)
    summary_path = Path("results") / f"summary_{stamp}.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    report: dict[str, object] = {
        "site": site_id,
        "window": [start, end],
        "rows": len(rows),
        "offset": fit_offset(rows),
        "summary": summary,
    }
    print(json.dumps(report, indent=2))
    return report


def run_calibrate(
    site_id: str, start: str, end: str, every_n: int = 6, *, final: bool = False
) -> dict[str, object]:
    return _run(site_id, start, end, every_n, label="calib", agent=False, final=final)


def run_eval(
    site_id: str,
    start: str,
    end: str,
    every_n: int = 1,
    *,
    agent: bool = False,
    final: bool = False,
    out_prefix: str = "",
) -> dict[str, object]:
    return _run(
        site_id,
        start,
        end,
        every_n,
        label="eval",
        agent=agent,
        final=final,
        out_prefix=out_prefix,
    )


def run_final() -> list[dict[str, object]]:
    """The one-shot test run: every frozen window, baseline and agent."""
    data = json.loads(TEST_WINDOWS_PATH.read_text())
    exclude = frozenset(str(f) for f in data.get("exclude_files", []))
    reports = []
    for site_id, windows in data.items():
        if not isinstance(windows, list):
            continue
        for w in windows:
            for agent in (False, True):
                reports.append(
                    _run(
                        site_id,
                        str(w["start"]),
                        str(w["end"]),
                        int(w["every"]),
                        label="eval",
                        agent=agent,
                        final=True,
                        exclude=exclude,
                    )
                )
    return reports


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def _fit_affine(rows: list[FrameRow]) -> dict[str, object]:
    """Least squares gage = a + b*raw over gate-passed rows with both values."""
    pts = [
        (r["stage_ft"], r["gage_ft"])
        for r in rows
        if r["gate_pass"] and r["stage_ft"] is not None and r["gage_ft"] is not None
    ]
    if len(pts) < 2:
        raise ValueError(f"need >=2 gate-passed rows to fit, got {len(pts)}")
    raw = np.array([p[0] for p in pts], dtype=float)
    g = np.array([p[1] for p in pts], dtype=float)
    b, a = np.polyfit(raw, g, 1)
    resid = g - (a + b * raw)
    return {
        "n": len(pts),
        "a": float(a),
        "b": float(b),
        "resid_std": float(resid.std(ddof=1)),
        "raw_min": float(raw.min()),
        "raw_max": float(raw.max()),
        "gage_min": float(g.min()),
        "gage_max": float(g.max()),
    }


def _rate_max(series: list[GageReading]) -> float | None:
    """1.5x the largest |dg/dt| between consecutive gage samples (ft/h)."""
    best = 0.0
    for prev, cur in itertools.pairwise(series):
        dt = (cur.time - prev.time).total_seconds()
        if dt > 0:
            best = max(best, abs(cur.value_ft - prev.value_ft) / (dt / 3600))
    return 1.5 * best if best > 0 else None


def run_calibrate_fit(site_id: str) -> dict[str, object]:
    """Reproducible affine fit over the site's calibration_windows.

    Perception runs with stage_calib forced to identity and datum_offset 0.
    The least-squares fit, residual sigma, and the gage-rate cap are written
    back to the site config; the fit report lands in results/calibfit_*.json.
    """
    site = sites.load_site(site_id)
    if not site.calibration_windows:
        raise ValueError(f"{site_id} has no calibration_windows in its config")
    settings = Settings()
    raw_site = site.model_copy(update={"stage_calib": [0.0, 1.0], "datum_offset_ft": 0.0})
    ctx = make_perception(raw_site, settings)
    client = make_client()
    rows: list[FrameRow] = []
    rate_max = 0.0
    window_desc = []
    for w in site.calibration_windows:
        start_dt, end_dt = _parse_time(w.start), _parse_time(w.end)
        _check_window(site_id, start_dt, end_dt, final=False)
        frames = frame_list(site, start_dt, end_dt, w.every)
        rows += run_window(ctx, frames, Path("data/cache"))
        series = gage.continuous(client, site.nwis_id, start_dt, end_dt)
        got = _rate_max(series)
        if got is not None:
            rate_max = max(rate_max, got)
        window_desc.append(f"{w.start}..{w.end} every {w.every}")
    fit = _fit_affine(rows)
    source = f"1.5x max |dg/dt| over calibration gage samples ({'; '.join(window_desc)})"
    a_fit = cast(float, fit["a"])
    b_fit = cast(float, fit["b"])
    resid_std = cast(float, fit["resid_std"])
    path = sites.SITES_DIR / f"{site_id}.json"
    cfg = json.loads(path.read_text())
    cfg["stage_calib"] = [round(a_fit, 4), round(b_fit, 4)]
    cfg["datum_offset_ft"] = 0.0
    cfg["sigma_calib_ft"] = round(resid_std, 4)
    cfg["rate_max_ft_per_h"] = round(rate_max, 4) if rate_max else None
    cfg["rate_max_source"] = source
    path.write_text(json.dumps(cfg, indent=1) + "\n")
    fit["rate_max_ft_per_h"] = round(rate_max, 4) if rate_max else None
    fit["rate_max_source"] = source
    Path("results").mkdir(exist_ok=True)
    Path(f"results/calibfit_{site_id}.json").write_text(json.dumps(fit, indent=2) + "\n")
    print(json.dumps(fit, indent=2))
    return fit
