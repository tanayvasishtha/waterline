"""Screen USGS staff-gauge cameras and rank sites for Waterline (`waterline screen`).

For every camera whose name or description mentions a staff gauge, this pulls one year of
gage height plus the posting interval, scores the site, and writes a contact sheet of the
latest frames for a final visual check.
"""

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from itertools import pairwise
from pathlib import Path

import cv2
import httpx
import numpy as np

from waterline.models import FrameRef, GageReading
from waterline.usgs import gage, nims
from waterline.usgs.http import UsgsError, make_client

STAFF_PATTERN = ("staff", "gage plate", "gauge plate")
MIN_RANGE_FT = 1.0
MAX_INTERVAL_MIN = 60
CONTACT_LIMIT = 12


@dataclass(frozen=True)
class SiteScore:
    cam_id: str
    nwis_id: str
    name: str
    range_ft: float
    events: int
    interval_min: float
    score: float
    note: str = ""


def is_staff_camera(camera: nims.CameraInfo) -> bool:
    text = f"{camera.cam_id} {camera.name} {camera.description}".lower()
    return any(p in text for p in STAFF_PATTERN) and camera.nwis_id is not None


def _events(readings: list[GageReading], rise_ft: float = 0.5) -> int:
    """Count distinct rises: 6-hour windows whose max-min exceeds rise_ft, non-overlapping."""
    count, last_peak = 0, None
    for window_start in range(0, len(readings), 1):
        t0 = readings[window_start].time
        window = [r for r in readings[window_start:] if r.time <= t0 + timedelta(hours=6)]
        if not window:
            continue
        spread = max(r.value_ft for r in window) - min(r.value_ft for r in window)
        if spread >= rise_ft and (last_peak is None or t0 - last_peak > timedelta(hours=24)):
            count += 1
            last_peak = t0
    return count


def _interval_min(frames: list[FrameRef]) -> float:
    if len(frames) < 2:
        return float("inf")
    spans = [b.time - a.time for a, b in pairwise(frames)]
    return float(min(s.total_seconds() / 60 for s in spans))


def score_one(client: httpx.Client, camera: nims.CameraInfo, now: datetime) -> SiteScore | None:
    assert camera.nwis_id is not None
    recent = nims.list_frames(client, camera, limit=3, newest_first=True)
    interval = _interval_min(sorted(recent, key=lambda f: f.time))
    if interval > MAX_INTERVAL_MIN:
        return SiteScore(camera.cam_id, camera.nwis_id, camera.name, 0, 0, interval, 0, "slow")
    readings = gage.continuous(client, camera.nwis_id, now - timedelta(days=365), now)
    if not readings:
        return SiteScore(camera.cam_id, camera.nwis_id, camera.name, 0, 0, interval, 0, "no data")
    values = [r.value_ft for r in readings]
    span = max(values) - min(values)
    events = _events(readings)
    if span < MIN_RANGE_FT:
        return SiteScore(
            camera.cam_id, camera.nwis_id, camera.name, span, events, interval, 0, "flat"
        )
    score = (
        0.55 * min(span / 6.0, 1.0)
        + 0.35 * min(events / 6.0, 1.0)
        + 0.10 * (15 / interval if interval else 0)
    )
    return SiteScore(camera.cam_id, camera.nwis_id, camera.name, span, events, interval, score)


def contact_sheet(
    client: httpx.Client,
    cameras: dict[str, nims.CameraInfo],
    cam_ids: list[str],
    out_path: Path,
) -> None:
    tiles = []
    for cam_id in cam_ids:
        camera = cameras[cam_id]
        frame = nims.list_frames(client, camera, limit=1, newest_first=True)
        if not frame:
            continue
        tile = nims.fetch_image(client, frame[0], out_path.parent / "cache")
        tile = cv2.resize(tile, (640, round(640 * tile.shape[0] / tile.shape[1])))
        cv2.putText(tile, cam_id.replace("_Staff", ""), (8, 24), 0, 0.6, (0, 0, 255), 2)
        tiles.append(tile)
    if not tiles:
        raise UsgsError("no frames to make a contact sheet")
    h = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles]
    cols = 4
    rows = [tiles[i : i + cols] for i in range(0, len(tiles), cols)]
    sheet = np.vstack(
        [np.hstack(r + [np.zeros((h, 640, 3), np.uint8)] * (cols - len(r))) for r in rows]
    )
    cv2.imwrite(str(out_path), sheet)


def run_screen(out_json: Path, sheet_path: Path) -> list[SiteScore]:
    client = make_client()
    cameras = {c.cam_id: c for c in nims.list_cameras(client)}
    staff = [c for c in cameras.values() if is_staff_camera(c)]
    now = datetime.now(UTC)
    scores = []
    for camera in sorted(staff, key=lambda c: c.cam_id):
        try:
            result = score_one(client, camera, now)
        except UsgsError as error:
            result = SiteScore(
                camera.cam_id, camera.nwis_id or "", camera.name, 0, 0, 0, 0, str(error)[:60]
            )
        if result:
            scores.append(result)
    scores.sort(key=lambda s: -s.score)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps([asdict(s) for s in scores], indent=1))
    contact_sheet(client, cameras, [s.cam_id for s in scores[:CONTACT_LIMIT]], sheet_path)
    return scores
