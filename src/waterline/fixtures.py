"""Record committed test fixtures: downscaled USGS frames plus gage truth at each frame time.

Runs once by a maintainer (`waterline replay-record`); tests then run offline.
USGS imagery is public domain (see NOTICE).
"""

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

from waterline.models import FrameRef
from waterline.usgs import gage, nims
from waterline.usgs.http import UsgsError, make_client

FIXTURE_WIDTH = 1280
JPEG_QUALITY = 85


@dataclass(frozen=True)
class FixtureEntry:
    site_id: str
    cam_id: str
    filename: str
    time: datetime
    gage_ft: float | None
    path: str


def downscale(image: np.ndarray, width: int = FIXTURE_WIDTH) -> np.ndarray:
    if image.shape[1] <= width:
        return image
    scale = width / image.shape[1]
    return cv2.resize(image, (width, round(image.shape[0] * scale)))


def _near_time(frames: list[FrameRef], want: datetime) -> FrameRef:
    return min(frames, key=lambda f: abs(f.time - want))


def record(
    site_id: str,
    cam_id: str,
    nwis_id: str,
    wanted: list[datetime],
    out_dir: Path,
) -> list[FixtureEntry]:
    """Download the frame nearest each wanted time, plus gage truth, under out_dir."""
    client = make_client()
    camera = next(c for c in nims.list_cameras(client) if c.cam_id == cam_id)
    frames_dir = out_dir / "frames" / cam_id
    frames_dir.mkdir(parents=True, exist_ok=True)

    lo = min(wanted) - timedelta(hours=2)
    hi = max(wanted) + timedelta(hours=2)
    all_frames = nims.list_frames(client, camera, after=lo, before=hi, limit=5000)
    if not all_frames:
        raise UsgsError(f"no NIMS frames for {cam_id} in {lo}..{hi}")
    readings = gage.continuous(client, nwis_id, lo, hi)

    entries = []
    for want in wanted:
        frame = _near_time(all_frames, want)
        image = nims.fetch_image(client, frame, out_dir / "cache")
        small = downscale(image)
        rel = f"frames/{cam_id}/{frame.filename}"
        out_path = out_dir / rel
        if not cv2.imwrite(str(out_path), small, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]):
            raise UsgsError(f"imwrite failed for {out_path}")
        truth = gage.nearest(readings, frame.time)
        entries.append(
            FixtureEntry(
                site_id, cam_id, frame.filename, frame.time, truth.value_ft if truth else None, rel
            )
        )

    manifest = [
        {
            "site_id": e.site_id,
            "cam_id": e.cam_id,
            "filename": e.filename,
            "time": e.time.isoformat(),
            "gage_ft": e.gage_ft,
            "path": e.path,
        }
        for e in entries
    ]
    manifest_path = out_dir / f"manifest_{site_id}.json"
    existing = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    merged = {row["filename"]: row for row in [*existing, *manifest]}
    manifest_path.write_text(json.dumps(list(merged.values()), indent=1))
    return entries


HANCOCK_WANTED = [
    # dry reference and sunny mid-range frame
    datetime(2026, 7, 27, 18, 0, tzinfo=UTC),
    datetime(2026, 7, 28, 15, 0, tzinfo=UTC),
    # flood rise, near peak, and recession
    datetime(2026, 7, 29, 16, 0, tzinfo=UTC),
    datetime(2026, 7, 30, 10, 0, tzinfo=UTC),
    datetime(2026, 7, 30, 18, 0, tzinfo=UTC),
    datetime(2026, 7, 31, 14, 0, tzinfo=UTC),
    # night (IR) frame
    datetime(2026, 7, 29, 6, 0, tzinfo=UTC),
]
