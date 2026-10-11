"""The production poll: one tick of the monitoring loop per site.

fetch newest frame -> agentic read -> compare to gage -> flag engine -> persist.
Everything the local replay does, plus a gage call and a store write. The same
code runs on a laptop cron, a Lambda schedule, or in replay over a fixture dir.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from waterline import sites
from waterline.agent.flag import FlagEngine
from waterline.agent.history import CameraHistory
from waterline.agent.loop import run_frame
from waterline.agent.tools import FrameNavigator
from waterline.eval import make_perception
from waterline.models import FrameRef, SiteConfig
from waterline.pipeline import Perception
from waterline.settings import Settings
from waterline.store import JsonlStore, write_json
from waterline.usgs import gage, nims
from waterline.usgs.http import make_client

# Per-site camera history: the last accepted camera read, for temporal
# plausibility. Lives for the process; a fresh process starts empty.
_HISTORIES: dict[str, CameraHistory] = {}


@dataclass
class PollResult:
    site_id: str
    frame_file: str | None
    status: str
    stage_ft: float | None
    gage_ft: float | None
    residual_ft: float | None
    flag_id: str | None
    steps: int


def _latest_frame(camera: nims.CameraInfo) -> FrameRef | None:
    frames = nims.list_frames(make_client(), camera, limit=2, newest_first=True)
    return frames[0] if frames else None


def _gage_at(site: SiteConfig, t: datetime) -> float | None:
    readings = gage.continuous(
        make_client(), site.nwis_id, t - timedelta(minutes=30), t + timedelta(minutes=5)
    )
    r = gage.nearest(readings, t, max_gap=timedelta(minutes=15))
    return r.value_ft if r else None


def poll_site(
    site_id: str,
    out: Path,
    last_file: str | None = None,
    engine: FlagEngine | None = None,
) -> PollResult:
    """One poll tick for one site. Persists reading + trace; emits a Flag if due."""
    settings = Settings()
    site = sites.load_site(site_id)
    ctx: Perception = make_perception(site, settings)
    engine = engine or FlagEngine(site_id)
    client = make_client()
    camera = {c.cam_id: c for c in nims.list_cameras(client)}[site.cam_id]

    ref = _latest_frame(camera)
    if ref is None:
        return PollResult(site_id, None, "no_frame", None, None, None, None, 0)
    if ref.filename == last_file:
        return PollResult(site_id, ref.filename, "unchanged", None, None, None, None, 0)

    image = nims.fetch_image(client, ref, settings.data_dir / "cache")
    gage_ft = _gage_at(site, ref.time)
    nav = FrameNavigator(camera, settings.data_dir / "cache")
    history = _HISTORIES.setdefault(site_id, CameraHistory())
    reading, trace = run_frame(ctx, np.asarray(image), ref, nav, history=history)
    history.update(reading)
    flag = engine.feed(reading, gage_ft, trace, settings, out / "events")

    readings = JsonlStore(out / "readings.jsonl")
    readings.append(reading)
    write_json(out / "traces" / f"{trace.trace_id}.json", trace)

    return PollResult(
        site_id=site_id,
        frame_file=ref.filename,
        status=reading.status,
        stage_ft=reading.stage_ft,
        gage_ft=gage_ft,
        residual_ft=(
            None if reading.stage_ft is None or gage_ft is None else reading.stage_ft - gage_ft
        ),
        flag_id=None if flag is None else flag.id,
        steps=len(trace.steps),
    )
