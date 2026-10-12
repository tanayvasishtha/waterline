"""Replay the agent over a dev window and dump each frame's final decision.

Mirrors eval.run_window's agent path: same frame selection, same
CameraHistory threading, the same live FrameNavigator. Images come from the
local cache; the only API traffic is frame listing.
"""

import json
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar

from waterline import sites
from waterline.agent.history import CameraHistory
from waterline.agent.loop import run_frame
from waterline.agent.tools import FrameNavigator
from waterline.eval import frame_list, make_perception
from waterline.settings import Settings
from waterline.usgs import nims
from waterline.usgs.http import UsgsError, make_client

F = TypeVar("F", bound=Callable[..., Any])


def _patient(fn: F) -> F:
    def wrapped(*a, **k):  # type: ignore[no-untyped-def]
        for attempt in range(90):
            try:
                return fn(*a, **k)
            except UsgsError as e:
                msg = str(e)
                if (
                    "rate limited" not in msg
                    and "ConnectError" not in msg
                    and "failed after retries" not in msg
                ):
                    raise
                print(f"    retry 65s (#{attempt + 1})", flush=True)
                time.sleep(65)
        raise UsgsError("limit stuck")

    return wrapped  # type: ignore[return-value]


nims.list_frames = _patient(nims.list_frames)
nims.list_cameras = _patient(nims.list_cameras)


def main() -> None:
    site_id, start, end, every_n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    site = sites.load_site(site_id)
    ctx = make_perception(site, Settings())
    client = make_client()
    camera = {c.cam_id: c for c in nims.list_cameras(client)}[site.cam_id]
    nav = FrameNavigator(camera, Path("data/cache"))
    start_dt = datetime.fromisoformat(start).replace(tzinfo=UTC)
    end_dt = datetime.fromisoformat(end).replace(tzinfo=UTC)
    frames = frame_list(site, start_dt, end_dt, every_n)
    history = CameraHistory()
    cache = Path("data/cache")
    out = []
    for ref in frames:
        image = nims.fetch_image(client, ref, cache)
        if image is None:
            continue
        reading, trace = run_frame(ctx, image, ref, nav, history=history)
        history.update(reading)
        out.append(
            {
                "time": ref.time.isoformat(),
                "file": ref.filename,
                "status": reading.status,
                "stage_ft": reading.stage_ft,
                "reading_time": reading.time.isoformat(),
                "reading_file": reading.frame_file,
                "last_step": (
                    {"tool": trace.steps[-1].tool, "decision": trace.steps[-1].decision}
                    if trace.steps
                    else None
                ),
                "reason": trace.reason,
            }
        )
    stamp = f"dev_trace_{site_id}_{start_dt.date()}_{end_dt.date()}"
    Path("results", f"{stamp}.json").write_text(json.dumps(out, indent=1))
    print(f"wrote results/{stamp}.json: {len(out)} frames")


if __name__ == "__main__":
    main()
