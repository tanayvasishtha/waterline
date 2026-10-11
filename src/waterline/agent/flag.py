"""Turn a persistent camera-vs-sensor disagreement into a Flag for a human.

The flag is the product. Every field is evidence a technician can audit:
the gage value, the camera's estimate, how long the disagreement persisted,
which frames prove it, and a draft explanation.
"""

import itertools
from dataclasses import dataclass, field
from pathlib import Path

from waterline.models import Flag, Reading, Trace
from waterline.settings import Settings
from waterline.store import event_dir, utc_stamp, write_json

from .fault import FaultTracker

_counter = itertools.count(1)


@dataclass
class FlagEngine:
    """Per-site residual tracker plus flag emission."""

    site_id: str
    trackers: dict[str, FaultTracker] = field(default_factory=dict)
    open_flag: Flag | None = None

    def tracker(self) -> FaultTracker:
        return self.trackers.setdefault(self.site_id, FaultTracker())

    def feed(
        self,
        reading: Reading,
        gage_ft: float | None,
        trace: Trace,
        settings: Settings,
        out: Path,
    ) -> Flag | None:
        """Update the residual tracker; emit a Flag when persistence is proven."""
        residual = (
            reading.stage_ft - gage_ft
            if reading.stage_ft is not None and gage_ft is not None
            else None
        )
        kind = self.tracker().feed(residual, gage_ft, settings)
        if kind is None:
            return None
        flag = Flag(
            id=f"flag-{self.site_id}-{utc_stamp()}-{next(_counter)}",
            site_id=self.site_id,
            opened_at=reading.time,
            latest_seen=reading.time,
            event_type=kind if kind in ("drift", "stuck", "step", "spike") else "disagreement",
            severity="alert" if kind == "stuck" else "watch",
            residual_ft=residual or 0.0,
            gage_ft=gage_ft,
            camera_stage_ft=reading.stage_ft,
            camera_sigma_ft=reading.sigma_ft,
            persistence_frames=settings.flag_persistence_frames,
            evidence_frames=[trace.frame_file],
            draft_text=_draft(kind, reading, gage_ft, residual),
            trace_id=trace.trace_id,
        )
        self.open_flag = flag
        write_json(event_dir(out, self.site_id, reading.time) / "flag.json", flag)
        return flag


def _draft(kind: str, reading: Reading, gage_ft: float | None, residual: float | None) -> str:
    g = f"{gage_ft:.2f} ft" if gage_ft is not None else "no data"
    c = f"{reading.stage_ft:.2f} ft" if reading.stage_ft is not None else "unreadable"
    r = f"{residual:+.2f} ft" if residual is not None else "n/a"
    if kind == "stuck":
        return (
            f"The gage reported the same value for several consecutive frames "
            f"while the camera kept changing. Gage: {g}. Camera: {c}. "
            "Suspect a frozen or disconnected sensor."
        )
    if kind == "step":
        return (
            f"The gage jumped by more than the plausible rate between frames. "
            f"Gage: {g}. Camera: {c}. Residual: {r}."
        )
    return (
        f"The camera and the gage disagree persistently beyond the calibrated "
        f"band. Gage: {g}. Camera: {c}. Residual: {r}. "
        "Suspect sensor drift or a shifted staff plate."
    )
