"""Turn a proven camera-vs-sensor disagreement into a Flag for a human.

The flag is the product. Every field is evidence a technician can audit:
the gage value, the camera's estimate, which frames prove it, and a draft
explanation that says only what the rule actually checked.
"""

import itertools
from dataclasses import dataclass, field
from pathlib import Path

from waterline.models import Flag, Reading, Trace
from waterline.settings import Settings
from waterline.store import event_dir, utc_stamp, write_json

from .fault import FaultEvent, FaultTracker, Obs

_counter = itertools.count(1)


@dataclass
class FlagEngine:
    """Per-site residual tracker plus flag emission."""

    site_id: str
    tracker: FaultTracker = field(default_factory=FaultTracker)
    open_flag: Flag | None = None

    def feed(
        self,
        reading: Reading,
        gage_ft: float | None,
        trace: Trace,
        settings: Settings,
        out: Path | None,
    ) -> Flag | None:
        """Update the fault tracker; emit a Flag when a rule proves its case."""
        obs = Obs(
            time=reading.time,
            file=trace.frame_file,
            gage_ft=gage_ft,
            camera_ft=reading.stage_ft if reading.status == "accepted" else None,
            sigma_ft=reading.sigma_ft,
        )
        ev = self.tracker.feed(obs, settings)
        if ev is None:
            return None
        flag = Flag(
            id=f"flag-{self.site_id}-{utc_stamp()}-{next(_counter)}",
            site_id=self.site_id,
            opened_at=ev.time,
            latest_seen=reading.time,
            event_type=ev.kind,
            severity="alert" if ev.kind == "stuck" else "watch",
            residual_ft=ev.residual_ft or 0.0,
            gage_ft=ev.gage_ft,
            camera_stage_ft=ev.camera_ft,
            camera_sigma_ft=ev.sigma_ft,
            persistence_frames=ev.persistence_frames,
            evidence_frames=[ev.file],
            draft_text=_draft(ev),
            trace_id=trace.trace_id,
        )
        self.open_flag = flag
        if out is not None:
            write_json(event_dir(out, self.site_id, ev.time) / "flag.json", flag)
        return flag


def _draft(ev: FaultEvent) -> str:
    g = f"{ev.gage_ft:.2f} ft" if ev.gage_ft is not None else "no data"
    c = f"{ev.camera_ft:.2f} ft" if ev.camera_ft is not None else "no accepted read"
    r = f"{ev.residual_ft:+.2f} ft" if ev.residual_ft is not None else "n/a"
    if ev.kind == "spike":
        return (
            "The gage made a one-frame reversal beyond the plausible swing and "
            f"the next frame did not carry the move. Gage: {g}. "
            "Suspect a sensor spike or telemetry glitch."
        )
    if ev.kind == "stuck":
        return (
            f"The gage repeated the same value for {ev.persistence_frames} "
            "consecutive frames while the camera's reads moved beyond their "
            f"noise band. Gage: {g}. Camera: {c}. "
            "Suspect a frozen or disconnected sensor."
        )
    if ev.kind == "step":
        return (
            "The gage moved differently from what the camera saw between "
            "accepted reads: the disagreement in the two changes exceeded "
            f"3 combined sigmas. Gage: {g}. Camera: {c}. Residual: {r}."
        )
    return (
        "The camera-minus-gage residual accumulated beyond 5 sigmas to one "
        f"side across recent accepted reads. Gage: {g}. Camera: {c}. "
        f"Residual: {r}. Suspect sensor drift or a shifted staff plate."
    )
