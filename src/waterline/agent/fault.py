"""Sensor-fault detection over camera-minus-gage residuals.

Four rules run per frame, in a fixed order: spike, stuck, step, drift. The
first rule whose condition holds emits; every rule still updates its state.
A rule emits once on the rising edge of its condition and re-arms only after
the condition goes false on a later frame (drift re-arms through its CUSUM
reset). The camera only counts when it produced an accepted numeric read:
a camera that is dark, blurred or below range cannot testify either way.
"""

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from waterline.settings import Settings

FaultKind = Literal["spike", "stuck", "step", "drift"]
FAULT_KINDS: tuple[FaultKind, ...] = ("spike", "stuck", "step", "drift")


@dataclass
class Obs:
    """One frame's evidence for the fault rules."""

    time: datetime
    file: str
    gage_ft: float | None
    camera_ft: float | None
    sigma_ft: float | None


@dataclass
class FaultEvent:
    """What a rule proved on one frame."""

    kind: FaultKind
    time: datetime
    file: str
    gage_ft: float | None
    camera_ft: float | None
    sigma_ft: float | None
    residual_ft: float | None
    persistence_frames: int


@dataclass
class FaultTracker:
    """Per-site state: gage history, CUSUM accumulators, previous accepted pair."""

    g_pos: float = 0.0
    g_neg: float = 0.0
    since_reset: int = 0
    hist: list[Obs] = field(default_factory=list)
    prev_accepted: Obs | None = None
    fired: dict[FaultKind, bool] = field(default_factory=lambda: {k: False for k in FAULT_KINDS})

    def feed(self, obs: Obs, settings: Settings) -> FaultEvent | None:
        """Update every rule's state with this frame; emit the first true one."""
        if obs.gage_ft is not None:
            self.hist.append(obs)
        events: dict[FaultKind, FaultEvent | None] = {
            "spike": self._spike(settings),
            "stuck": self._stuck(settings),
            "step": self._step(obs, settings),
            "drift": self._drift(obs, settings),
        }
        for kind in FAULT_KINDS:
            ev = events[kind]
            if self._edge(kind, ev is not None):
                return ev
        return None

    def _edge(self, kind: FaultKind, condition: bool) -> bool:
        """Rising-edge latch: true once per episode, re-arms when it clears."""
        if condition and not self.fired[kind]:
            self.fired[kind] = True
            return True
        if not condition:
            self.fired[kind] = False
        return False

    def _spike(self, settings: Settings) -> FaultEvent | None:
        """A one-frame reversal in the gage series that does not carry through."""
        if len(self.hist) < 3:
            return None
        t2, t1, t0 = self.hist[-3:]
        g2, g1, g0 = t2.gage_ft, t1.gage_ft, t0.gage_ft
        assert g2 is not None and g1 is not None and g0 is not None
        e = g1 - (g2 + g0) / 2
        if abs(e) <= settings.spike_min_ft or abs(g0 - g2) >= abs(e) / 2:
            return None
        return FaultEvent(
            kind="spike",
            time=t1.time,
            file=t1.file,
            gage_ft=g1,
            camera_ft=t1.camera_ft,
            sigma_ft=t1.sigma_ft,
            residual_ft=None if t1.camera_ft is None else t1.camera_ft - g1,
            persistence_frames=1,
        )

    def _stuck(self, settings: Settings) -> FaultEvent | None:
        """The gage repeats one value while the camera provably keeps moving."""
        h = self.hist[-settings.stuck_frames :]
        if len(h) < settings.stuck_frames or len({o.gage_ft for o in h}) != 1:
            return None
        cams = [o for o in h if o.camera_ft is not None and o.sigma_ft is not None]
        if len(cams) < 2:
            return None
        first, last_cam = cams[0], cams[-1]
        assert (
            first.camera_ft is not None
            and first.sigma_ft is not None
            and last_cam.camera_ft is not None
            and last_cam.sigma_ft is not None
        )
        s = math.sqrt(first.sigma_ft**2 + last_cam.sigma_ft**2)
        if abs(last_cam.camera_ft - first.camera_ft) <= settings.stuck_camera_z * s:
            return None
        last = h[-1]
        return FaultEvent(
            kind="stuck",
            time=last.time,
            file=last.file,
            gage_ft=last.gage_ft,
            camera_ft=last.camera_ft,
            sigma_ft=last.sigma_ft,
            residual_ft=(
                None
                if last.camera_ft is None or last.gage_ft is None
                else last.camera_ft - last.gage_ft
            ),
            persistence_frames=settings.stuck_frames,
        )

    def _step(self, obs: Obs, settings: Settings) -> FaultEvent | None:
        """The gage jumps by more than the camera agrees it should have."""
        prev = self.prev_accepted
        if obs.gage_ft is not None and obs.camera_ft is not None and obs.sigma_ft is not None:
            self.prev_accepted = obs
        if (
            prev is None
            or obs.gage_ft is None
            or obs.camera_ft is None
            or obs.sigma_ft is None
            or prev.gage_ft is None
            or prev.camera_ft is None
            or prev.sigma_ft is None
        ):
            return None
        d = (obs.gage_ft - prev.gage_ft) - (obs.camera_ft - prev.camera_ft)
        s = math.sqrt(obs.sigma_ft**2 + prev.sigma_ft**2)
        if abs(d) <= settings.step_z * s:
            return None
        return FaultEvent(
            kind="step",
            time=obs.time,
            file=obs.file,
            gage_ft=obs.gage_ft,
            camera_ft=obs.camera_ft,
            sigma_ft=obs.sigma_ft,
            residual_ft=obs.camera_ft - obs.gage_ft,
            persistence_frames=1,
        )

    def _drift(self, obs: Obs, settings: Settings) -> FaultEvent | None:
        """Standardised CUSUM: one-sided accumulation of sigma-scaled residuals."""
        if obs.gage_ft is None or obs.camera_ft is None or not obs.sigma_ft:
            return None
        z = (obs.camera_ft - obs.gage_ft) / obs.sigma_ft
        self.since_reset += 1
        self.g_pos = max(0.0, self.g_pos + z - settings.cusum_k)
        self.g_neg = max(0.0, self.g_neg - z - settings.cusum_k)
        if max(self.g_pos, self.g_neg) <= settings.cusum_h:
            return None
        persistence = self.since_reset
        self.g_pos = 0.0
        self.g_neg = 0.0
        self.since_reset = 0
        return FaultEvent(
            kind="drift",
            time=obs.time,
            file=obs.file,
            gage_ft=obs.gage_ft,
            camera_ft=obs.camera_ft,
            sigma_ft=obs.sigma_ft,
            residual_ft=obs.camera_ft - obs.gage_ft,
            persistence_frames=persistence,
        )
