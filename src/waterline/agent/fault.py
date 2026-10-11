"""Sensor-fault detection: CUSUM on camera-minus-gage residuals.

CUSUM (cumulative sum) accumulates drift evidence. One-sided: a slowly drifting
transducer is caught before it becomes a disaster. Stuck sensors are caught by
identical consecutive gage values.
"""

from dataclasses import dataclass, field

from waterline.settings import Settings


@dataclass
class CusumState:
    """One site's accumulated drift evidence."""

    g_pos: float = 0.0
    g_neg: float = 0.0
    over: int = 0


def update(state: CusumState, residual_ft: float, settings: Settings) -> CusumState:
    """Feed one (camera minus gage) residual. Returns the updated state."""
    k = settings.cusum_k
    state.g_pos = max(0.0, state.g_pos + residual_ft - k * 0.1)
    state.g_neg = min(0.0, state.g_neg + residual_ft + k * 0.1)
    if abs(state.g_pos) > settings.cusum_h * 0.1 or abs(state.g_neg) > settings.cusum_h * 0.1:
        state.over += 1
    else:
        state.over = 0
    return state


def drift_flagged(state: CusumState, settings: Settings) -> bool:
    return state.over >= settings.flag_persistence_frames


def is_stuck(values: list[float]) -> bool:
    """A sensor that reports the exact same value repeatedly is frozen."""
    return len(values) >= 4 and len(set(values[-4:])) == 1


def is_step(prev: float | None, now: float, threshold_ft: float = 0.8) -> bool:
    return prev is not None and abs(now - prev) > threshold_ft


@dataclass
class FaultTracker:
    """Per-site state: CUSUM plus a short gage history."""

    cusum: CusumState = field(default_factory=CusumState)
    gage_history: list[float] = field(default_factory=list)

    def feed(
        self, residual_ft: float | None, gage_ft: float | None, settings: Settings
    ) -> str | None:
        """Returns the fault type to flag, or None."""
        if gage_ft is not None:
            self.gage_history.append(gage_ft)
        if is_stuck(self.gage_history):
            return "stuck"
        if residual_ft is None:
            return None
        update(self.cusum, residual_ft, settings)
        if drift_flagged(self.cusum, settings):
            return "drift"
        if len(self.gage_history) >= 2 and is_step(self.gage_history[-2], gage_ft or 0.0):
            return "step"
        return None
