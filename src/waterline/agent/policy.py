"""Deterministic triage policy: the floor the LLM variant must beat.

The policy only sees what the frame shows (quality metrics, sigma) plus what
has already been tried. It never sees the gage.

Rules are data: the first rule whose condition holds and whose action is
untried wins. This is deliberately readable -- the video and the report show
this table.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

from waterline.models import AgentAction
from waterline.pipeline import FrameResult
from waterline.quality import quality_ok
from waterline.settings import Settings


@dataclass
class LoopState:
    """What the loop has tried so far."""

    tried: set[AgentAction] = field(default_factory=set)
    steps: int = 0
    alt_refs: int = 0
    self_inconsistent: bool = False  # a repair changed the reading's answer
    implausible: bool = False  # the read cannot be reconciled with recent history


@dataclass(frozen=True)
class Decision:
    action: AgentAction
    reason: str

    @property
    def terminal(self) -> bool:
        return self.action in {
            AgentAction.ACCEPT,
            AgentAction.ABSTAIN,
            AgentAction.HUMAN_REVIEW,
        }


def _reg_failed(r: FrameResult) -> bool:
    return r.registration is None


def _night(r: FrameResult) -> bool:
    return r.quality.is_night


def _below(r: FrameResult) -> bool:
    return r.reading.reason == "below_plate"


def _no_reading(r: FrameResult) -> bool:
    return r.detection is None or r.reading.stage_ft is None


def _glare(r: FrameResult, s: Settings) -> bool:
    return r.quality.glare_frac > s.glare_action_frac


def _blur(r: FrameResult, s: Settings) -> bool:
    return r.quality.blur_var < s.blur_var_min


def _uncertain(r: FrameResult, s: Settings) -> bool:
    """A wide bootstrap sigma. Method disagreement is recorded on the reading
    but does not gate acceptance: on the calibration windows, method B's value
    carried no information about method A's error. The calibration floor in
    sigma_ft is measurement honesty, not indecision, so it does not count."""
    return r.detection is not None and r.detection.sigma_ft > s.accept_sigma_ft


def _confirmable(st: LoopState) -> bool:
    """Neighbour fetches only help an inconsistent or implausible read. A
    sigma-uncertain frame's own measurement is noisy, and a neighbour
    agreeing with noise confirms nothing."""
    return st.self_inconsistent or st.implausible


def _flagged(r: FrameResult, st: LoopState, s: Settings) -> bool:
    """Sigma-uncertain, a changed-answer repair, or a read that does not
    reconcile with recent history."""
    return st.self_inconsistent or st.implausible or _uncertain(r, s)


def _ok(r: FrameResult, s: Settings) -> bool:
    return quality_ok(r.quality, s)


# (condition, action, reason). A rule fires only if the action is untried.
RULES: tuple[tuple[Callable[[FrameResult, LoopState, Settings], bool], AgentAction, str], ...] = (
    # registration failed: night -> neighbor; day -> one contrast lift -> neighbor
    (
        lambda r, st, s: _reg_failed(r) and _night(r),
        AgentAction.FETCH_NEXT,
        "night frame cannot register; pull the next frame",
    ),
    (
        lambda r, st, s: _reg_failed(r),
        AgentAction.ENHANCE_GAMMA,
        "registration failed; try one contrast lift",
    ),
    (
        lambda r, st, s: _reg_failed(r),
        AgentAction.FETCH_NEXT,
        "still cannot register; pull the next frame",
    ),
    # a frame that registers but is night: pull a day frame rather than guess
    (
        lambda r, st, s: _night(r),
        AgentAction.FETCH_NEXT,
        "night frame has no readable ticks; fetch a day frame",
    ),
    # below-range is a valid reading, not a failure
    (
        lambda r, st, s: _below(r),
        AgentAction.ACCEPT,
        "water surface is below the plate",
    ),
    # nothing found: contrast, then neighbor
    (
        lambda r, st, s: _no_reading(r),
        AgentAction.ENHANCE_CLAHE,
        "no boundary found; try local contrast",
    ),
    (
        lambda r, st, s: _no_reading(r),
        AgentAction.FETCH_NEXT,
        "still no boundary; pull the next frame",
    ),
    # quality issues on a frame that did read: targeted repair
    (
        lambda r, st, s: _glare(r, s),
        AgentAction.MASK_GLARE,
        "glare covers the plate; try inpainting",
    ),
    (
        lambda r, st, s: _blur(r, s),
        AgentAction.ENHANCE_CLAHE,
        "soft frame; try a sharpening pass",
    ),
    # uncertainty: other reference, region re-check, temporal neighbors, human
    (
        lambda r, st, s: _flagged(r, st, s) and st.alt_refs > 1,
        AgentAction.ALT_REFERENCE,
        "uncertain; re-register on the other reference",
    ),
    (
        lambda r, st, s: _flagged(r, st, s) and _confirmable(st),
        AgentAction.FETCH_PREV,
        "uncertain; compare the previous frame",
    ),
    (
        lambda r, st, s: _flagged(r, st, s) and _confirmable(st),
        AgentAction.FETCH_NEXT,
        "uncertain; compare the next frame",
    ),
    # done: quality passes AND the reading is not fighting itself
    (
        lambda r, st, s: _ok(r, s) and not _flagged(r, st, s),
        AgentAction.ACCEPT,
        "within sigma",
    ),
)


def decide(result: FrameResult, state: LoopState, settings: Settings) -> Decision:
    for condition, action, reason in RULES:
        if action not in state.tried and condition(result, state, settings):
            return Decision(action, reason)
    if _flagged(result, state, settings):
        if _uncertain(result, settings) and not _confirmable(state):
            sig = result.detection.sigma_ft if result.detection is not None else 0.0
            return Decision(
                AgentAction.HUMAN_REVIEW,
                f"bootstrap sigma {sig:.2f} above {settings.accept_sigma_ft:.2f} after repairs",
            )
        return Decision(AgentAction.HUMAN_REVIEW, "uncertainty persists after all re-reads")
    return Decision(AgentAction.ABSTAIN, "no repair left to try")
