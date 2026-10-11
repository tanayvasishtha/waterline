"""The agentic loop: perceive -> decide -> act, bounded and fully traced.

The policy sees only the frame's evidence. It never sees the gage value, so a
decision cannot be accused of fitting the answer.
"""

import math
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

from waterline.models import AgentAction, AgentStep, FrameRef, Reading, Trace
from waterline.pipeline import FrameResult, Perception, read_frame
from waterline.settings import Settings

from .history import CameraHistory
from .policy import Decision, LoopState, decide
from .tools import ENHANCE_OPS, FrameNavigator

Chooser = Callable[[FrameResult, LoopState, Settings], Decision]

REPAIR_ACTIONS = frozenset(ENHANCE_OPS) | {AgentAction.ALT_REFERENCE}


def _reconcile(
    old: FrameResult, new: FrameResult, action: AgentAction
) -> tuple[FrameResult, str | None, bool, float | None]:
    """Repair self-consistency: adopt r_new or keep r_old.

    A repair that moves a read more than 3 combined sigmas changed the answer:
    keep r_old and mark the frame uncertain. With no read on r_old, adopt r_new.
    Otherwise adopt only a repair that does not widen the bootstrap sigma.
    Returns (result, rejection reason, frame-uncertain flag, |stage delta|).
    """
    if action not in REPAIR_ACTIONS:
        return new, None, False, None
    o, n = old.reading, new.reading
    if o.stage_ft is None:
        return new, None, False, None
    if n.stage_ft is None:
        return old, "repair rejected: lost the reading", True, None
    delta = abs(n.stage_ft - o.stage_ft)
    s = math.sqrt((o.sigma_ft or 0.0) ** 2 + (n.sigma_ft or 0.0) ** 2)
    if delta > 3.0 * s:
        return old, f"repair rejected: changed the reading by {delta:.2f} ft", True, delta
    if (n.sigma_ft or 0.0) > (o.sigma_ft or 0.0):
        return old, "repair rejected: no gain", False, delta
    return new, None, False, delta


def _plausibility(
    history: CameraHistory | None,
    result: FrameResult,
    t: datetime,
    rate_max: float | None,
) -> float | None:
    """The camera-vs-history jump, when the new reading is implausibly far."""
    if history is None or result.reading.stage_ft is None:
        return None
    return history.jump_beyond(result.reading.stage_ft, result.reading.sigma_ft, t, rate_max)


def _apply(
    decision: Decision,
    image: np.ndarray,
    current_ref: FrameRef,
    result: FrameResult,
    *,
    excluded: set[str],
    nav: FrameNavigator | None,
) -> tuple[np.ndarray, FrameRef, str | None, bool]:
    """Run one non-terminal action; returns (image, ref, enhance_op, applied)."""
    if decision.action in ENHANCE_OPS:
        return image, current_ref, ENHANCE_OPS[decision.action], True
    if decision.action == AgentAction.ALT_REFERENCE:
        if result.registration is not None:
            excluded.add(result.registration.ref_name)
        return image, current_ref, None, True
    if decision.action in (AgentAction.FETCH_PREV, AgentAction.FETCH_NEXT):
        if nav is None:
            return image, current_ref, None, False
        direction = -1 if decision.action == AgentAction.FETCH_PREV else 1
        next_ref = nav.neighbor(current_ref.time, direction)
        if next_ref is not None:
            return nav.image(next_ref), next_ref, None, True
    return image, current_ref, None, False


def _status_for(result: FrameResult, action: AgentAction) -> tuple[str, str | None]:
    if action == AgentAction.ACCEPT:
        return (
            ("below_range", result.reading.reason)
            if result.reading.reason == "below_plate"
            else ("accepted", result.reading.reason)
        )
    if action == AgentAction.HUMAN_REVIEW:
        return "pending", "escalated for human review"
    return "abstained", "no confident reading after the action budget"


def run_frame(
    ctx: Perception,
    image: np.ndarray,
    ref: FrameRef,
    nav: FrameNavigator | None = None,
    *,
    chooser: Chooser = decide,
    history: CameraHistory | None = None,
) -> tuple[Reading, Trace]:
    """One full agentic pass over a frame. Returns the final reading and its trace."""
    settings = ctx.settings
    state = LoopState(alt_refs=len(ctx.bank.descriptors))
    current_ref = ref
    excluded: set[str] = set()
    result = read_frame(ctx, image, ref.time, seed=0)
    started = datetime.now(UTC)
    steps: list[AgentStep] = []
    decision = Decision(AgentAction.ABSTAIN, "loop never ran")
    repair_rejected = False
    jump = _plausibility(history, result, current_ref.time, ctx.site.rate_max_ft_per_h)
    state.forced_uncertain = jump is not None

    while state.steps < settings.agent_step_budget:
        decision = chooser(result, state, settings)
        state.tried.add(decision.action)
        if decision.terminal:
            break
        if nav is None and decision.action in (
            AgentAction.FETCH_PREV,
            AgentAction.FETCH_NEXT,
        ):
            state.steps += 1
            continue
        t0 = time.perf_counter()
        image, current_ref, state_op, applied = _apply(
            decision, image, current_ref, result, excluded=excluded, nav=nav
        )
        rejection = None
        delta = None
        if applied:
            new_result = read_frame(
                ctx,
                image,
                current_ref.time,
                enhance_op=state_op,
                seed=0,
                exclude_refs=frozenset(excluded),
            )
            result, rejection, rejected, delta = _reconcile(result, new_result, decision.action)
            repair_rejected = repair_rejected or rejected
            jump = _plausibility(history, result, current_ref.time, ctx.site.rate_max_ft_per_h)
            state.forced_uncertain = repair_rejected or jump is not None
        ms = int((time.perf_counter() - t0) * 1000)
        r = result.reading
        summary = (
            f"stage={r.stage_ft} sigma={r.sigma_ft} disagree={r.methods_disagree} reason={r.reason}"
        )
        if delta is not None:
            summary += f" dR={delta:.2f}"
        if jump is not None:
            summary += f" jump={jump:.2f}"
        steps.append(
            AgentStep(
                step=len(steps) + 1,
                tool=decision.action.value,
                args={},
                result_summary=summary,
                decision=rejection or decision.reason,
                latency_ms=ms,
            )
        )
        state.steps += 1
    else:
        decision = Decision(AgentAction.HUMAN_REVIEW, "action budget exhausted")

    status, reason = _status_for(result, decision.action)
    reading = Reading(
        site_id=ctx.site.site_id,
        time=current_ref.time,
        stage_ft=result.reading.stage_ft,
        sigma_ft=result.reading.sigma_ft,
        method_a_ft=result.reading.method_a_ft,
        method_b_ft=result.reading.method_b_ft,
        methods_disagree=result.reading.methods_disagree,
        status=status,
        reason=reason,
        frame_file=current_ref.filename,
    )
    trace = Trace(
        trace_id=f"{ctx.site.site_id}-{current_ref.time.strftime('%Y%m%dT%H%MZ')}",
        site_id=ctx.site.site_id,
        frame_time=current_ref.time,
        frame_file=current_ref.filename,
        started_at=started,
        finished_at=datetime.now(UTC),
        steps=steps,
        final_status=status,
        final_stage_ft=reading.stage_ft,
        final_sigma_ft=reading.sigma_ft,
        reason=reason,
        policy="deterministic",
    )
    return reading, trace


def replay(
    ctx: Perception,
    frames_dir: Path,
    manifest: list[dict[str, object]],
    nav: FrameNavigator | None = None,
) -> list[tuple[Reading, Trace]]:
    """Fixture/replay path: read committed frames without touching the network."""
    out = []
    for entry in manifest:
        image = cv2.imread(str(frames_dir / str(entry["path"])))
        if image is None:
            continue
        ref = FrameRef(
            cam_id=ctx.site.cam_id,
            filename=str(entry["filename"]),
            time=datetime.fromisoformat(str(entry["time"])),
            url="",
        )
        out.append(run_frame(ctx, image, ref, nav))
    return out
