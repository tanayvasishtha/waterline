"""The agentic loop: perceive -> decide -> act, bounded and fully traced.

The policy sees only the frame's evidence. It never sees the gage value, so a
decision cannot be accused of fitting the answer.
"""

import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

from waterline.models import AgentAction, AgentStep, FrameRef, Reading, Trace
from waterline.pipeline import FrameResult, Perception, read_frame
from waterline.settings import Settings

from .policy import Decision, LoopState, decide
from .tools import ENHANCE_OPS, FrameNavigator

Chooser = Callable[[FrameResult, LoopState, Settings], Decision]


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
    chooser: Chooser = decide,
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

    while state.steps < settings.agent_step_budget:
        decision = chooser(result, state, settings)
        state.tried.add(decision.action)
        if decision.terminal:
            break
        t0 = time.perf_counter()
        applied = False
        state_op = None
        if decision.action in ENHANCE_OPS:
            state_op = ENHANCE_OPS[decision.action]
            applied = True
        elif decision.action == AgentAction.ALT_REFERENCE:
            if result.registration is not None:
                excluded.add(result.registration.ref_name)
            applied = True
        elif decision.action in (AgentAction.FETCH_PREV, AgentAction.FETCH_NEXT):
            if nav is None:
                state.steps += 1
                continue
            direction = -1 if decision.action == AgentAction.FETCH_PREV else 1
            next_ref = nav.neighbor(current_ref.time, direction)
            if next_ref is not None:
                current_ref = next_ref
                image = nav.image(next_ref)
                applied = True
        if applied:
            result = read_frame(
                ctx,
                image,
                current_ref.time,
                enhance_op=state_op,
                seed=0,
                exclude_refs=frozenset(excluded),
            )
        ms = int((time.perf_counter() - t0) * 1000)
        r = result.reading
        steps.append(
            AgentStep(
                step=len(steps) + 1,
                tool=decision.action.value,
                args={},
                result_summary=f"stage={r.stage_ft} sigma={r.sigma_ft} reason={r.reason}",
                decision=decision.reason,
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
    )
    trace = Trace(
        trace_id=f"{ctx.site.site_id}-{ref.time.strftime('%Y%m%dT%H%MZ')}",
        site_id=ctx.site.site_id,
        frame_time=ref.time,
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
