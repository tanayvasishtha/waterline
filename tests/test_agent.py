"""Agent policy, loop and fault-tracker tests. Fixtures keep them offline."""

import json
import types
from datetime import UTC, datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import pytest

from waterline import sites
from waterline.agent.fault import FaultTracker, Obs
from waterline.agent.flag import FlagEngine
from waterline.agent.loop import run_frame
from waterline.agent.policy import LoopState, decide
from waterline.eval import make_perception
from waterline.models import (
    AgentAction,
    FrameRef,
    QualityReport,
    Reading,
    Trace,
)
from waterline.pipeline import FrameResult
from waterline.settings import Settings


def _quality(**kw: object) -> QualityReport:
    base = {
        "mean_v": 150.0,
        "mean_s": 60.0,
        "blur_var": 500.0,
        "glare_frac": 0.0,
        "contrast": 0.4,
        "plate_visibility": 45.0,
        "is_night": False,
        "registration_inliers": 300,
        "inlier_ratio": 0.7,
        "shift_px": 1.0,
    }
    base.update(kw)
    return QualityReport(**base)


def _result(**kw: object) -> FrameResult:
    """A FrameResult with sane defaults; kw overrides quality/reading fields."""
    qkw = {k[2:]: v for k, v in kw.items() if k.startswith("q_")}
    rkw = {k[2:]: v for k, v in kw.items() if k.startswith("r_")}
    reading = Reading(
        site_id="t",
        time=datetime.now(UTC),
        stage_ft=rkw.get("stage_ft", 10.0),
        sigma_ft=rkw.get("sigma_ft", 0.1),
        status=rkw.get("status", "pending"),
        reason=rkw.get("reason"),
        methods_disagree=rkw.get("methods_disagree", False),
    )
    return FrameResult(
        reading=reading,
        quality=_quality(**qkw),
        registration=types.SimpleNamespace(ref_name="ref1") if kw.get("registered", True) else None,  # type: ignore[arg-type]
        detection=(
            types.SimpleNamespace(sigma_ft=rkw.get("sigma_ft", 0.1))  # type: ignore[arg-type]
            if kw.get("detected", True)
            else None
        ),
        strip=None,
        warped=None,
    )


S = Settings()


def test_policy_accepts_clean_reading() -> None:
    d = decide(_result(), LoopState(), S)
    assert d.action == AgentAction.ACCEPT


def test_policy_below_plate_is_accepted_not_guessed() -> None:
    d = decide(_result(r_stage_ft=None, r_reason="below_plate"), LoopState(), S)
    assert d.action == AgentAction.ACCEPT


def test_policy_night_fetches_a_day_frame() -> None:
    d = decide(_result(q_is_night=True, r_stage_ft=None), LoopState(), S)
    assert d.action == AgentAction.FETCH_NEXT


def test_policy_uncertain_escalates_after_tools() -> None:
    state = LoopState(
        tried={
            AgentAction.ENHANCE_CLAHE,
            AgentAction.MASK_GLARE,
            AgentAction.FETCH_PREV,
            AgentAction.FETCH_NEXT,
            AgentAction.ALT_REFERENCE,
        }
    )
    d = decide(_result(r_sigma_ft=0.9), state, S)
    assert d.action == AgentAction.HUMAN_REVIEW


def test_policy_never_repeats_an_action() -> None:
    state = LoopState(tried={AgentAction.FETCH_NEXT})
    d = decide(_result(q_is_night=True, r_stage_ft=None), state, S)
    assert d.action != AgentAction.FETCH_NEXT


def _obs(i: int, gage: float | None, camera: float | None = None, sigma: float = 0.2) -> Obs:
    return Obs(
        time=datetime(2026, 8, 1, tzinfo=UTC) + timedelta(minutes=15 * i),
        file=f"f{i}.jpg",
        gage_ft=gage,
        camera_ft=camera,
        sigma_ft=sigma,
    )


def test_flood_rise_together_flags_nothing() -> None:
    t = FaultTracker()
    events = [t.feed(_obs(i, 10.0 + i, 10.0 + i), S) for i in range(8)]
    assert all(e is None for e in events)


def test_stuck_needs_camera_confirmation() -> None:
    t = FaultTracker()
    # gage flat, camera below range on every frame: the camera cannot testify.
    events = [t.feed(_obs(i, 9.41, None), S) for i in range(8)]
    assert all(e is None for e in events)


def test_stuck_fires_when_camera_moves() -> None:
    t = FaultTracker()
    events = [t.feed(_obs(i, 10.0, 10.0 + 0.2 * i), S) for i in range(4)]
    assert events[-1] is not None and events[-1].kind == "stuck"


def test_cusum_fires_on_one_sigma_shift_not_noise() -> None:
    rng = np.random.default_rng(1)
    t = FaultTracker()
    noise = []
    for i, v in enumerate(rng.normal(0, 1, 200)):
        gage = 10.0 + 0.01 * i  # slow real rise, so only CUSUM could fire
        noise.append(t.feed(_obs(i, gage, gage + 0.1 * float(v), 0.1), S))
    assert all(e is None for e in noise)
    # a persistent +1-sigma bias must accumulate to the threshold quickly
    shift = [t.feed(_obs(200 + i, 12.0, 12.1, 0.1), S) for i in range(15)]
    assert any(e is not None and e.kind == "drift" for e in shift)


def test_spike_flags_the_reversed_frame() -> None:
    t = FaultTracker()
    events = [
        t.feed(_obs(i, g, 10.1), S) for i, g in enumerate([10.0, 10.0, 10.0, 13.0, 10.2, 10.1])
    ]
    # at f3 the +3 ft jump is only detectable as a step (disagreeing move);
    # the next frame proves it a one-frame reversal, which is the spike flag
    assert events[3] is not None and events[3].kind == "step"
    spike = events[4]
    assert spike is not None and spike.kind == "spike"
    assert spike.file == "f3.jpg"
    assert spike.time == datetime(2026, 8, 1, 0, 45, tzinfo=UTC)


def test_suppression_and_rearm() -> None:
    t = FaultTracker()
    cam = iter(10.0 + 0.2 * i for i in range(20))
    gage_seq = [10.0] * 6 + [10.6] + [11.0] * 6
    events = [t.feed(_obs(i, gage_seq[i], next(cam)), S) for i in range(13)]
    kinds = [i for i, e in enumerate(events) if e is not None and e.kind == "stuck"]
    # fires once on the first flat stretch, stays latched while it holds,
    # re-arms when the gage moves, fires again on the second flat stretch
    assert kinds == [3, 10]


def test_stuck_needs_four_identical_gage() -> None:
    t = FaultTracker()
    short = [t.feed(_obs(i, 10.0, 10.0 + 0.2 * i), S) for i in range(3)]
    assert all(e is None for e in short)
    t2 = FaultTracker()
    wiggle = [
        t2.feed(_obs(i, g, 10.0 + 0.2 * i), S) for i, g in enumerate([10.0, 10.1, 10.0, 10.0])
    ]
    assert all(e is None for e in wiggle)


@pytest.fixture(scope="module")
def ctx() -> object:
    site = sites.load_site("hancock")
    return site, make_perception(site, Settings())


def _fixture_ref(entry: dict, site) -> FrameRef:
    return FrameRef(
        cam_id=site.cam_id,
        filename=entry["filename"],
        time=datetime.fromisoformat(entry["time"]),
        url="",
    )


def test_loop_accepts_a_flood_frame(ctx: object) -> None:
    site, perception = ctx
    manifest = json.loads(Path("tests/fixtures/manifest_hancock.json").read_text())
    flood = next(e for e in manifest if e["gage_ft"] > 17)
    image = cv2.imread(str(Path("tests/fixtures") / flood["path"]))
    reading, trace = run_frame(perception, np.asarray(image), _fixture_ref(flood, site))
    assert reading.status == "accepted"
    assert abs((reading.stage_ft or 0) - flood["gage_ft"]) < 1.0
    assert trace.final_status == "accepted"
    assert len(trace.steps) <= 2


def test_loop_abstains_or_below_range_on_dry(ctx: object) -> None:
    site, perception = ctx
    manifest = json.loads(Path("tests/fixtures/manifest_hancock.json").read_text())
    dry = next(e for e in manifest if e["gage_ft"] < 9.4)
    image = cv2.imread(str(Path("tests/fixtures") / dry["path"]))
    reading, _ = run_frame(perception, np.asarray(image), _fixture_ref(dry, site))
    assert reading.status in {"below_range", "abstained"}
    assert reading.stage_ft is None or abs(reading.stage_ft - dry["gage_ft"]) < 1.0


def test_flag_engine_emits_drift_flag(tmp_path: Path) -> None:
    engine = FlagEngine("hancock")
    flag = None
    for i in range(6):
        t = datetime(2026, 8, 1, 12, tzinfo=UTC) + timedelta(minutes=15 * i)
        reading = Reading(
            site_id="hancock",
            time=t,
            stage_ft=11.0,
            sigma_ft=0.4,
            status="accepted",
        )
        trace = Trace(
            trace_id=f"t{i}",
            site_id="hancock",
            frame_time=t,
            frame_file=f"f{i}.jpg",
            started_at=t,
            finished_at=t,
            steps=[],
            final_status="accepted",
            final_stage_ft=11.0,
            final_sigma_ft=0.4,
            reason=None,
            policy="deterministic",
        )
        gage = 10.0 + 0.05 * i  # real drift; camera over-reads by about a foot
        flag = engine.feed(reading, gage, trace, S, tmp_path) or flag
    assert flag is not None
    assert flag.site_id == "hancock"
    assert flag.event_type == "drift"
    assert flag.residual_ft == pytest.approx(11.0 - 10.1)
    assert flag.persistence_frames == 3
    assert (tmp_path / "hancock").exists()
