"""Agent policy, loop and fault-tracker tests. Fixtures keep them offline."""

import inspect
import json
import types
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

import cv2
import numpy as np
import pytest

from waterline import sites
from waterline.agent import loop as loop_mod
from waterline.agent.fault import FaultTracker, Obs
from waterline.agent.flag import FlagEngine
from waterline.agent.history import CameraHistory
from waterline.agent.loop import run_frame
from waterline.agent.policy import Decision, LoopState, decide
from waterline.eval import FrameRow, _agent_summary, make_perception
from waterline.models import (
    AgentAction,
    FrameRef,
    QualityReport,
    Reading,
    Trace,
)
from waterline.pipeline import FrameResult, Perception
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


def test_policy_method_disagreement_does_not_gate() -> None:
    # method B disagreeing is logged, not gated on: a tight-sigma read accepts
    d = decide(_result(r_methods_disagree=True, r_sigma_ft=0.1), LoopState(), S)
    assert d.action == AgentAction.ACCEPT


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
    cams = [10.0, 10.3, 10.7, 11.0]  # trend 1.0 ft > 3 combined sigmas
    events = [t.feed(_obs(i, 10.0, cams[i]), S) for i in range(4)]
    assert events[-1] is not None and events[-1].kind == "stuck"


def test_stuck_ignores_camera_jitter() -> None:
    rng = np.random.default_rng(11)
    fires = 0
    for _ in range(1000):
        t = FaultTracker()
        ev = None
        for i, v in enumerate(rng.normal(0, 0.2, 4)):
            ev = t.feed(_obs(i, 10.0, 10.0 + float(v), 0.2), S) or ev
        fires += int(ev is not None)
    assert fires / 1000 < 0.05


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
    cam = iter(10.0 + 0.4 * i for i in range(20))
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


def test_run_frame_keeps_neighbour_time_and_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # a frame with no stage substitutes the neighbour outright: the reading
    # keeps the neighbour's own time and file (the v2 fix)
    t0 = datetime(2026, 8, 1, tzinfo=UTC)
    ref = _fake_ref(t0)
    nb_ref = FrameRef(cam_id="c", filename="neighbour.jpg", time=t0 + timedelta(hours=1), url="")
    seq = iter([_result(r_stage_ft=None), _result(r_stage_ft=9.9, r_sigma_ft=0.2)])
    monkeypatch.setattr(loop_mod, "read_frame", lambda *a, **k: next(seq))

    class FakeNav:
        def neighbor(self, when: datetime, direction: int, skip: int = 1) -> FrameRef:
            return nb_ref

        def image(self, r: FrameRef) -> np.ndarray:
            return np.zeros((8, 8, 3), np.uint8)

    calls = [
        Decision(AgentAction.FETCH_NEXT, "test"),
        Decision(AgentAction.ACCEPT, "test"),
    ]
    reading, trace = run_frame(
        cast("Perception", _fake_ctx()),
        np.zeros((8, 8, 3), np.uint8),
        ref,
        FakeNav(),  # type: ignore[arg-type]
        chooser=_scripted(calls),  # type: ignore[arg-type]
    )
    assert reading.time == nb_ref.time
    assert reading.frame_file == "neighbour.jpg"
    assert trace.frame_time == nb_ref.time
    assert trace.frame_file == "neighbour.jpg"


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


def _agent_row(**kw: object) -> FrameRow:
    base: dict[str, object] = {
        "file": "f.jpg",
        "time": "2026-08-01T00:00:00+00:00",
        "gage_ft": 10.0,
        "stage_ft": 10.0,
        "sigma_ft": 0.2,
        "method_a_ft": 10.0,
        "method_b_ft": 10.0,
        "reason": None,
        "blur_var": 500.0,
        "glare_frac": 0.0,
        "mean_v": 150.0,
        "inliers": 300,
        "inlier_ratio": 0.7,
        "shift_px": 1.0,
        "plate_visibility": 45.0,
        "is_night": False,
        "gate_pass": True,
        "agent_stage_ft": 10.5,
        "agent_sigma_ft": 0.2,
        "agent_gage_ft": 10.0,
        "agent_status": "accepted",
        "agent_steps": 1,
    }
    base.update(kw)
    return cast(FrameRow, base)


def test_agent_summary_scores_only_accepted_rows() -> None:
    rows = [
        _agent_row(agent_stage_ft=10.5, agent_gage_ft=10.0, gage_ft=8.0),
        _agent_row(agent_status="abstained", agent_stage_ft=99.0),
    ]
    agent = _agent_summary(rows)
    assert agent is not None
    assert agent["read"] == 1
    assert agent["mae_ft"] == pytest.approx(0.5)


def _fake_ctx(rate_max: float | None = None, alt_refs: int = 1) -> object:
    site = types.SimpleNamespace(site_id="t", cam_id="c", rate_max_ft_per_h=rate_max)
    return types.SimpleNamespace(
        site=site,
        settings=Settings(),
        bank=types.SimpleNamespace(descriptors={f"r{i}": i for i in range(alt_refs)}),
    )


def _fake_ref(t: datetime) -> FrameRef:
    return FrameRef(cam_id="c", filename="f.jpg", time=t, url="")


def _scripted(calls: list[Decision], seen: list[bool] | None = None) -> object:
    seq = iter(calls)

    def chooser(r: object, st: LoopState, s: Settings) -> Decision:
        if seen is not None:
            seen.append(st.self_inconsistent or st.implausible)
        return next(seq)

    return chooser


def test_repair_that_moves_the_reading_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 0.86 ft move at sigma 0.15->0.22: 3 combined sigmas is 0.80, so the
    # repair changed the answer and the frame counts as uncertain
    seq = iter(
        [
            _result(r_stage_ft=10.0, r_sigma_ft=0.15),
            _result(r_stage_ft=10.86, r_sigma_ft=0.22),
        ]
    )
    monkeypatch.setattr(loop_mod, "read_frame", lambda *a, **k: next(seq))
    seen: list[bool] = []
    calls = [
        Decision(AgentAction.ENHANCE_GAMMA, "try a contrast lift"),
        Decision(AgentAction.HUMAN_REVIEW, "still unsure"),
    ]
    reading, trace = run_frame(
        cast("Perception", _fake_ctx()),
        np.zeros((8, 8, 3), np.uint8),
        _fake_ref(datetime(2026, 8, 1, tzinfo=UTC)),
        chooser=_scripted(calls, seen),  # type: ignore[arg-type]
    )
    assert trace.steps[0].decision == "repair rejected: changed the reading by 0.86 ft"
    assert trace.steps[0].result_summary.endswith("dR=0.86")
    assert reading.stage_ft == 10.0  # r_old kept
    assert seen == [False, True]  # repair rejection forced the uncertain path
    assert reading.status == "pending"


def test_repair_that_only_lowers_sigma_is_adopted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seq = iter(
        [
            _result(r_stage_ft=10.0, r_sigma_ft=0.30),
            _result(r_stage_ft=10.05, r_sigma_ft=0.10),
        ]
    )
    monkeypatch.setattr(loop_mod, "read_frame", lambda *a, **k: next(seq))
    calls = [
        Decision(AgentAction.ENHANCE_GAMMA, "try a contrast lift"),
        Decision(AgentAction.ACCEPT, "within sigma"),
    ]
    reading, trace = run_frame(
        cast("Perception", _fake_ctx()),
        np.zeros((8, 8, 3), np.uint8),
        _fake_ref(datetime(2026, 8, 1, tzinfo=UTC)),
        chooser=_scripted(calls),  # type: ignore[arg-type]
    )
    assert trace.steps[0].decision == "try a contrast lift"  # adopted, no rejection
    assert reading.stage_ft == 10.05
    assert reading.sigma_ft == 0.10


def test_implausible_jump_forces_uncertain_then_review(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    t0 = datetime(2026, 8, 1, tzinfo=UTC)
    history = CameraHistory(stage_ft=10.0, sigma_ft=0.2, time=t0)
    # prev 10.0 ft; frame 1 h later reads 12.5; rate cap 0.5 ft/h gives
    # band 0.5 + 3*0.283 = 1.35, and the neighbours read the same 12.5,
    # so every fetched frame stays implausible and the loop escalates
    seq = iter([_result(r_stage_ft=12.5, r_sigma_ft=0.2), _result(r_stage_ft=10.0, r_sigma_ft=0.2)])
    monkeypatch.setattr(loop_mod, "read_frame", lambda *a, **k: next(seq))
    ref = _fake_ref(t0 + timedelta(hours=1))

    class FakeNav:
        def neighbor(self, when: datetime, direction: int, skip: int = 1) -> FrameRef:
            return FrameRef(
                cam_id="c",
                filename=f"n{direction}.jpg",
                time=t0 + timedelta(hours=2.5),
                url="",
            )

        def image(self, r: FrameRef) -> np.ndarray:
            return np.zeros((8, 8, 3), np.uint8)

    seen: list[bool] = []

    def spy(r: object, st: LoopState, s: Settings) -> Decision:
        seen.append(st.self_inconsistent or st.implausible)
        return decide(cast("FrameResult", r), st, s)

    reading, trace = run_frame(
        cast("Perception", _fake_ctx(rate_max=0.5)),
        np.zeros((8, 8, 3), np.uint8),
        ref,
        FakeNav(),  # type: ignore[arg-type]
        chooser=spy,  # type: ignore[arg-type]
        history=history,
    )
    # the implausible flag sent it to a neighbour; the neighbour reads the real
    # 10.0 ft and disagrees with 12.5, so the frame escalates
    assert seen == [True]
    assert "bound=1.60 d=2.50" in trace.steps[0].result_summary
    assert trace.steps[0].decision == "neighbour disagrees by 2.50 ft"
    assert [s.tool for s in trace.steps] == ["fetch_prev"]
    assert reading.status == "pending"
    assert reading.stage_ft == 12.5


def test_plausible_flood_rise_is_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    t0 = datetime(2026, 8, 1, tzinfo=UTC)
    history = CameraHistory(stage_ft=10.0, sigma_ft=0.2, time=t0)
    monkeypatch.setattr(
        loop_mod, "read_frame", lambda *a, **k: _result(r_stage_ft=10.5, r_sigma_ft=0.2)
    )
    seen: list[bool] = []

    def spy(r: object, st: LoopState, s: Settings) -> Decision:
        seen.append(st.self_inconsistent or st.implausible)
        return decide(cast("FrameResult", r), st, s)

    reading, _ = run_frame(
        cast("Perception", _fake_ctx(rate_max=0.5)),
        np.zeros((8, 8, 3), np.uint8),
        _fake_ref(t0 + timedelta(hours=1)),
        chooser=spy,  # type: ignore[arg-type]
        history=history,
    )
    assert seen == [False]  # 0.5 ft in an hour under a 0.5 ft/h cap: plausible
    assert reading.status == "accepted"


def test_camera_history_has_no_gage() -> None:
    assert set(inspect.signature(CameraHistory.update).parameters) == {"self", "reading"}
    assert set(CameraHistory.__dataclass_fields__) == {"stage_ft", "sigma_ft", "time"}


def _nb_nav(t: datetime) -> object:
    class FakeNav:
        def neighbor(self, when: datetime, direction: int, skip: int = 1) -> FrameRef:
            return FrameRef(cam_id="c", filename=f"nb{direction}.jpg", time=t, url="")

        def image(self, r: FrameRef) -> np.ndarray:
            return np.zeros((8, 8, 3), np.uint8)

    return FakeNav()


def test_lost_repair_keeps_original_and_accepts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # a repair that loses the reading does not flag the frame: r_old is kept
    # and accepted on its own merits, without a single neighbour fetch
    seq = iter([_result(r_stage_ft=0.634, r_sigma_ft=0.054), _result(r_stage_ft=None)])
    monkeypatch.setattr(loop_mod, "read_frame", lambda *a, **k: next(seq))
    calls = [
        Decision(AgentAction.MASK_GLARE, "try inpainting"),
        Decision(AgentAction.ACCEPT, "within sigma"),
    ]
    reading, trace = run_frame(
        cast("Perception", _fake_ctx()),
        np.zeros((8, 8, 3), np.uint8),
        _fake_ref(datetime(2026, 8, 1, tzinfo=UTC)),
        chooser=_scripted(calls),  # type: ignore[arg-type]
    )
    assert [s.tool for s in trace.steps] == ["mask_glare"]
    assert trace.steps[0].decision == "repair rejected: lost the reading"
    assert reading.status == "accepted"
    assert reading.stage_ft == 0.634


def test_changed_answer_confirmed_by_neighbour(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # a repair that changed the answer flags the frame; a neighbour that
    # agrees inside the travel-time bound confirms r_old at the ORIGINAL time
    t0 = datetime(2026, 8, 1, 12, tzinfo=UTC)
    nb = t0 + timedelta(hours=1)
    seq = iter(
        [
            _result(r_stage_ft=10.0, r_sigma_ft=0.15),
            _result(r_stage_ft=10.86, r_sigma_ft=0.22),
            _result(r_stage_ft=9.9, r_sigma_ft=0.2),
        ]
    )
    monkeypatch.setattr(loop_mod, "read_frame", lambda *a, **k: next(seq))
    calls = [
        Decision(AgentAction.ENHANCE_GAMMA, "try a contrast lift"),
        Decision(AgentAction.FETCH_PREV, "compare previous"),
    ]
    reading, trace = run_frame(
        cast("Perception", _fake_ctx(rate_max=0.5)),
        np.zeros((8, 8, 3), np.uint8),
        _fake_ref(t0),
        _nb_nav(nb),  # type: ignore[arg-type]
        chooser=_scripted(calls),  # type: ignore[arg-type]
    )
    assert trace.steps[0].decision == "repair rejected: changed the reading by 0.86 ft"
    assert trace.steps[1].decision == "confirmed by neighbour at 13:00"
    assert "dt_h=1.00" in trace.steps[1].result_summary
    assert reading.status == "accepted"
    assert reading.stage_ft == 10.0  # r_old, not the repair or the neighbour
    assert reading.time == t0
    assert reading.frame_file == "f.jpg"


def test_neighbour_disagreement_goes_to_review(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    t0 = datetime(2026, 8, 1, 12, tzinfo=UTC)
    nb = t0 + timedelta(hours=1)
    seq = iter(
        [
            _result(r_stage_ft=10.0, r_sigma_ft=0.15),
            _result(r_stage_ft=10.86, r_sigma_ft=0.22),
            _result(r_stage_ft=13.0, r_sigma_ft=0.2),
        ]
    )
    monkeypatch.setattr(loop_mod, "read_frame", lambda *a, **k: next(seq))
    calls = [
        Decision(AgentAction.ENHANCE_GAMMA, "try a contrast lift"),
        Decision(AgentAction.FETCH_PREV, "compare previous"),
    ]
    reading, trace = run_frame(
        cast("Perception", _fake_ctx(rate_max=0.5)),
        np.zeros((8, 8, 3), np.uint8),
        _fake_ref(t0),
        _nb_nav(nb),  # type: ignore[arg-type]
        chooser=_scripted(calls),  # type: ignore[arg-type]
    )
    assert trace.steps[1].decision == "neighbour disagrees by 3.00 ft"
    assert reading.status == "pending"
    assert reading.stage_ft == 10.0  # the pending reading keeps r_old


def test_no_neighbour_could_confirm(monkeypatch: pytest.MonkeyPatch) -> None:
    t0 = datetime(2026, 8, 1, 12, tzinfo=UTC)
    seq = iter(
        [
            _result(r_stage_ft=10.0, r_sigma_ft=0.15),
            _result(r_stage_ft=10.86, r_sigma_ft=0.22),
            _result(r_stage_ft=None),
            _result(r_stage_ft=None),
        ]
    )
    monkeypatch.setattr(loop_mod, "read_frame", lambda *a, **k: next(seq))
    calls = [
        Decision(AgentAction.ENHANCE_GAMMA, "try a contrast lift"),
        Decision(AgentAction.FETCH_PREV, "compare previous"),
        Decision(AgentAction.FETCH_NEXT, "compare next"),
    ]
    reading, trace = run_frame(
        cast("Perception", _fake_ctx(rate_max=0.5)),
        np.zeros((8, 8, 3), np.uint8),
        _fake_ref(t0),
        _nb_nav(t0 + timedelta(hours=1)),  # type: ignore[arg-type]
        chooser=_scripted(calls),  # type: ignore[arg-type]
    )
    assert [s.tool for s in trace.steps] == [
        "enhance_gamma",
        "fetch_prev",
        "fetch_next",
    ]
    assert trace.steps[-1].decision == "no neighbour could confirm"
    assert reading.status == "pending"
