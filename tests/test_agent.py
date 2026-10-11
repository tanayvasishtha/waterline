"""Agent policy, loop and fault-tracker tests. Fixtures keep them offline."""

import json
from datetime import UTC, datetime
from pathlib import Path

import types

import cv2
import numpy as np
import pytest

from waterline import sites
from waterline.agent.fault import FaultTracker, is_stuck
from waterline.agent.loop import run_frame
from waterline.agent.policy import LoopState, decide
from waterline.eval import make_perception
from waterline.models import (
    AgentAction,
    FrameRef,
    QualityReport,
    Reading,
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
        detection=types.SimpleNamespace(sigma_ft=rkw.get("sigma_ft", 0.1)) if kw.get("detected", True) else None,  # type: ignore[arg-type]
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


def test_fault_tracker_catches_drift() -> None:
    t = FaultTracker()
    flagged = None
    for i in range(30):
        flagged = t.feed(0.4 + 0.05 * i, 10.0 + 0.01 * i, S)
    assert flagged == "drift"


def test_fault_tracker_catches_stuck_sensor() -> None:
    t = FaultTracker()
    flagged = [t.feed(None, 9.41, S) for _ in range(6)]
    assert "stuck" in flagged


def test_stuck_needs_four_identical() -> None:
    assert is_stuck([1.0, 1.0, 1.0, 1.0])
    assert not is_stuck([1.0, 1.0, 1.0])
    assert not is_stuck([1.0, 1.1, 1.0, 1.0])


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
    reading, trace = run_frame(
        perception, np.asarray(image), _fixture_ref(flood, site)
    )
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
