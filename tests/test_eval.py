import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from waterline import eval as ev
from waterline.eval import FrameRow, fit_offset, summarize
from waterline.models import FrameRef


def _row(stage: float | None, gage: float | None, **kw: object) -> FrameRow:
    row: FrameRow = {
        "file": str(kw.get("file", "f.jpg")),
        "time": str(kw.get("time", "2026-08-01T12:00:00+00:00")),
        "gage_ft": gage,
        "stage_ft": stage,
        "sigma_ft": 0.3 if stage is not None else None,
        "method_a_ft": stage,
        "method_b_ft": None,
        "reason": kw.get("reason"),
        "blur_var": 400.0,
        "glare_frac": 0.0,
        "mean_v": 150.0,
        "inliers": 200,
        "inlier_ratio": 0.7,
        "shift_px": 1.0,
        "plate_visibility": 40.0,
        "is_night": bool(kw.get("is_night", False)),
        "gate_pass": bool(kw.get("gate_pass", True)),
        "agent_stage_ft": kw.get("agent_stage_ft"),
        "agent_sigma_ft": kw.get("agent_sigma_ft"),
        "agent_gage_ft": kw.get("agent_gage_ft"),
        "agent_status": kw.get("agent_status"),
        "agent_steps": kw.get("agent_steps"),
        "agent_frame_file": kw.get("agent_frame_file"),
    }
    return row


def test_fit_offset_median() -> None:
    rows = [_row(10.0, 9.5), _row(11.0, 10.4), _row(None, 9.9), _row(10.2, None)]
    fit = fit_offset(rows)
    assert fit["offset_ft"] == pytest.approx(0.55)
    assert fit["n"] == 2


def test_summarize_counts_and_metrics() -> None:
    rows = [
        _row(10.0, 9.8),
        _row(10.5, 10.4),
        _row(None, 9.0, reason="below_plate"),
        _row(None, 9.1, is_night=True, gate_pass=False),
    ]
    s = summarize(rows)["all"]
    assert s["frames"] == 4
    assert s["gate_pass"] == 3
    assert s["read"] == 2
    assert s["below_plate"] == 1
    assert s["coverage"] == 0.5
    assert abs(s["mae_ft"] - 0.15) < 1e-9


def test_agent_summary_dedups_shared_source_frames() -> None:
    rows = [
        _row(
            10.0,
            9.8,
            file="a.jpg",
            agent_status="accepted",
            agent_stage_ft=10.1,
            agent_gage_ft=9.9,
            agent_frame_file="a.jpg",
        ),
        _row(
            None,
            9.7,
            file="b.jpg",
            is_night=True,
            gate_pass=False,
            agent_status="accepted",
            agent_stage_ft=10.0,
            agent_gage_ft=9.6,
            agent_frame_file="c.jpg",
        ),
        _row(
            10.0,
            9.6,
            file="c.jpg",
            agent_status="accepted",
            agent_stage_ft=10.0,
            agent_gage_ft=9.6,
            agent_frame_file="c.jpg",
        ),
    ]
    s = summarize(rows)["all"]["agent"]
    # b.jpg's accepted read comes from c.jpg, so b and c share one source
    assert s["accepted"] == 2
    assert s["deduped"] == 1
    assert s["substituted"] == 1
    assert s["read"] == 2
    assert abs(s["mae_ft"] - (0.2 + 0.4) / 2) < 1e-9


def test_cli_help_lists_commands() -> None:
    out = subprocess.run(
        [sys.executable, "-m", "waterline", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert "calibrate" in out.stdout and "eval" in out.stdout


def _freeze(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tw = tmp_path / "test_windows.json"
    tw.write_text(
        json.dumps(
            {"hancock": [{"start": "2026-05-01T00:00Z", "end": "2026-05-10T00:00Z", "every": 2}]}
        )
    )
    monkeypatch.setattr(ev, "TEST_WINDOWS_PATH", tw)


def test_eval_refuses_frozen_test_window(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _freeze(tmp_path, monkeypatch)
    with pytest.raises(ev.TestWindowError, match="frozen test window"):
        ev._check_window(
            "hancock",
            datetime(2026, 5, 3, tzinfo=UTC),
            datetime(2026, 5, 4, tzinfo=UTC),
            final=False,
        )
    # a window clear of the frozen range is fine
    ev._check_window(
        "hancock",
        datetime(2026, 6, 1, tzinfo=UTC),
        datetime(2026, 6, 2, tzinfo=UTC),
        final=False,
    )
    # a different site is unaffected
    ev._check_window(
        "in_overland",
        datetime(2026, 5, 3, tzinfo=UTC),
        datetime(2026, 5, 4, tzinfo=UTC),
        final=False,
    )


def test_final_refuses_on_protocol_hash_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _freeze(tmp_path, monkeypatch)
    proto = tmp_path / "PROTOCOL.md"
    proto.write_text("frozen protocol text")
    bad = tmp_path / "hash.txt"
    bad.write_text("0" * 64)
    monkeypatch.setattr(ev, "PROTOCOL_PATH", proto)
    monkeypatch.setattr(ev, "PROTOCOL_HASH_PATH", bad)
    with pytest.raises(ev.TestWindowError, match="hash"):
        ev._check_window(
            "hancock",
            datetime(2026, 5, 3, tzinfo=UTC),
            datetime(2026, 5, 4, tzinfo=UTC),
            final=True,
        )
    bad.write_text(hashlib.sha256(proto.read_bytes()).hexdigest())
    ev._check_window(
        "hancock",
        datetime(2026, 5, 3, tzinfo=UTC),
        datetime(2026, 5, 4, tzinfo=UTC),
        final=True,
    )


def test_select_excludes_reference_frames() -> None:
    refs = [
        FrameRef(cam_id="c", filename=f"f{i}.jpg", time=datetime(2026, 5, 1, i, tzinfo=UTC), url="")
        for i in range(5)
    ]
    sel = ev._select(refs, 1, frozenset({"f2.jpg"}))
    assert [f.filename for f in sel] == ["f0.jpg", "f1.jpg", "f3.jpg", "f4.jpg"]
    sel = ev._select(refs, 2, frozenset({"f0.jpg"}))
    assert [f.filename for f in sel] == ["f2.jpg", "f4.jpg"]
