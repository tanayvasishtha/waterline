import subprocess
import sys

import pytest

from waterline.eval import FrameRow, fit_offset, summarize


def _row(stage: float | None, gage: float | None, **kw: object) -> FrameRow:
    row: FrameRow = {
        "file": "f.jpg",
        "time": "2026-08-01T12:00:00+00:00",
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


def test_cli_help_lists_commands() -> None:
    out = subprocess.run(
        [sys.executable, "-m", "waterline", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert "calibrate" in out.stdout and "eval" in out.stdout
