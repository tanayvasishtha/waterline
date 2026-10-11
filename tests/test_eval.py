import subprocess
import sys

import pytest

from waterline.eval import fit_offset, summarize


def test_fit_offset_median() -> None:
    rows = [
        {"stage_ft": 10.0, "gage_ft": 9.5},
        {"stage_ft": 11.0, "gage_ft": 10.4},
        {"stage_ft": None, "gage_ft": 9.9},
        {"stage_ft": 10.2, "gage_ft": None},
    ]
    fit = fit_offset(rows)
    assert fit["offset_ft"] == pytest.approx(0.55)
    assert fit["n"] == 2


def test_summarize_counts_and_metrics() -> None:
    rows = [
        {"stage_ft": 10.0, "gage_ft": 9.8, "sigma_ft": 0.3, "reason": None},
        {"stage_ft": 10.5, "gage_ft": 10.4, "sigma_ft": 0.3, "reason": None},
        {"stage_ft": None, "gage_ft": 9.0, "sigma_ft": None, "reason": "below_plate"},
        {"stage_ft": None, "gage_ft": 9.1, "sigma_ft": None, "reason": "registration failed"},
    ]
    s = summarize(rows)
    assert s["frames"] == 4
    assert s["read"] == 2
    assert s["below_plate"] == 1
    assert s["coverage"] == 0.5
    assert abs(s["mae_ft"] - 0.15) < 1e-9


def test_cli_help_lists_commands() -> None:
    out = subprocess.run(
        [sys.executable, "-m", "waterline", "--help"], capture_output=True, text=True, check=False
    )
    assert "calibrate" in out.stdout and "eval" in out.stdout
