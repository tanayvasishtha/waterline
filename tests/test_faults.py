"""Injected-fault harness tests: deterministic output plus a hand-checked trial."""

import json
from datetime import UTC, datetime, timedelta

from waterline.faults import Row, collect, inject, run_faults
from waterline.settings import Settings


def _rows(n: int, gage: float = 10.0, camera: float = 10.0, sigma: float = 0.1) -> list[Row]:
    return [
        Row(
            time=datetime(2026, 9, 1, tzinfo=UTC) + timedelta(minutes=30 * i),
            file=f"f{i}.jpg",
            gage_ft=gage,
            camera_ft=camera,
            sigma_ft=sigma,
        )
        for i in range(n)
    ]


def test_run_faults_is_deterministic() -> None:
    rows = _rows(16)
    a = run_faults("t", rows)
    b = run_faults("t", rows)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    assert a["seed"] == 20261026
    assert a["trials_per_type"] == 20


def test_step_injection_fires_at_onset() -> None:
    # gage flat at 10.0, camera flat at 10.3 (sigma 0.1). A +0.8 ft gage step at
    # i0=4 disagrees with the camera by 0.8 ft >> 3 combined sigmas (0.42 ft).
    rows = _rows(12)
    flags = collect("t", inject(rows, "step", 4), Settings())
    assert flags[0][0] == 4
    assert flags[0][1].event_type == "step"


def test_clean_flat_series_flags_nothing() -> None:
    flags = collect("t", _rows(12), Settings())
    assert flags == []


def test_clean_flags_do_not_count_as_detections() -> None:
    # flat series whose last row jumps +0.9 ft: the clean run flags a step at
    # index 15. The injected drift (0.005 ft per 15 min) is far too small to
    # trigger anything else, so every detection would be that masked flag.
    rows = _rows(16)
    rows[15].gage_ft = 10.9
    report = run_faults("t", rows)
    drift = report["faults"]["drift"]
    assert drift["recall"] == 0.0
    assert drift["recall_any"] == 0.0
