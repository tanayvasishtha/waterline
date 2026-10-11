"""Dashboard API tests: eval-only readings, summary metrics, filtering, 404s."""

import csv
import json
from pathlib import Path

import pytest

from waterline import web

COLUMNS = [
    "file",
    "time",
    "gage_ft",
    "stage_ft",
    "sigma_ft",
    "method_a_ft",
    "method_b_ft",
    "reason",
    "blur_var",
    "glare_frac",
    "mean_v",
    "inliers",
    "inlier_ratio",
    "shift_px",
    "plate_visibility",
    "is_night",
    "gate_pass",
]


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def row(time: str, **kw: str) -> dict[str, str]:
    base = {c: "" for c in COLUMNS}
    base.update(
        {
            "file": f"f_{time}.jpg",
            "time": time,
            "gage_ft": "10.0",
            "sigma_ft": "0.2",
            "is_night": "False",
            "gate_pass": "True",
        }
    )
    base.update(kw)
    return base


@pytest.fixture
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    results = tmp_path / "results"
    data = tmp_path / "data"
    results.mkdir()
    data.mkdir()
    monkeypatch.setattr(web, "RESULTS", results)
    monkeypatch.setattr(web, "DATA", data)
    return tmp_path


def test_readings_skip_calib_csvs(data_dir: Path) -> None:
    write_csv(
        data_dir / "results" / "calib_hancock_2026-07-01_2026-07-31.csv",
        [row("2026-07-15T00:00:00+00:00", stage_ft="9.9")],
    )
    write_csv(
        data_dir / "results" / "eval_hancock_2026-08-01_2026-08-05.csv",
        [row("2026-08-02T00:00:00+00:00", stage_ft="10.1")],
    )
    rows = web._readings("hancock")
    assert len(rows) == 1
    assert rows[0]["time"] == "2026-08-02T00:00:00+00:00"


def test_readings_skip_agent_eval_csv(data_dir: Path) -> None:
    write_csv(
        data_dir / "results" / "eval_agent_hancock_2026-08-01_2026-08-05.csv",
        [row("2026-08-03T00:00:00+00:00", stage_ft="9.9")],
    )
    write_csv(
        data_dir / "results" / "eval_hancock_2026-08-01_2026-08-05.csv",
        [row("2026-08-02T00:00:00+00:00", stage_ft="10.1")],
    )
    rows = web._readings("hancock")
    assert len(rows) == 1
    assert rows[0]["time"] == "2026-08-02T00:00:00+00:00"


def test_readings_status_map(data_dir: Path) -> None:
    write_csv(
        data_dir / "results" / "eval_hancock_2026-08-01_2026-08-05.csv",
        [
            row("2026-08-01T00:00:00+00:00", reason="below_plate"),
            row("2026-08-02T00:00:00+00:00", gate_pass="False", stage_ft="10.4"),
            row("2026-08-03T00:00:00+00:00", stage_ft="10.2"),
            row("2026-08-04T00:00:00+00:00"),
        ],
    )
    statuses = [r["status"] for r in web._readings("hancock")]
    assert statuses == ["below_range", "abstained", "read", "abstained"]


def test_summary_metrics(data_dir: Path) -> None:
    write_csv(
        data_dir / "results" / "eval_hancock_2026-08-01_2026-08-05.csv",
        [
            row("2026-08-01T00:00:00+00:00", stage_ft="10.1"),
            row("2026-08-01T01:00:00+00:00", stage_ft="10.2"),
            row("2026-08-01T02:00:00+00:00", stage_ft="10.3"),
            row("2026-08-01T03:00:00+00:00", stage_ft="10.4"),
            row("2026-08-01T04:00:00+00:00", reason="below_plate"),
            row("2026-08-01T05:00:00+00:00", gate_pass="False", stage_ft="11.0"),
        ],
    )
    s = web._summary("hancock")
    assert s["window"] == ["2026-08-01", "2026-08-05"]
    assert s["n_frames"] == 6
    assert s["n_reads"] == 4
    assert s["n_below_range"] == 1
    assert s["n_abstained"] == 1
    assert s["mae_ft"] == pytest.approx(0.25)
    assert s["median_ft"] == pytest.approx(0.25)
    assert s["p90_ft"] == pytest.approx(0.37)


def test_summary_no_eval_csv(data_dir: Path) -> None:
    s = web._summary("oh_wetland")
    assert s["n_frames"] == 0
    assert s["mae_ft"] is None
    assert s["window"] is None


def test_traces_filter_then_slice(data_dir: Path) -> None:
    tdir = data_dir / "data" / "traces"
    tdir.mkdir(parents=True)
    for i in range(60):
        (tdir / f"hancock-202608{i:02d}T0000Z.json").write_text(
            json.dumps({"site_id": "hancock", "trace_id": f"h-{i}"})
        )
    for i in range(60):
        (tdir / f"zzzsite-202609{i:02d}T0000Z.json").write_text(
            json.dumps({"site_id": "zzzsite", "trace_id": f"z-{i}"})
        )
    traces = web._traces("hancock")
    assert len(traces) == 50
    assert all(t["site_id"] == "hancock" for t in traces)
    assert traces[-1]["trace_id"] == "h-59"


def test_unknown_site_and_endpoint_404(data_dir: Path) -> None:
    for path in ("/api/site", "/api/summary", "/api/readings"):
        status, _, body = web.api(path, {"site": ["not_a_site"]})
        assert status == 404, path
        assert json.loads(body)["error"] == "unknown site"
    status, _, _ = web.api("/api/nope", {})
    assert status == 404


def test_flags_site_filter(data_dir: Path) -> None:
    for site, stamp in (("hancock", "20260801T0000Z"), ("in_overland", "20260802T0000Z")):
        d = data_dir / "data" / "events" / site / stamp
        d.mkdir(parents=True)
        (d / "flag.json").write_text(
            json.dumps({"site_id": site, "opened_at": "2026-08-01T00:00:00Z"})
        )
    assert len(web._flags(None)) == 2
    mine = web._flags("hancock")
    assert len(mine) == 1
    assert mine[0]["site_id"] == "hancock"
