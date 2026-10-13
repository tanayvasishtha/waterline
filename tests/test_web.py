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


AGENT_COLUMNS = [
    *COLUMNS,
    "agent_stage_ft",
    "agent_sigma_ft",
    "agent_gage_ft",
    "agent_status",
    "agent_steps",
    "agent_frame_file",
]


def write_agent_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=AGENT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def agent_row(time: str, **kw: str) -> dict[str, str]:
    base = {c: "" for c in AGENT_COLUMNS}
    base.update(row(time))
    base.update(kw)
    return base


def _final_summary_doc(blocks: dict[str, dict]) -> dict:
    stats = {
        "frames": 2,
        "gate_pass": 2,
        "read": 1,
        "below_plate": 0,
        "coverage": 0.5,
        "mae_ft": 0.1,
        "med_ae_ft": 0.1,
        "p90_ae_ft": 0.1,
        "within_2sigma": 1.0,
        "agent": {
            "accepted": 1,
            "deduped": 0,
            "substituted": 0,
            "below_range": 0,
            "abstained": 0,
            "pending": 1,
            "read": 1,
            "mae_ft": 0.08,
            "avg_steps": 0.5,
        },
    }
    return {
        "site": "hancock",
        "all": {"all": {**stats, "mae_ft": 0.2}},
        "blocks": {k: {"all": dict(v)} for k, v in blocks.items()},
    }


def test_final_readings_join_and_substitute_time(data_dir: Path) -> None:
    label = "2026-09-01_2026-09-16"
    write_csv(
        data_dir / "results" / f"final_eval_hancock_{label}.csv",
        [
            row("2026-09-01T02:00:03+00:00", file="Cam___2026-09-01T02-00-03Z.jpg"),
            row(
                "2026-09-01T04:00:09+00:00", file="Cam___2026-09-01T04-00-09Z.jpg", stage_ft="10.1"
            ),
        ],
    )
    write_agent_csv(
        data_dir / "results" / f"final_eval_agent_hancock_{label}.csv",
        [
            agent_row(
                "2026-09-01T02:00:03+00:00",
                file="Cam___2026-09-01T02-00-03Z.jpg",
                agent_status="accepted",
                agent_stage_ft="10.2",
                agent_gage_ft="10.0",
                agent_frame_file="Cam___2026-09-01T04-00-09Z.jpg",
            ),
            agent_row(
                "2026-09-01T04:00:09+00:00",
                file="Cam___2026-09-01T04-00-09Z.jpg",
                agent_status="pending",
            ),
        ],
    )
    (data_dir / "results" / "final_summary_hancock.json").write_text(
        json.dumps(_final_summary_doc({label: {}}))
    )
    rows = web._final_readings("hancock", "all")
    assert len(rows) == 2
    first = rows[0]
    assert first["agent_status"] == "accepted"
    assert first["agent_stage_ft"] == 10.2
    # substituted read plots at the source frame's time, not the frame's own
    assert first["agent_time"] == "2026-09-01T04:00:09Z"
    assert rows[1]["agent_status"] == "pending"
    assert rows[1]["agent_time"] == "2026-09-01T04:00:09+00:00"


def test_final_readings_agent_csv_not_in_baseline(data_dir: Path) -> None:
    write_csv(
        data_dir / "results" / "final_eval_hancock_2026-09-01_2026-09-16.csv",
        [row("2026-09-01T04:00:09+00:00", stage_ft="10.1")],
    )
    write_agent_csv(
        data_dir / "results" / "final_eval_agent_hancock_2026-09-01_2026-09-16.csv",
        [agent_row("2026-09-01T02:00:03+00:00", agent_status="accepted", agent_stage_ft="10.2")],
    )
    rows = web._final_readings("hancock", "all")
    assert len(rows) == 1
    assert rows[0]["time"] == "2026-09-01T04:00:09+00:00"


def test_final_summary_block_selection(data_dir: Path) -> None:
    blocks = {
        "2026-04-14_2026-05-28": {"mae_ft": 0.35},
        "2026-09-01_2026-09-16": {"mae_ft": 0.25},
    }
    (data_dir / "results" / "final_summary_hancock.json").write_text(
        json.dumps(_final_summary_doc(blocks))
    )
    assert web._final_windows("hancock") == list(blocks)
    s = web._final_summary("hancock", "2026-09-01_2026-09-16")
    assert s is not None and s["baseline"]["mae_ft"] == 0.25
    s = web._final_summary("hancock", "all")
    assert s is not None and s["baseline"]["mae_ft"] == 0.2
    assert web._final_summary("hancock", "1999-01-01_1999-01-02") is None
    # a site with no final files still answers 'all' with an empty block
    s = web._final_summary("oh_wetland", "all")
    assert s is not None and s["frames"] == 0 and s["baseline"]["mae_ft"] is None


def test_final_faults_endpoint(data_dir: Path) -> None:
    doc = {
        "injected": True,
        "site": "hancock",
        "skipped": [],
        "clean": {"flags": 8, "false_flags_per_site_week": 0.95},
        "faults": {
            "drift": {
                "recall": 0.95,
                "median_delay_h": 27.0,
                "type_match": 1.0,
                "n_detected": 38,
                "trials": 40,
            }
        },
    }
    (data_dir / "results" / "final_faults_hancock.json").write_text(json.dumps(doc))
    status, _, body = web.api("/api/final_faults", {"site": ["hancock"]})
    assert status == 200
    got = json.loads(body)
    assert got["injected"] is True
    assert got["faults"]["drift"]["recall"] == 0.95
    status, _, body = web.api("/api/final_faults", {"site": ["oh_wetland"]})
    assert status == 200
    assert json.loads(body)["available"] is False


def test_final_unknown_window_404(data_dir: Path) -> None:
    write_csv(
        data_dir / "results" / "final_eval_hancock_2026-09-01_2026-09-16.csv",
        [row("2026-09-01T04:00:09+00:00", stage_ft="10.1")],
    )
    (data_dir / "results" / "final_summary_hancock.json").write_text(
        json.dumps(_final_summary_doc({"2026-09-01_2026-09-16": {}}))
    )
    for ep in ("/api/final_summary", "/api/final_readings"):
        status, _, body = web.api(ep, {"site": ["hancock"], "window": ["bogus"]})
        assert status == 404, ep
        assert json.loads(body)["error"] == "unknown window"
    status, _, _ = web.api(
        "/api/final_readings", {"site": ["hancock"], "window": ["2026-09-01_2026-09-16"]}
    )
    assert status == 200


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


def test_final_flags_endpoint(data_dir: Path) -> None:
    flag = {
        "id": "hancock-drift-20260902T0100Z",
        "site_id": "hancock",
        "opened_at": "2026-09-02T01:00:00+00:00",
        "latest_seen": "2026-09-05T01:00:00+00:00",
        "event_type": "drift",
        "severity": "watch",
        "residual_ft": 0.31,
        "gage_ft": 10.0,
        "camera_stage_ft": 10.31,
        "camera_sigma_ft": 0.1,
        "persistence_frames": 4,
        "evidence_frames": ["Cam___2026-09-02T01-00-00Z.jpg"],
        "draft_text": "camera reads above the gage",
        "trace_id": None,
        "status": "open",
        "window": "2026-09-01_2026-09-16",
    }
    (data_dir / "results" / "final_flags_hancock.json").write_text(json.dumps([flag]))
    status, _, body = web.api("/api/final_flags", {"site": ["hancock"]})
    assert status == 200
    got = json.loads(body)
    assert len(got) == 1
    assert got[0]["event_type"] == "drift"
    assert got[0]["residual_ft"] == 0.31
    # empty state: a site with no flags file answers []
    status, _, body = web.api("/api/final_flags", {"site": ["oh_wetland"]})
    assert status == 200
    assert json.loads(body) == []


def _trace(tdir: Path, name: str, stamp: str) -> None:
    tdir.mkdir(parents=True, exist_ok=True)
    (tdir / name).write_text(
        json.dumps(
            {
                "trace_id": name[:-5],
                "site_id": "hancock",
                "frame_time": stamp,
                "frame_file": "f.jpg",
                "steps": [],
                "final_status": "accepted",
                "final_stage_ft": 10.0,
            }
        )
    )


def test_final_traces_window_filter(data_dir: Path) -> None:
    (data_dir / "results" / "final_summary_hancock.json").write_text(
        json.dumps(_final_summary_doc({"2026-09-01_2026-09-16": {}}))
    )
    tdir = data_dir / "results" / "final_traces" / "hancock"
    _trace(tdir, "hancock-20260905T1200Z.json", "2026-09-05T12:00:00+00:00")
    _trace(tdir, "hancock-20260420T1200Z.json", "2026-04-20T12:00:00+00:00")
    all_t = web._final_traces("hancock", "all")
    assert all_t is not None and len(all_t) == 2
    win = web._final_traces("hancock", "2026-09-01_2026-09-16")
    assert win is not None and len(win) == 1
    assert str(win[0]["frame_time"]).startswith("2026-09-05")
    assert web._final_traces("hancock", "bogus") is None


def test_final_traces_year_block(data_dir: Path) -> None:
    (data_dir / "results" / "final_summary_in_overland.json").write_text(
        json.dumps(_final_summary_doc({"2026": {}}))
    )
    tdir = data_dir / "results" / "final_traces" / "in_overland"
    _trace(tdir, "in_overland-20260404T1200Z.json", "2026-04-04T12:00:00+00:00")
    _trace(tdir, "in_overland-20250730T1200Z.json", "2025-07-30T12:00:00+00:00")
    win = web._final_traces("in_overland", "2026")
    assert win is not None and len(win) == 1
    assert str(win[0]["frame_time"]).startswith("2026-04-04")


def test_final_traces_empty_states(data_dir: Path) -> None:
    # no summary and no trace dir: 'all' is an empty list, a named window 404s
    assert web._final_traces("oh_wetland", "all") == []
    assert web._final_traces("oh_wetland", "2026") is None
    # summary exists but no saved traces yet: empty list
    (data_dir / "results" / "final_summary_hancock.json").write_text(
        json.dumps(_final_summary_doc({"2026-09-01_2026-09-16": {}}))
    )
    assert web._final_traces("hancock", "all") == []
    status, _, body = web.api("/api/final_traces", {"site": ["hancock"], "window": ["all"]})
    assert status == 200
    assert json.loads(body) == []
