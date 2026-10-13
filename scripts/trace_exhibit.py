"""Pick the decision exhibit: one trace where OpenCV output changed the plan.

The Agentic Vision qualifier needs a logged trace where a vision observation
(registration, plate visibility, quality gate, uncertainty) steers the next
tool call. Every non-terminal step in these traces is exactly that: the
deterministic policy only calls a tool in response to what the last read
returned, so any trace with steps qualifies. Candidates are additionally
required to end accepted (a substituted read counts, since the trace's final
status is "accepted" on the frame the reading came from).

Selection rule: the trace with the most distinct tools wins; ties go to the
earliest frame time. Writes results/exhibit_trace.md (a step table pairing
each observation with the settings threshold that gates it) and
results/exhibit_trace.json (same content plus the per-tool counts), and
copies the before/after frames out of the cache as results/exhibit_before.jpg
and results/exhibit_after.jpg.

Run from the repo root: uv run python scripts/trace_exhibit.py
"""

import csv
import json
import shutil
import sys
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2

from waterline import eval as ev
from waterline import sites
from waterline.pipeline import read_frame
from waterline.settings import Settings

RESULTS = Path("results")
CACHE = Path("data/cache")
TRACE_ROOT = RESULTS / "final_traces"


def _file_stamp(name: str) -> str:
    stamp = name.rsplit("___", 1)[1].rsplit(".", 1)[0]
    day, clock = stamp.split("T")
    parts = clock.rstrip("Z").split("-")
    return f"{day.replace('-', '')}T{parts[0]}{parts[1]}Z"


def _load_traces() -> list[dict[str, Any]]:
    out = []
    for site_dir in sorted(TRACE_ROOT.iterdir()):
        if not site_dir.is_dir():
            continue
        for path in sorted(site_dir.glob("*.json")):
            out.append(json.loads(path.read_text()))
    return out


def _row_for(
    trace: dict[str, Any], rows: dict[tuple[str, str], dict[str, str]]
) -> dict[str, str] | None:
    """The final agent CSV row whose expected trace file is this trace."""
    return rows.get((str(trace["site_id"]), f"{trace['trace_id']}.json"))


def _threshold_for(decision: str, tool: str) -> str:
    """The settings knob behind the branch that fired, when there is one."""
    s = Settings()
    d = decision.lower()
    rules: list[tuple[tuple[str, ...], str]] = [
        (
            ("night",),
            f"night sat<{s.night_mean_sat_max} or visibility<{s.plate_visibility_min}",
        ),
        (
            ("register", "registration"),
            f"inliers>={s.min_inliers}, ratio>={s.min_inlier_ratio}",
        ),
        (("boundary", "contrast"), f"contrast>={s.detect_min_contrast}"),
        (("glare",), f"glare_frac<={s.glare_action_frac}"),
        (("uncertain", "sigma"), f"sigma<={s.accept_sigma_ft} ft"),
        (("plausible", "jump"), "jump<=rate_max_ft_per_h (site)"),
        (("budget",), f"steps<{s.agent_step_budget}"),
        (("repair rejected",), "repair keeps the reading and gains sigma"),
        (("neighbour unreadable", "confirm"), "neighbour passes the quality gate"),
    ]
    for needles, label in rules:
        if any(n in d for n in needles):
            return label
    return "" if tool in ("accept", "abstain", "human_review") else "policy default"


def _fmt_obs(summary: str) -> str:
    """Format 'stage=.. sigma=.. disagree=.. reason=..' for the table."""
    out = summary
    for key, nd in (("stage", 2), ("sigma", 3)):
        head = key + "="
        if head in out:
            pre, rest = out.split(head, 1)
            val, _, tail = rest.partition(" ")
            with suppress(ValueError):
                val = f"{float(val):.{nd}f}"
            out = pre + head + val + (" " + tail if tail else "")
    return out


def _parse_time(filename: str) -> datetime:
    stamp = filename.rsplit("___", 1)[1].rsplit(".", 1)[0]
    return datetime.strptime(stamp, "%Y-%m-%dT%H-%M-%SZ")


def _initial_obs(site: str, frame_name: str | None) -> str:
    """Recompute the first perceive result on the exhibit's original frame.

    The trace only stores post-tool observations, so the observation that
    drove step 1 comes from rerunning the frozen read on the same pixels.
    """
    if not frame_name:
        return "(initial frame not known)"
    cand = _find_cached(frame_name)
    if cand is None:
        return "(initial frame not in the local cache)"
    image = cv2.imread(str(cand))
    if image is None:
        return "(initial frame unreadable)"
    ctx = ev.make_perception(sites.load_site(site), Settings())
    res = read_frame(ctx, image, _parse_time(frame_name), seed=0)
    r = res.reading
    stage = f"{r.stage_ft:.2f}" if r.stage_ft is not None else "None"
    sigma = f"{r.sigma_ft:.3f}" if r.sigma_ft is not None else "None"
    return f"stage={stage} sigma={sigma} disagree={r.methods_disagree} reason={r.reason}"


def _esc(v: str) -> str:
    """Escape pipes for the markdown table."""
    return v.replace("|", "\\|")


def _neg_time(t: dict[str, Any]) -> float:
    """Earlier frame_time sorts larger."""
    return -datetime.fromisoformat(str(t["frame_time"]).replace("Z", "+00:00")).timestamp()


def _find_cached(name: str) -> Path | None:
    for cam_dir in CACHE.iterdir():
        cand = cam_dir / "overlay" / name
        if cand.exists():
            return cand
    return None


def _copy_frame(label: str, name: str | None) -> str | None:
    if not name:
        return None
    cand = _find_cached(name)
    if cand is None:
        print(f"MISSING {label} {name}")
        return None
    dest = RESULTS / f"exhibit_{label}.jpg"
    shutil.copyfile(cand, dest)
    return f"exhibit_{label}.jpg"


def _write_exhibit(best: dict[str, Any], row: dict[str, str] | None) -> None:
    lines = [
        f"# Decision exhibit: {best['trace_id']}",
        "",
        f"Site {best['site_id']}, frame {best['frame_time']}. One frame's run on",
        "the held-out test windows. Every tool call below was chosen by the",
        "deterministic policy from what OpenCV reported for the current image,",
        "not scripted: each row shows the observation and the threshold that",
        "gated the branch.",
        "",
        "| step | tool | what OpenCV reported | gating threshold | why this tool |",
        "|---|---|---|---|---|",
    ]
    obs_prev = _initial_obs(str(best["site_id"]), (row or {}).get("file"))
    for s in best["steps"]:
        obs = _fmt_obs(obs_prev)
        dec = str(s["decision"])
        thr = _threshold_for(str(s["decision"]), str(s["tool"]))
        lines.append(f"| {s['step']} | {s['tool']} | {_esc(obs)} | {thr} | {_esc(dec)} |")
        obs_prev = str(s["result_summary"])
    stage = best.get("final_stage_ft")
    gage = (row or {}).get("agent_gage_ft") or (row or {}).get("gage_ft")
    final = f"**{best['final_status']}**"
    if isinstance(stage, float):
        final += f" at {stage:.2f} ft"
    if gage:
        final += f" (gage {float(gage):.2f} ft)"
    if best.get("reason"):
        final += f". Reason: {best['reason']}"
    lines += ["", f"Final: {final}.", ""]
    for label, name in (("before", (row or {}).get("file")), ("after", best["frame_file"])):
        img = _copy_frame(label, name)
        if img:
            lines += [f"![{label}]({img})", ""]
    (RESULTS / "exhibit_trace.md").write_text("\n".join(lines))


def main() -> int:
    traces = _load_traces()
    if not traces:
        print("no final traces found; run scripts/final_traces.py first")
        return 1

    # CSV rows keyed by the trace filename they produce
    rows: dict[tuple[str, str], dict[str, str]] = {}
    for site in ("hancock", "in_overland"):
        for path in sorted(RESULTS.glob(f"final_eval_agent_{site}_*.csv")):
            with path.open() as fh:
                for r in csv.DictReader(fh):
                    src = r["agent_frame_file"] or r["file"]
                    rows[(site, f"{site}-{_file_stamp(src)}.json")] = r

    tool_traces: dict[str, int] = {}
    n_with_tools = 0
    candidates = []
    for t in traces:
        steps = t.get("steps", [])
        if not steps:
            continue
        n_with_tools += 1
        tools = {str(s["tool"]) for s in steps}
        for tool in tools:
            tool_traces[tool] = tool_traces.get(tool, 0) + 1
        if t["final_status"] == "accepted":
            candidates.append(t)
    if not candidates:
        print("no qualifying traces (with steps, ending accepted)")
        return 1
    best = max(
        candidates,
        key=lambda t: (len({str(s["tool"]) for s in t["steps"]}), _neg_time(t)),
    )
    print(
        f"exhibit: {best['trace_id']} ({len(best['steps'])} steps, "
        f"{len({s['tool'] for s in best['steps']})} tools)"
    )
    _write_exhibit(best, _row_for(best, rows))
    (RESULTS / "exhibit_trace.json").write_text(
        json.dumps(
            {
                "trace_id": best["trace_id"],
                "site_id": best["site_id"],
                "frame_time": best["frame_time"],
                "final_status": best["final_status"],
                "final_stage_ft": best.get("final_stage_ft"),
                "steps": best["steps"],
                "tools_used": sorted({str(s["tool"]) for s in best["steps"]}),
                "scanned": len(traces),
                "traces_with_tool_steps": n_with_tools,
                "traces_by_tool": tool_traces,
            },
            indent=2,
        )
    )
    print(f"scanned={len(traces)} with_tool_steps={n_with_tools}")
    for tool, n in sorted(tool_traces.items(), key=lambda kv: -kv[1]):
        print(f"  {tool}: {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
