"""Gage-only episode discovery: no perception, no agent runs.

Earliest NIMS frame per camera, the gage history since, then every on-plate
episode (gage >= threshold, dips under 6 h merged) with a NIMS frame count.
Writes results/episodes.json; checkpoints per site to results/tmp/.
"""

import json
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import ParamSpec, TypedDict, TypeVar

import httpx

from waterline.models import GageReading
from waterline.usgs import gage, nims
from waterline.usgs.http import UsgsError, make_client

PROBE = datetime(2015, 1, 1, tzinfo=UTC)
NOW = datetime.now(UTC)
LIST_LIMIT = 1000
MERGE_GAP = timedelta(hours=6)
CKPT = Path("results/tmp")


class SiteSpec(TypedDict):
    nwis: str
    cam: str
    on_plate: float
    used: list[tuple[str, datetime, datetime]]


class EpisodeRow(TypedDict):
    start: str
    end: str
    duration_h: float
    min_gage_ft: float
    max_gage_ft: float
    nims_frames: int
    overlaps_used: list[str]


SITES: dict[str, SiteSpec] = {
    "hancock": {
        "nwis": "01421500",
        "cam": "NY_East_Branch_Delaware_River_at_Hancock_Staff",
        "on_plate": 10.0,
        "used": [
            (
                "eval 2026-07-27..08-01",
                datetime(2026, 7, 27, tzinfo=UTC),
                datetime(2026, 8, 2, tzinfo=UTC),
            ),
            (
                "calib August 2026",
                datetime(2026, 8, 1, tzinfo=UTC),
                datetime(2026, 9, 1, tzinfo=UTC),
            ),
        ],
    },
    "in_overland": {
        "nwis": "411229084541101",
        "cam": "IN_Overland_Flow_East_of_Bull_Rapids_Road_near_Harlan_STAFF",
        "on_plate": 0.4,
        "used": [
            (
                "calib 2026-07-15..08-15",
                datetime(2026, 7, 15, tzinfo=UTC),
                datetime(2026, 8, 16, tzinfo=UTC),
            ),
            (
                "eval+calib September 2026",
                datetime(2026, 9, 1, tzinfo=UTC),
                datetime(2026, 10, 1, tzinfo=UTC),
            ),
        ],
    },
    "oh_wetland": {
        "nwis": "411607084241200",
        "cam": "OH_Inflow_to_wetland_near_Winn_Road_near_Defiance_STAFF",
        "on_plate": 0.3,
        "used": [
            (
                "calib 2026-08-10..08-18",
                datetime(2026, 8, 10, tzinfo=UTC),
                datetime(2026, 8, 19, tzinfo=UTC),
            ),
            (
                "eval+calib September 2026",
                datetime(2026, 9, 1, tzinfo=UTC),
                datetime(2026, 10, 1, tzinfo=UTC),
            ),
        ],
    },
}

P = ParamSpec("P")
R = TypeVar("R")


def call(fn: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
    """Retry a USGS call through the rolling rate-limit window."""
    for attempt in range(60):
        try:
            return fn(*args, **kwargs)
        except UsgsError as err:
            if "rate limited" not in str(err):
                raise
            print(f"    rate limited; sleeping 65 s (attempt {attempt + 1})", flush=True)
            time.sleep(65)
    raise UsgsError("rate limit never cleared in 65 min")


def earliest_frame(client: httpx.Client, cam: nims.CameraInfo) -> datetime | None:
    """The earliest retrievable frame, binary-searched by month."""
    probe = call(nims.list_frames, client, cam, after=PROBE, limit=100)
    if not probe:
        return None
    lo, hi = PROBE, min(f.time for f in probe)
    while (hi - lo).days > 31:
        mid = lo + (hi - lo) / 2
        if call(nims.list_frames, client, cam, after=lo, before=mid, limit=10):
            hi = min(hi, mid)
        else:
            lo = mid
    final = call(
        nims.list_frames, client, cam, after=PROBE, before=hi + timedelta(days=1), limit=100
    )
    return min(f.time for f in final) if final else hi


def count_frames(client: httpx.Client, cam: nims.CameraInfo, a: datetime, b: datetime) -> int:
    frames = call(nims.list_frames, client, cam, after=a, before=b, limit=LIST_LIMIT)
    if len(frames) < LIST_LIMIT:
        return len(frames)
    mid = a + (b - a) / 2
    return count_frames(client, cam, a, mid) + count_frames(client, cam, mid, b)


def gage_start(client: httpx.Client, nwis: str, start: datetime, end: datetime) -> datetime:
    """Binary-search month windows for the first non-empty gage series."""
    lo, hi = start, end
    if call(gage.continuous, client, nwis, lo, min(lo + timedelta(days=31), hi)):
        return lo
    while (hi - lo).days > 31:
        mid = lo + (hi - lo) / 2
        got = call(gage.continuous, client, nwis, mid, min(mid + timedelta(days=31), hi))
        if got:
            hi = mid
        else:
            lo = mid
    return lo


def gage_series(
    client: httpx.Client, nwis: str, start: datetime, end: datetime
) -> list[GageReading]:
    out: list[GageReading] = []
    t = start
    while t < end:
        nxt = min(t + timedelta(days=365), end)
        try:
            out.extend(call(gage.continuous, client, nwis, t, nxt))
        except UsgsError as err:
            print(f"    year chunk {t.date()} failed ({err}); retrying monthly", flush=True)
            m = t
            while m < nxt:
                mn = min(m + timedelta(days=31), nxt)
                try:
                    out.extend(call(gage.continuous, client, nwis, m, mn))
                except UsgsError as err2:
                    print(f"      month {m.date()} failed: {err2}", flush=True)
                m = mn
        t = nxt
    return sorted({r.time: r for r in out}.values(), key=lambda r: r.time)


def find_episodes(
    readings: list[GageReading], threshold: float
) -> list[tuple[datetime, datetime, float, float]]:
    """Maximal runs of gage >= threshold; sub-6 h dips below merge."""
    eps: list[tuple[datetime, datetime, float, float]] = []
    cur: list[object] | None = None  # [start, end, min, max]
    dip_start: datetime | None = None
    for r in readings:
        on = r.value_ft >= threshold
        if cur is None:
            if on:
                cur = [r.time, r.time, r.value_ft, r.value_ft]
        elif on:
            if dip_start is not None and r.time - dip_start >= MERGE_GAP:
                eps.append(cur)  # type: ignore[arg-type]
                cur = [r.time, r.time, r.value_ft, r.value_ft]
            else:
                cur[1] = r.time
                cur[2] = min(float(cur[2]), r.value_ft)  # type: ignore[arg-type]
                cur[3] = max(float(cur[3]), r.value_ft)  # type: ignore[arg-type]
            dip_start = None
        elif dip_start is None:
            dip_start = r.time
    if cur is not None:
        if dip_start is not None:
            cur[1] = dip_start  # episode ends where it first went below
        eps.append(cur)  # type: ignore[arg-type]
    return eps


def overlaps(a0: datetime, a1: datetime, b0: datetime, b1: datetime) -> bool:
    return a0 < b1 and b0 < a1


def main() -> None:
    client = make_client()
    cams = {c.cam_id: c for c in call(nims.list_cameras, client)}
    CKPT.mkdir(exist_ok=True)
    out: dict[str, object] = {}
    for name, cfg in SITES.items():
        ck = CKPT / f"ep_{name}.json"
        if ck.exists():
            saved = json.loads(ck.read_text())
            if isinstance(saved, dict) and "episodes" in saved:
                print(f"{name}: checkpoint exists, skipping", flush=True)
                out[name] = saved
                continue
        cam = cams[cfg["cam"]]
        # already discovered on earlier runs; skips the binary search
        known = {
            "hancock": "2026-04-14T13:00:04+00:00",
            "in_overland": "2022-11-15T13:56:22+00:00",
            "oh_wetland": "2022-11-13T03:00:29+00:00",
        }
        first = (
            datetime.fromisoformat(known[name]) if name in known else earliest_frame(client, cam)
        )
        print(f"{name}: earliest frame {first}", flush=True)
        if first is None:
            continue
        gstart = gage_start(client, cfg["nwis"], first, NOW)
        print(f"  gage data starts ~{gstart.date()}", flush=True)
        series = gage_series(client, cfg["nwis"], gstart, NOW)
        first_t = series[0].time if series else None
        last_t = series[-1].time if series else None
        print(f"  gage readings: {len(series)} ({first_t} .. {last_t})", flush=True)
        eps = find_episodes(series, cfg["on_plate"])
        print(f"  {len(eps)} episodes; counting frames", flush=True)
        rows: list[EpisodeRow] = json.loads(ck.read_text()) if ck.exists() else []
        done = len(rows)
        if done:
            print(f"  resuming at episode {done + 1}", flush=True)
        for i, (s, e, mn, mx) in enumerate(eps):
            if i < done:
                continue
            n = count_frames(client, cam, s, e)
            used = [lbl for lbl, u0, u1 in cfg["used"] if overlaps(s, e, u0, u1)]
            rows.append(
                {
                    "start": s.isoformat(),
                    "end": e.isoformat(),
                    "duration_h": round((e - s).total_seconds() / 3600, 2),
                    "min_gage_ft": mn,
                    "max_gage_ft": mx,
                    "nims_frames": n,
                    "overlaps_used": used,
                }
            )
            ck.write_text(json.dumps(rows))
            mark = f"  USED:{','.join(used)}" if used else ""
            hours = (e - s).total_seconds() / 3600
            print(
                f"  {s.isoformat()[:16]} -> {e.isoformat()[:16]}  {hours:>8.1f} h"
                f"  gage {mn:>6.2f}-{mx:>6.2f}  frames {n:>5}{mark}",
                flush=True,
            )
        site_out: dict[str, object] = {
            "nwis_id": cfg["nwis"],
            "cam_id": cfg["cam"],
            "on_plate_ft": cfg["on_plate"],
            "earliest_frame": first.isoformat(),
            "episodes": rows,
        }
        ck.write_text(json.dumps(site_out, indent=2))
        out[name] = site_out
    Path("results/episodes.json").write_text(json.dumps(out, indent=2))
    print("wrote results/episodes.json", flush=True)


if __name__ == "__main__":
    main()
