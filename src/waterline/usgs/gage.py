"""USGS gage height: continuous sensor values and technician field measurements.

Primary source is the OGC Water Data API (https://api.waterdata.usgs.gov/ogcapi/v1).
If it fails, continuous values fall back to the legacy NWIS instantaneous-values service.
Both sources are parsed into the same GageReading records.
"""

from bisect import bisect_left
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

import httpx

from waterline.models import GageReading
from waterline.usgs.http import UsgsError, get_json

OGC = "https://api.waterdata.usgs.gov/ogcapi/v1/collections"
LEGACY_IV = "https://waterservices.usgs.gov/nwis/iv/"
GAGE_HEIGHT = "00065"
PAGE_LIMIT = 10000
MAX_PAGES = 200
NO_DATA = -999999.0

Json = dict[str, Any]


def _iso(t: datetime) -> str:
    return t.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _qualifier(raw: object) -> str | None:
    if isinstance(raw, list):
        return ",".join(str(q) for q in raw) or None
    return str(raw) if raw else None


def _approval(raw: object) -> Literal["Approved", "Provisional"]:
    return "Approved" if str(raw).lower().startswith("a") else "Provisional"


def _ogc_features(client: httpx.Client, collection: str, params: dict[str, str]) -> list[Json]:
    url: str | None = f"{OGC}/{collection}/items"
    query: dict[str, str] | None = {"f": "json", "limit": str(PAGE_LIMIT), **params}
    features: list[Json] = []
    for _ in range(MAX_PAGES):
        if url is None:
            return features
        page = get_json(client, url, query)
        if not isinstance(page, dict):
            raise UsgsError(f"OGC {collection} returned a non-object page")
        features.extend(page.get("features", []))
        url = next((link["href"] for link in page.get("links", []) if link["rel"] == "next"), None)
        query = None  # the next link already carries every parameter
    raise UsgsError(f"OGC {collection} paging exceeded {MAX_PAGES} pages")


def _from_ogc(features: list[Json]) -> list[GageReading]:
    readings = []
    for feature in features:
        p = feature["properties"]
        if p.get("value") in (None, ""):
            continue
        readings.append(
            GageReading(
                time=datetime.fromisoformat(p["time"]),
                value_ft=float(p["value"]),
                approval=_approval(p.get("approval_status")),
                qualifier=_qualifier(p.get("qualifier")),
            )
        )
    return sorted(readings, key=lambda r: r.time)


def _from_legacy(payload: object) -> list[GageReading]:
    try:
        series: Json = payload["value"]["timeSeries"][0]  # type: ignore[index]  # untyped JSON
        values: list[Json] = series["values"][0]["value"]
    except (KeyError, IndexError, TypeError) as error:
        raise UsgsError("NWIS IV response had no gage height series") from error
    readings = []
    for item in values:
        value = float(item["value"])
        if value <= NO_DATA:
            continue
        codes = item.get("qualifiers", [])
        readings.append(
            GageReading(
                time=datetime.fromisoformat(item["dateTime"]),
                value_ft=value,
                approval="Approved" if "A" in codes else "Provisional",
                qualifier=_qualifier([c for c in codes if c not in ("A", "P")]),
            )
        )
    return sorted(readings, key=lambda r: r.time)


def continuous(
    client: httpx.Client, nwis_id: str, start: datetime, end: datetime
) -> list[GageReading]:
    """Sensor gage height between start and end (inclusive), oldest first."""
    try:
        features = _ogc_features(
            client,
            "continuous",
            {
                "monitoring_location_id": f"USGS-{nwis_id}",
                "parameter_code": GAGE_HEIGHT,
                "datetime": f"{_iso(start)}/{_iso(end)}",
            },
        )
        return _from_ogc(features)
    except UsgsError:
        payload = get_json(
            client,
            LEGACY_IV,
            {
                "format": "json",
                "sites": nwis_id,
                "parameterCd": GAGE_HEIGHT,
                "startDT": _iso(start),
                "endDT": _iso(end),
            },
        )
        return _from_legacy(payload)


def field_measurements(
    client: httpx.Client, nwis_id: str, start: datetime, end: datetime
) -> list[GageReading]:
    """Gage height read by a USGS technician on site visits (independent of the sensor)."""
    features = _ogc_features(
        client,
        "field-measurements",
        {
            "monitoring_location_id": f"USGS-{nwis_id}",
            "parameter_code": GAGE_HEIGHT,
            "datetime": f"{_iso(start)}/{_iso(end)}",
        },
    )
    return _from_ogc(features)


def nearest(
    readings: Sequence[GageReading], t: datetime, max_gap: timedelta = timedelta(minutes=10)
) -> GageReading | None:
    """The reading closest to t, if one lies within max_gap. `readings` must be sorted by time."""
    times = [r.time for r in readings]
    i = bisect_left(times, t)
    candidates = [readings[j] for j in (i - 1, i) if 0 <= j < len(readings)]
    best = min(candidates, key=lambda r: abs(r.time - t), default=None)
    return best if best is not None and abs(best.time - t) <= max_gap else None
