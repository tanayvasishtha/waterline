from datetime import UTC, datetime, timedelta
from itertools import pairwise

import httpx

from tests.conftest import load_fixture
from waterline.models import GageReading
from waterline.usgs import gage

START = datetime(2026, 7, 29, 0, 0, tzinfo=UTC)
END = datetime(2026, 7, 29, 3, 0, tzinfo=UTC)


def _ogc_two_pages(request: httpx.Request) -> httpx.Response:
    if "cursor" in request.url.params:
        return httpx.Response(200, json=load_fixture("ogc_continuous_p2.json"))
    assert request.url.params["monitoring_location_id"] == "USGS-01421500"
    return httpx.Response(200, json=load_fixture("ogc_continuous_p1.json"))


def test_continuous_follows_next_links(mock_client) -> None:
    readings = gage.continuous(mock_client(_ogc_two_pages), "01421500", START, END)
    assert len(readings) == 13
    assert readings[0].value_ft == 9.56
    assert all(a.time < b.time for a, b in pairwise(readings))
    assert {r.approval for r in readings} <= {"Approved", "Provisional"}


def test_continuous_falls_back_to_legacy(mock_client) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if "waterservices" in request.url.host:
            return httpx.Response(200, json=load_fixture("nwis_iv.json"))
        return httpx.Response(503)

    readings = gage.continuous(mock_client(handler), "01421500", START, END)
    assert readings[0].value_ft == 9.56
    assert readings[0].approval == "Provisional"


def test_field_measurements_parse(mock_client) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        page = load_fixture("ogc_field.json")
        assert isinstance(page, dict)
        if {"cursor", "offset"} & set(request.url.params):
            page = {**page, "links": []}
        return httpx.Response(200, json=page)

    client = mock_client(handler)
    readings = gage.field_measurements(client, "01421500", START, END)
    assert readings
    assert all(isinstance(r.value_ft, float) for r in readings)


def _r(minute: int) -> GageReading:
    return GageReading(time=START + timedelta(minutes=minute), value_ft=1.0, approval="Approved")


def test_nearest_within_gap() -> None:
    series = [_r(0), _r(15), _r(30)]
    hit = gage.nearest(series, START + timedelta(minutes=17))
    assert hit is not None
    assert hit.time == START + timedelta(minutes=15)


def test_nearest_outside_gap_is_none() -> None:
    assert gage.nearest([_r(0)], START + timedelta(minutes=40)) is None
    assert gage.nearest([], START) is None
