from datetime import UTC, datetime
from pathlib import Path

import cv2
import httpx
import numpy as np
import pytest

from tests.conftest import load_fixture
from waterline.usgs import nims
from waterline.usgs.http import UsgsError

HANCOCK = "NY_East_Branch_Delaware_River_at_Hancock_Staff"


def _routes(request: httpx.Request) -> httpx.Response:
    if request.url.path.endswith("/cameras"):
        return httpx.Response(200, json=load_fixture("nims_cameras.json"))
    if request.url.path.endswith("/listFiles"):
        assert request.url.params["camId"] == HANCOCK
        return httpx.Response(200, json=load_fixture("nims_listfiles.json"))
    return httpx.Response(404)


def test_frame_time_parses_utc() -> None:
    t = nims.frame_time(f"{HANCOCK}___2026-07-29T16-00-18Z.jpg")
    assert t == datetime(2026, 7, 29, 16, 0, 18, tzinfo=UTC)


def test_frame_time_rejects_bad_name() -> None:
    with pytest.raises(UsgsError):
        nims.frame_time("not_a_frame.jpg")


def test_list_cameras_and_frames(mock_client) -> None:
    client = mock_client(_routes)
    cameras = {c.cam_id: c for c in nims.list_cameras(client)}
    assert cameras[HANCOCK].nwis_id == "01421500"
    frames = nims.list_frames(client, cameras[HANCOCK], limit=5)
    assert len(frames) == 5
    assert frames[0].url.startswith(cameras[HANCOCK].full_dir)
    assert frames == sorted(frames, key=lambda f: f.time)


def test_image_url_sizes(mock_client) -> None:
    camera = nims.list_cameras(mock_client(_routes))[0]
    assert "/720/" in nims.image_url(camera, "x.jpg", "720")
    assert "/thumbnail/" in nims.image_url(camera, "x.jpg", "thumb")


def test_fetch_image_caches(tmp_path: Path, mock_client) -> None:
    ok, jpg = cv2.imencode(".jpg", np.full((8, 8, 3), 200, np.uint8))
    assert ok
    calls = []

    def serve(request: httpx.Request) -> httpx.Response:
        calls.append(request.url)
        return httpx.Response(200, content=jpg.tobytes())

    client = mock_client(serve)
    camera = next(c for c in nims.list_cameras(mock_client(_routes)) if c.cam_id == HANCOCK)
    frame = nims.list_frames(mock_client(_routes), camera, limit=1)[0]
    first = nims.fetch_image(client, frame, tmp_path)
    second = nims.fetch_image(client, frame, tmp_path)
    assert first.shape == (8, 8, 3)
    assert np.array_equal(first, second)
    assert len(calls) == 1


def test_http_error_is_one_line(mock_client) -> None:
    client = mock_client(lambda request: httpx.Response(503))
    with pytest.raises(UsgsError) as info:
        nims.list_cameras(client)
    assert "\n" not in str(info.value)
