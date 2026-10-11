"""USGS National Imagery Management System: cameras, frame listings, and cached downloads.

API docs: https://api.waterdata.usgs.gov/docs/nims/
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import cv2
import httpx
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from waterline.models import FrameRef
from waterline.usgs.http import UsgsError, get_bytes, get_json

API = "https://api.waterdata.usgs.gov/nims/v0"
FILENAME_TIME_FORMAT = "%Y-%m-%dT%H-%M-%SZ"
ImageSize = Literal["full", "720", "thumb"]


class CameraInfo(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore", populate_by_name=True)

    cam_id: str = Field(alias="camId")
    nwis_id: str | None = Field(None, alias="nwisId")
    name: str = Field(alias="camName")
    description: str = Field("", alias="camDesc")
    tz: str = "UTC"
    newest_image: datetime | None = Field(None, alias="newestImageDT")
    hidden: bool = Field(False, alias="hideCam")
    full_dir: str = Field(alias="overlayDir")
    small_dir: str = Field(alias="smallDir")
    thumb_dir: str = Field(alias="thumbDir")


def frame_time(filename: str) -> datetime:
    """Parse the UTC capture time from a NIMS filename like `CAM___2026-07-29T16-00-18Z.jpg`."""
    try:
        stamp = filename.rsplit("___", 1)[1].removesuffix(".jpg")
        return datetime.strptime(stamp, FILENAME_TIME_FORMAT).replace(tzinfo=UTC)
    except (IndexError, ValueError) as error:
        raise UsgsError(f"unexpected NIMS filename: {filename}") from error


def image_url(camera: CameraInfo, filename: str, size: ImageSize = "full") -> str:
    base = {"full": camera.full_dir, "720": camera.small_dir, "thumb": camera.thumb_dir}[size]
    return base + filename


def list_cameras(client: httpx.Client) -> list[CameraInfo]:
    payload = get_json(client, f"{API}/cameras")
    if not isinstance(payload, list):
        raise UsgsError("NIMS /cameras did not return a list")
    return [CameraInfo.model_validate(item) for item in payload]


def list_frames(
    client: httpx.Client,
    camera: CameraInfo,
    *,
    after: datetime | None = None,
    before: datetime | None = None,
    limit: int = 1000,
    newest_first: bool = False,
    size: ImageSize = "full",
) -> list[FrameRef]:
    params = {"camId": camera.cam_id, "limit": str(limit), "recent": str(newest_first).lower()}
    if after:
        params["after"] = after.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    if before:
        params["before"] = before.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    payload = get_json(client, f"{API}/listFiles", params)
    if not isinstance(payload, list):
        raise UsgsError("NIMS /listFiles did not return a list")
    return [
        FrameRef(
            cam_id=camera.cam_id,
            filename=name,
            time=frame_time(name),
            url=image_url(camera, name, size),
        )
        for name in payload
        if isinstance(name, str) and name.endswith(".jpg")
    ]


def fetch_image(client: httpx.Client, frame: FrameRef, cache_dir: Path) -> np.ndarray:
    """Return the frame as BGR, downloading once and reusing the on-disk copy afterwards."""
    size_dir = frame.url.split("/")[-3]
    path = cache_dir / frame.cam_id / size_dir / frame.filename
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(get_bytes(client, frame.url))
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        path.unlink(missing_ok=True)
        raise UsgsError(f"could not decode image {frame.filename}")
    return image
