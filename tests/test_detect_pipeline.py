import json
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
import pytest

from waterline import detect, plate, sites
from waterline.models import SiteConfig
from waterline.pipeline import Perception, overlay, read_frame
from waterline.register import build_reference
from waterline.settings import Settings

CAM = "NY_East_Branch_Delaware_River_at_Hancock_Staff"
MANIFEST = Path("tests/fixtures/manifest_hancock.json")


@pytest.fixture(scope="module")
def site() -> SiteConfig:
    return sites.load_site("hancock")


@pytest.fixture(scope="module")
def ctx(site: SiteConfig) -> Perception:
    bank = build_reference(site, Path("sites"), Settings())
    return Perception(site, Settings(), bank)


def _manifest() -> list[dict]:
    return json.loads(MANIFEST.read_text())


@pytest.mark.parametrize("entry", _manifest(), ids=lambda e: e["filename"][-15:])
def test_fixture_readings_match_gage(entry: dict, ctx: Perception, site: SiteConfig) -> None:
    image = cv2.imread(str(Path("tests/fixtures") / entry["path"]))
    result = read_frame(ctx, image, datetime.fromisoformat(entry["time"]))
    gage = entry["gage_ft"]
    if gage < site.plate_bottom_ft + 0.3:
        # At or below the plate's physical bottom: "below_plate" or a reading
        # right at the bottom edge are both honest answers.
        if result.reading.stage_ft is not None:
            assert abs(result.reading.stage_ft - gage) < 0.6, entry["filename"]
    else:
        assert result.reading.stage_ft is not None
        assert abs(result.reading.stage_ft - gage) < 0.6, entry["filename"]


def test_synthetic_boundary_detected(site: SiteConfig) -> None:
    # Top half plate-like (striped), bottom half dark water.
    img = np.zeros((600, 48, 3), np.uint8)
    img[:300] = 210
    for y in range(0, 300, 40):
        img[y : y + 8] = (50, 50, 200)  # red ticks (BGR)
    img[300:] = (40, 30, 20)
    strip = plate.Strip(img, px_per_ft=160.0, homography=np.eye(3))
    found = detect.detect(strip, site, Settings())
    assert found.row_px == pytest.approx(300, abs=15)


def test_all_plate_strip_reports_below(site: SiteConfig) -> None:
    img = np.full((600, 48, 3), 210, np.uint8)
    for y in range(0, 600, 40):
        img[y : y + 8] = (50, 50, 200)
    strip = plate.Strip(img, px_per_ft=160.0, homography=np.eye(3))
    found = detect.detect(strip, site, Settings())
    assert found.reason == "below_plate"


def test_overlay_draws_line(ctx: Perception) -> None:
    entry = _manifest()[2]  # flood frame
    image = cv2.imread(str(Path("tests/fixtures") / entry["path"]))
    result = read_frame(ctx, image, datetime.now(UTC))
    drawn = overlay(image, result)
    assert (drawn != image).any()
