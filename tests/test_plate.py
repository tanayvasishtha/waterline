import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from waterline import plate
from waterline.models import SiteConfig
from waterline.settings import Settings

CAM = "NY_East_Branch_Delaware_River_at_Hancock_Staff"
DRY = f"{CAM}___2026-07-27T18-00-12Z.jpg"


@pytest.fixture(scope="module")
def site() -> SiteConfig:
    return SiteConfig.model_validate(json.loads(Path("sites/hancock.json").read_text()))


@pytest.fixture(scope="module")
def strip(site: SiteConfig) -> plate.Strip:
    frame = cv2.imread(str(Path("tests/fixtures/frames") / CAM / DRY))
    return plate.rectify(frame, site, Settings())


def test_strip_has_target_scale(strip: plate.Strip, site: SiteConfig) -> None:
    expected_h = Settings().px_per_ft_target * (site.plate_top_ft - site.plate_bottom_ft)
    assert strip.image.shape == (round(expected_h), Settings().strip_width_px, 3)
    assert abs(strip.px_per_ft - Settings().px_per_ft_target) < 1


def test_tick_scale_recovers_period(strip: plate.Strip, site: SiteConfig) -> None:
    fit = plate.tick_scale(strip, site, site.tick_color)
    assert fit.ok
    # Hancock ticks every 0.5 ft: measured px/ft should agree with geometry within 15%.
    assert abs(fit.px_per_ft - strip.px_per_ft) / strip.px_per_ft < 0.15


def test_tick_scale_handles_blank(strip: plate.Strip, site: SiteConfig) -> None:
    blank = plate.Strip(np.zeros_like(strip.image), strip.px_per_ft, strip.homography)
    assert not plate.tick_scale(blank, site, site.tick_color).ok


def test_row_ft_round_trip(strip: plate.Strip, site: SiteConfig) -> None:
    for ft in (site.plate_top_ft, site.plate_bottom_ft, 15.0):
        assert plate.row_to_ft(plate.ft_to_row(ft, strip, site), strip, site) == pytest.approx(ft)


def test_synthetic_strip_period() -> None:
    # A striped strip with a known period: dark bars every 40 px in a 960 px strip.
    img = np.full((960, 48, 3), 220, np.uint8)
    img[::40] = 30
    img[10::40] = 30
    img[20::40] = 30
    strip = plate.Strip(img, px_per_ft=160.0, homography=np.eye(3))
    cfg = json.loads(Path("sites/hancock.json").read_text())
    cfg["tick_spacing_ft"] = 0.25  # 40 px per tick -> 160 px/ft
    site = SiteConfig.model_validate(cfg)
    fit = plate.tick_scale(strip, site, "black")
    assert fit.ok
    assert fit.px_per_ft == pytest.approx(160.0, rel=0.05)
