import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from waterline import register as reg
from waterline.models import SiteConfig
from waterline.settings import Settings

CAM = "NY_East_Branch_Delaware_River_at_Hancock_Staff"
FIX = Path(__file__).parent / "fixtures" / "frames" / CAM
REF = f"{CAM}___2026-07-27T18-00-12Z.jpg"
OTHER = f"{CAM}___2026-07-30T18-00-12Z.jpg"


def _frame(name: str) -> np.ndarray:
    return cv2.imread(str(FIX / name))


@pytest.fixture(scope="module")
def site() -> SiteConfig:
    cfg = json.loads(Path("sites/hancock.json").read_text())
    cfg["reference_frames"] = {"day": REF}
    return SiteConfig.model_validate(cfg)


@pytest.fixture(scope="module")
def bank(site: SiteConfig) -> reg.RefBank:
    image = _frame(REF)
    mask = reg._mask_for(site, image.shape[1], image.shape[0])
    kps, desc = reg._detect(image, mask, cv2.SIFT.create())
    assert desc is not None
    bank = reg.RefBank()
    bank.descriptors["day"] = (kps, desc, (image.shape[1], image.shape[0]))
    return bank


def test_self_registration_is_identity(site: SiteConfig, bank: reg.RefBank) -> None:
    result = reg.register(_frame(REF), site, bank, Settings())
    assert result is not None
    assert result.inliers >= Settings().min_inliers
    assert result.reproj_rmse_px < 2.0
    assert result.shift_px < 2.0


def test_synthetic_shift_is_recovered(site: SiteConfig, bank: reg.RefBank) -> None:
    frame = _frame(REF)
    moved = cv2.warpAffine(
        frame, np.float32([[1, 0, -25], [0, 1, 10]]), (frame.shape[1], frame.shape[0])
    )
    result = reg.register(moved, site, bank, Settings())
    assert result is not None
    translation = result.homography[:2, 2]
    assert np.linalg.norm(translation - [25.0, -10.0]) < 8.0


def test_blank_frame_is_rejected(site: SiteConfig, bank: reg.RefBank) -> None:
    blank = np.zeros((600, 800, 3), np.uint8)
    assert reg.register(blank, site, bank, Settings()) is None


def test_real_other_frame_registers(site: SiteConfig, bank: reg.RefBank) -> None:
    result = reg.register(_frame(OTHER), site, bank, Settings())
    assert result is not None
    assert result.inlier_ratio >= Settings().min_inlier_ratio
