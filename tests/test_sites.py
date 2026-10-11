import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from waterline import sites
from waterline.models import SiteConfig


def _config(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "site_id": "test",
        "cam_id": "CAM",
        "nwis_id": "00000000",
        "name": "Test site",
        "tz": "America/New_York",
        "reference_frames": {"day": "ref_day.jpg"},
        "plate_quad": [[0.1, 0.2], [0.3, 0.2], [0.3, 0.9], [0.1, 0.9]],
        "plate_top_ft": 20.0,
        "plate_bottom_ft": 2.0,
        "tick_color": "auto",
        "static_mask": [[[0.0, 0.0], [0.4, 0.0], [0.4, 0.5], [0.0, 0.5]]],
        "datum_offset_ft": -0.8,
    }
    base.update(overrides)
    return base


def test_load_site(tmp_path: Path) -> None:
    (tmp_path / "test.json").write_text(json.dumps(_config()))
    site = sites.load_site("test", tmp_path)
    assert site.nwis_id == "00000000"
    assert not sites.validate_normalized(site)


def test_quad_scales_to_pixels() -> None:
    site = SiteConfig.model_validate(_config())
    quad = sites.quad_pixels(site, 2000, 1000)
    assert quad[2] == (600.0, 900.0)


def test_inverted_plate_is_flagged() -> None:
    site = SiteConfig.model_validate(_config(plate_top_ft=1.0, plate_bottom_ft=5.0))
    assert "plate_top_ft" in sites.validate_normalized(site)[0]


def test_out_of_range_coordinate_is_flagged() -> None:
    bad = _config()
    bad["plate_quad"][0] = [1.2, 0.2]  # type: ignore[index]
    site = SiteConfig.model_validate(bad)
    assert any("quad" in p for p in sites.validate_normalized(site))


def test_missing_config_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        sites.load_site("ghost", tmp_path)


def test_missing_fields_rejected() -> None:
    with pytest.raises(ValidationError):
        SiteConfig.model_validate({"site_id": "x"})
