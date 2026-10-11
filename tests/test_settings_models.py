from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from waterline.models import GageReading, Reading
from waterline.settings import Settings


def test_defaults_are_replay_mode() -> None:
    assert Settings().mode == "replay"


def test_env_override_is_typed() -> None:
    s = Settings.from_env({"WATERLINE_MIN_INLIERS": "90", "WATERLINE_MODE": "live", "PATH": "x"})
    assert s.min_inliers == 90
    assert s.mode == "live"


def test_unknown_env_setting_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings.from_env({"WATERLINE_NOT_A_SETTING": "1"})


def test_bad_mode_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings.from_env({"WATERLINE_MODE": "sometimes"})


def test_gage_reading_rejects_unknown_approval() -> None:
    with pytest.raises(ValidationError):
        GageReading(time=datetime.now(UTC), value_ft=1.0, approval="Maybe")


def test_reading_defaults_to_pending() -> None:
    r = Reading(site_id="x", time=datetime.now(UTC), stage_ft=12.1, sigma_ft=0.1)
    assert r.status == "pending"
    assert not r.methods_disagree
