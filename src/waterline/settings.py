"""Every tunable number in Waterline, each with the reason it has its value.

Values marked "calibration" are refit on each site's calibration window only
(see eval/PROTOCOL.md) and never on test data.
"""

import os
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ENV_PREFIX = "WATERLINE_"


class Settings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    mode: Literal["live", "replay"] = Field(
        "replay", description="replay runs from committed fixtures with no network or AWS"
    )

    # Registration
    sift_features: int = Field(4000, description="enough keypoints on 1080p frames in the spike")
    ratio_test: float = Field(0.7, description="Lowe ratio; dropped ambiguous matches in the spike")
    ransac_reproj_px: float = Field(4.0, description="full-res frames; 4 px kept 238+ inliers")
    min_inliers: int = Field(60, description="spike's worst good frame had 238; 60 leaves margin")
    min_inlier_ratio: float = Field(0.35, description="spike's worst good frame was 0.68")
    shift_alert_px: float = Field(8.0, description="plate corners moving more = camera moved")

    # Plate geometry
    strip_width_px: int = Field(48, description="spans tick marks and digits on the spike plate")
    px_per_ft_target: int = Field(120, description="about the native resolution of the spike plate")

    # Waterline detection
    changepoint_min_segment_px: int = Field(40, description="a third of a foot at the target scale")
    detect_min_contrast: float = Field(0.8, description="below this, call it no boundary")
    plate_visibility_min: float = Field(5.0, description="tick SNR below this is unreadable")
    sigma_floor_ft: float = Field(0.05, description="tick spacing limits precision to about this")
    accept_sigma_ft: float = Field(0.25, description="calibration: wider readings escalate")

    # Frame quality
    glare_frac_max: float = Field(0.08, description="calibration: saturated share of the strip")
    night_mean_sat_max: float = Field(12.0, description="IR night frames are near grayscale")
    blur_var_min: float = Field(40.0, description="calibration: Laplacian variance on the strip")

    # Sensor fault detection
    cusum_k: float = Field(0.5, description="CUSUM slack in sigmas; targets 1-sigma shifts")
    cusum_h: float = Field(5.0, description="CUSUM threshold; one-sided ARL0 near 900 frames")
    flag_persistence_frames: int = Field(4, description="one hour at a 15-minute frame interval")

    # Agent
    agent_step_budget: int = Field(6, description="enough for enhance, alt reference, 2 neighbours")
    bedrock_model_id: str = Field("", description="set per account once model access is granted")
    bedrock_daily_cap: int = Field(300, description="keeps Bedrock spend near $1/day at worst")
    bedrock_max_tokens: int = Field(600, description="a tool call plus a short flag draft")

    # Live operation
    poll_minutes: int = Field(15, description="USGS cameras post every 5 to 60 minutes")

    @classmethod
    def from_env(cls, environ: dict[str, str] | None = None) -> "Settings":
        env = os.environ if environ is None else environ
        overrides = {
            key[len(ENV_PREFIX) :].lower(): value
            for key, value in env.items()
            if key.startswith(ENV_PREFIX)
        }
        return cls.model_validate(overrides)
