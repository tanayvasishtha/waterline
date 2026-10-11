"""Typed records passed between modules, stored, and returned by the API."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Point = tuple[float, float]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class GageReading(_Frozen):
    time: datetime
    value_ft: float
    approval: Literal["Approved", "Provisional"]
    qualifier: str | None = None


class FrameRef(_Frozen):
    cam_id: str
    filename: str
    time: datetime
    url: str


class SiteConfig(_Frozen):
    site_id: str
    cam_id: str
    nwis_id: str
    name: str
    tz: str
    reference_frames: dict[str, str] = Field(min_length=1, description="lighting -> image file")
    plate_quad: tuple[Point, Point, Point, Point] = Field(
        description="plate corners as fractions of image size, TL TR BR BL"
    )
    plate_top_ft: float = Field(description="stage value printed at the top edge of plate_quad")
    plate_bottom_ft: float = Field(description="stage value at the bottom edge of plate_quad")
    tick_color: Literal["red", "black", "auto"] = "auto"
    tick_spacing_ft: float = Field(0.5, description="stage between adjacent plate marks")
    static_mask: list[list[Point]] = Field(
        min_length=1, description="fixed-structure polygons as image fractions"
    )
    datum_offset_ft: float = 0.0
    sigma_calib_ft: float = Field(0.15, description="residual std from calibration")
    nws_flood_stage_ft: float | None = None


class QualityReport(_Frozen):
    mean_v: float
    mean_s: float
    blur_var: float
    glare_frac: float
    contrast: float
    plate_visibility: float
    is_night: bool
    registration_inliers: int
    inlier_ratio: float
    shift_px: float


ReadingStatus = Literal["accepted", "abstained", "pending"]


class Reading(_Frozen):
    site_id: str
    time: datetime
    stage_ft: float | None
    sigma_ft: float | None
    method_a_ft: float | None = None
    method_b_ft: float | None = None
    methods_disagree: bool = False
    status: ReadingStatus = "pending"
    reason: str | None = None


class AgentStep(_Frozen):
    step: int
    tool: str
    args: dict[str, object]
    result_summary: str
    decision: str
    latency_ms: float
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    fallback: bool = False


FlagStatus = Literal["open", "approved", "dismissed"]


class Flag(_Frozen):
    id: str
    site_id: str
    opened_at: datetime
    residual_ft: float
    evidence_frames: list[str]
    draft_text: str
    status: FlagStatus = "open"
    reviewer: str | None = None
    note: str | None = None
