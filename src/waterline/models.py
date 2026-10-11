"""Typed records passed between modules, stored, and returned by the API."""

from datetime import datetime
from enum import StrEnum
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


ReadingStatus = Literal["accepted", "abstained", "pending", "below_range"]


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
    """A verified camera-vs-sensor disagreement for human review. The auditable product."""

    id: str
    site_id: str
    opened_at: datetime
    latest_seen: datetime | None = None
    event_type: Literal["drift", "stuck", "step", "spike", "disagreement"] = "disagreement"
    severity: Literal["watch", "alert"] = "watch"
    residual_ft: float
    gage_ft: float | None = None
    camera_stage_ft: float | None = None
    camera_sigma_ft: float | None = None
    persistence_frames: int = 0
    evidence_frames: list[str]
    draft_text: str
    trace_id: str | None = None
    status: FlagStatus = "open"
    reviewer: str | None = None
    note: str | None = None


class AgentAction(StrEnum):
    """One step the agent may take. Terminal states are not actions."""

    ENHANCE_CLAHE = "enhance_clahe"
    ENHANCE_GAMMA = "enhance_gamma"
    MASK_GLARE = "mask_glare"
    ALT_REFERENCE = "alt_reference"
    FETCH_PREV = "fetch_prev"
    FETCH_NEXT = "fetch_next"
    ACCEPT = "accept"
    ABSTAIN = "abstain"
    HUMAN_REVIEW = "human_review"


class Trace(BaseModel):
    """The full record of one frame's agentic run -- the audit artifact."""

    trace_id: str
    site_id: str
    frame_time: datetime
    frame_file: str
    started_at: datetime
    finished_at: datetime
    steps: list[AgentStep]
    final_status: str
    final_stage_ft: float | None
    final_sigma_ft: float | None
    reason: str | None
    policy: Literal["deterministic", "llm", "llm_fallback"]
    llm_calls: int = 0
