"""The pure perception path: frame -> reading + quality + overlays. No network, no AWS."""

from dataclasses import dataclass
from datetime import datetime

import cv2
import numpy as np

from waterline import enhance
from waterline.detect import Detection, detect
from waterline.models import QualityReport, Reading, SiteConfig
from waterline.plate import Strip, rectify
from waterline.quality import assess
from waterline.register import RefBank, Registration, register, warp
from waterline.settings import Settings


@dataclass(frozen=True)
class Perception:
    """Site + reference bank + thresholds; everything a frame needs to be read."""

    site: SiteConfig
    settings: Settings
    bank: RefBank


@dataclass
class FrameResult:
    reading: Reading
    quality: QualityReport
    registration: Registration | None
    detection: Detection | None
    strip: Strip | None
    warped: np.ndarray | None


def read_frame(
    ctx: Perception,
    image: np.ndarray,
    frame_time: datetime,
    enhance_op: str | None = None,
    seed: int = 0,
) -> FrameResult:
    site, settings, bank = ctx.site, ctx.settings, ctx.bank
    if enhance_op:
        image = enhance.apply_op(image, enhance_op)

    registration = register(image, site, bank, settings)
    if registration is None:
        return FrameResult(
            reading=Reading(
                site_id=site.site_id,
                time=frame_time,
                stage_ft=None,
                sigma_ft=None,
                status="pending",
                reason="registration failed",
            ),
            quality=assess(image, site, settings),
            registration=None,
            detection=None,
            strip=None,
            warped=None,
        )

    ref_w, ref_h = bank.descriptors[registration.ref_name][2]
    warped = warp(image, registration, (ref_w, ref_h))
    strip = rectify(warped, site, settings)
    detection = detect(strip, site, settings, seed=seed)
    quality = assess(image, site, settings, strip, registration)

    offset = site.datum_offset_ft
    a_ft = detection.method_a_ft
    b_ft = detection.method_b_ft
    stage_ft = None if a_ft is None else a_ft + offset
    return FrameResult(
        reading=Reading(
            site_id=site.site_id,
            time=frame_time,
            stage_ft=stage_ft,
            sigma_ft=detection.sigma_ft if a_ft is not None else None,
            method_a_ft=None if a_ft is None else a_ft + offset,
            method_b_ft=None if b_ft is None else b_ft + offset,
            methods_disagree=detection.methods_disagree,
            status="pending",
            reason=detection.reason,
        ),
        quality=quality,
        registration=registration,
        detection=detection,
        strip=strip,
        warped=warped,
    )


def overlay(frame: np.ndarray, result: FrameResult) -> np.ndarray:
    """Draw the detected waterline back onto the original frame."""
    out = frame.copy()
    if result.registration is None or result.detection is None or result.strip is None:
        return out
    if result.detection.row_px is None:
        return out
    row = result.detection.row_px
    ends = np.array([[[0, row]], [[result.strip.image.shape[1], row]]], np.float32)
    # strip -> reference -> frame = inverse of (frame -> reference -> strip)
    to_frame = np.linalg.inv(result.strip.homography @ result.registration.homography)
    pts = cv2.perspectiveTransform(ends, to_frame)
    cv2.polylines(out, [pts.astype(np.int32)], False, (0, 255, 0), 3)
    if result.reading.stage_ft is not None:
        sigma = result.reading.sigma_ft or 0.0
        cv2.putText(
            out,
            f"{result.reading.stage_ft:.2f}ft +- {sigma:.2f}",
            (10, 40),
            0,
            1.0,
            (0, 255, 0),
            2,
        )
    return out
