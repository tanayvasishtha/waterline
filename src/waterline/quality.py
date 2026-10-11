"""Frame quality features: is this image readable, and if not, why?

Everything here feeds the agent's triage: a bad frame is not a bad reading,
it is an abstention or a re-look.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from waterline.models import QualityReport, SiteConfig
from waterline.plate import Strip, tick_scale
from waterline.register import Registration
from waterline.settings import Settings


@dataclass(frozen=True)
class Observed:
    """What registration and rectification produced for one frame."""

    strip: Strip
    registration: Registration
    ref_strip: np.ndarray | None


def assess(
    frame: np.ndarray,
    site: SiteConfig,
    settings: Settings,
    obs: Observed | None = None,
) -> QualityReport:
    strip = obs.strip if obs else None
    registration = obs.registration if obs else None
    ref_strip = obs.ref_strip if obs else None
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mean_v = float(hsv[:, :, 2].mean())
    mean_s = float(hsv[:, :, 1].mean())
    is_night = mean_s < settings.night_mean_sat_max or mean_v < 40

    if obs is None:
        return QualityReport(
            mean_v=mean_v,
            mean_s=mean_s,
            blur_var=0.0,
            glare_frac=0.0,
            contrast=0.0,
            plate_visibility=0.0,
            is_night=is_night,
            registration_inliers=0,
            inlier_ratio=0.0,
            shift_px=float("inf"),
        )

    gray = cv2.cvtColor(strip.image, cv2.COLOR_BGR2GRAY)
    blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if ref_strip is not None:
        v_now = cv2.cvtColor(strip.image, cv2.COLOR_BGR2HSV)[:, :, 2].astype(np.int16)
        v_ref = cv2.cvtColor(ref_strip, cv2.COLOR_BGR2HSV)[:, :, 2].astype(np.int16)
        # glare is *new* brightness: sun flare or blown-out water, not the pale plate
        glare = (v_now - v_ref > 40) & (v_now > 200)
    else:
        v_now = cv2.cvtColor(strip.image, cv2.COLOR_BGR2HSV)[:, :, 2].astype(np.int16)
        glare = v_now > 245
    contrast = float(gray.std() / (gray.mean() + 1e-6))

    # Tick periodicity in the upper half of the strip is the plate's own readability signal.
    top = Strip(strip.image[: strip.image.shape[0] // 2], strip.px_per_ft, strip.homography)
    plate_visibility = tick_scale(top, site, site.tick_color).snr

    return QualityReport(
        mean_v=mean_v,
        mean_s=mean_s,
        blur_var=blur_var,
        glare_frac=float(glare.mean()),
        contrast=contrast,
        plate_visibility=plate_visibility,
        is_night=is_night,
        registration_inliers=registration.inliers if registration else 0,
        inlier_ratio=registration.inlier_ratio if registration else 0.0,
        shift_px=registration.shift_px if registration else float("inf"),
    )


def quality_ok(report: QualityReport, settings: Settings) -> bool:
    return (
        report.blur_var >= settings.blur_var_min
        and report.glare_frac <= settings.glare_frac_max
        and report.plate_visibility >= settings.plate_visibility_min
        and not report.is_night
        and report.registration_inliers >= settings.min_inliers
    )
