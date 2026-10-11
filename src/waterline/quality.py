"""Frame quality features: is this image readable, and if not, why?

Everything here feeds the agent's triage: a bad frame is not a bad reading,
it is an abstention or a re-look.
"""

import cv2
import numpy as np

from waterline.models import QualityReport, SiteConfig
from waterline.plate import Strip, tick_scale
from waterline.register import Registration
from waterline.settings import Settings


def assess(
    frame: np.ndarray,
    site: SiteConfig,
    settings: Settings,
    strip: Strip | None = None,
    registration: Registration | None = None,
) -> QualityReport:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mean_v = float(hsv[:, :, 2].mean())
    mean_s = float(hsv[:, :, 1].mean())
    is_night = mean_s < settings.night_mean_sat_max or mean_v < 40

    if strip is None:
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
    strip_hsv = cv2.cvtColor(strip.image, cv2.COLOR_BGR2HSV)
    glare = (strip_hsv[:, :, 2] > 240) & (strip_hsv[:, :, 1] < 40)
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
        and not report.is_night
        and report.registration_inliers >= settings.min_inliers
    )
