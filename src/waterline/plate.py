"""Rectify the staff plate into a vertical strip and measure the tick scale.

Rectification uses a homography from the configured plate quad onto a tall, narrow
canvas, so one pixel row always means the same stage everywhere on the plate. The
tick scale is measured independently per frame by FFT on the row-wise tick profile:
periodic marks give a sharp spectral peak at the tick spacing.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from waterline import sites
from waterline.models import SiteConfig
from waterline.settings import Settings


@dataclass(frozen=True)
class Strip:
    image: np.ndarray  # rectified BGR strip, height ~= px_per_ft_target * plate span
    px_per_ft: float  # nominal, from config geometry
    homography: np.ndarray  # reference pixels -> strip pixels


@dataclass(frozen=True)
class TickFit:
    px_per_ft: float
    tick_period_px: float
    snr: float
    ok: bool


def rectify(frame_in_ref: np.ndarray, site: SiteConfig, settings: Settings) -> Strip:
    """Warp the reference-space frame so the plate quad fills a strip image."""
    ref_h, ref_w = frame_in_ref.shape[:2]
    quad = np.asarray(sites.quad_pixels(site, ref_w, ref_h), np.float32)
    span_ft = site.plate_top_ft - site.plate_bottom_ft
    strip_h = round(settings.px_per_ft_target * span_ft)
    dst = np.asarray(
        [[0, 0], [settings.strip_width_px, 0], [settings.strip_width_px, strip_h], [0, strip_h]],
        np.float32,
    )
    homography = cv2.getPerspectiveTransform(quad, dst)
    image = cv2.warpPerspective(frame_in_ref, homography, (settings.strip_width_px, strip_h))
    return Strip(image=image, px_per_ft=strip_h / span_ft, homography=homography)


def row_to_ft(row: float, strip: Strip, site: SiteConfig) -> float:
    return site.plate_top_ft - row / strip.px_per_ft


def ft_to_row(stage_ft: float, strip: Strip, site: SiteConfig) -> float:
    return (site.plate_top_ft - stage_ft) * strip.px_per_ft


def tick_profile(strip: Strip, tick_color: str) -> np.ndarray:
    """Per-row strength of tick-colored pixels in the strip."""
    hsv = cv2.cvtColor(strip.image, cv2.COLOR_BGR2HSV)
    if tick_color == "red":
        lo = cv2.inRange(hsv, np.array([0, 100, 50]), np.array([15, 255, 255]))
        hi = cv2.inRange(hsv, np.array([165, 100, 50]), np.array([180, 255, 255]))
        mask = np.logical_or(np.asarray(lo, bool), np.asarray(hi, bool)).astype(np.uint8) * 255
    else:
        # dark tick marks and numerals on a light plate
        mask = np.asarray(cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 130, 120])))
    return np.asarray(mask.astype(np.float32).mean(axis=1))


def tick_scale(strip: Strip, site: SiteConfig, tick_color: str) -> TickFit:
    """Dominant tick period from the FFT of the row profile, in pixels per foot."""
    profile = tick_profile(strip, tick_color)
    n = len(profile)
    centered = profile - profile.mean()
    centered *= np.hanning(n)
    spectrum = np.abs(np.fft.rfft(centered))
    if spectrum.size < 4 or spectrum.max() == 0:
        return TickFit(strip.px_per_ft, 0.0, 0.0, ok=False)

    # Search near the nominal tick period only: the strongest periodicity on a plate
    # is usually the foot pattern, and locking onto it would double the scale.
    nominal_tick_px = strip.px_per_ft * site.tick_spacing_ft
    lo = max(2, int(n / (nominal_tick_px * 1.35)))
    hi = min(spectrum.size - 1, int(n / max(nominal_tick_px * 0.65, 2)))
    if hi <= lo:
        return TickFit(strip.px_per_ft, 0.0, 0.0, ok=False)
    window = spectrum[lo : hi + 1]
    peak = lo + int(window.argmax())
    # Parabolic refinement around the peak bin.
    if 0 < peak < spectrum.size - 1:
        a, b, c = spectrum[peak - 1 : peak + 2]
        denom = a - 2 * b + c  # negative at a true maximum
        if abs(denom) > 1e-9:
            peak = peak + 0.5 * (a - c) / denom
            peak = min(max(peak, float(lo)), float(hi))
    tick_period_px = n / peak
    peak_bin = min(round(peak), spectrum.size - 1)
    snr = float(spectrum[peak_bin] / (np.median(spectrum[1:]) + 1e-9))
    return TickFit(
        px_per_ft=tick_period_px / site.tick_spacing_ft,
        tick_period_px=tick_period_px,
        snr=snr,
        ok=snr > 3.0,
    )
