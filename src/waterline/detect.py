"""Find the waterline in a rectified plate strip.

Two independent methods so disagreement itself becomes a signal:

- Method A fits a single change point: rows above look like the plate (tick pattern
  energy, plate color, texture) and rows below do not.
- Method B reports the lowest row where the tick pattern still reads, then maps that
  row to stage. It cannot see a waterline where ticks still show, which is exactly
  why disagreement between A and B is useful.

Sigma is a bootstrap of the change-point location, not a guess.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from waterline.models import SiteConfig
from waterline.plate import Strip, row_to_ft, tick_profile
from waterline.settings import Settings


@dataclass(frozen=True)
class Detection:
    row_px: float | None  # strip row of the waterline, or None if none found
    method_a_ft: float | None
    method_b_ft: float | None
    sigma_ft: float
    methods_disagree: bool
    reason: str | None  # why there is no reading


def _plate_score(strip: Strip, site: SiteConfig) -> np.ndarray:
    """Per-row plate-likeness: tick energy plus closeness to the top's plate color."""
    ticks = tick_profile(strip, site.tick_color)

    lab = np.asarray(cv2.cvtColor(strip.image, cv2.COLOR_BGR2LAB), np.float32)
    top_rows = lab[: max(strip.image.shape[0] // 4, 8)]
    plate_color = np.asarray(np.median(top_rows.reshape(-1, 3), axis=0), np.float32)
    color_dist = np.linalg.norm(lab - plate_color, axis=2).mean(axis=1)

    ticks_z = (ticks - ticks.mean()) / (ticks.std() + 1e-6)
    dist_z = (color_dist - color_dist.mean()) / (color_dist.std() + 1e-6)
    return np.asarray(ticks_z - 1.5 * dist_z)


def _change_point(score: np.ndarray, min_seg: int) -> tuple[int, float]:
    """Best single split minimizing within-segment variance; returns (row, contrast)."""
    n = len(score)
    csum = np.concatenate([[0.0], np.cumsum(score)])
    csum2 = np.concatenate([[0.0], np.cumsum(score * score)])

    def seg_cost(lo: int, hi: int) -> float:
        count = hi - lo
        return float(csum2[hi] - csum2[lo] - (csum[hi] - csum[lo]) ** 2 / count)

    costs = np.array(
        [
            seg_cost(0, k) + seg_cost(k, n) if min_seg <= k <= n - min_seg else np.inf
            for k in range(n)
        ]
    )
    k = int(costs.argmin())
    contrast = abs(score[:k].mean() - score[k:].mean()) / (score.std() + 1e-6)
    return k, float(contrast)


def _tick_floor(strip: Strip, site: SiteConfig) -> float | None:
    """Lowest strip row that still carries the tick pattern (method B)."""
    profile = tick_profile(strip, site.tick_color)
    top = profile[: len(profile) // 2]
    threshold = top.mean() + 0.5 * top.std()
    hits = np.flatnonzero(profile > threshold)
    return float(hits.max()) if hits.size else None


def _bootstrap_sigma(
    score: np.ndarray, center: int, min_seg: int, rng: np.random.Generator, n_boot: int = 30
) -> float:
    """Std of the change point when the score is re-noised at its own MAD."""
    mad = np.median(np.abs(score - np.median(score))) * 1.4826
    rows = []
    for _ in range(n_boot):
        noisy = score + rng.normal(0, max(mad, 1e-3), len(score))
        k, _ = _change_point(noisy, min_seg)
        rows.append(k)
    return float(np.std(rows))


def detect(strip: Strip, site: SiteConfig, settings: Settings, seed: int = 0) -> Detection:
    if waterline_is_below_plate(strip, site):
        return Detection(None, None, None, 0.0, False, "below_plate")
    score = _plate_score(strip, site)
    n = len(score)
    min_seg = max(settings.changepoint_min_segment_px, int(n * 0.03))
    k, contrast = _change_point(score, min_seg)

    above_plate = score[:k].mean() > score[k:].mean()
    if contrast < settings.detect_min_contrast or not above_plate:
        return Detection(None, None, None, 0.0, False, "no plate-water boundary on strip")

    row_a = float(k)
    row_b = _tick_floor(strip, site)

    rng = np.random.default_rng(seed)
    sigma_px = _bootstrap_sigma(score, k, min_seg, rng)
    sigma_ft = max(sigma_px / strip.px_per_ft, settings.sigma_floor_ft)

    ft_a = row_to_ft(row_a, strip, site)
    ft_b = row_to_ft(row_b, strip, site) if row_b is not None else None
    disagree = ft_b is not None and abs(ft_a - ft_b) > 2 * sigma_ft

    return Detection(row_a, ft_a, ft_b, sigma_ft, disagree, None)


def waterline_is_below_plate(strip: Strip, site: SiteConfig) -> bool:
    """True when the whole strip reads as plate (water below the board's bottom edge)."""
    ticks = tick_profile(strip, site.tick_color)
    bottom = ticks[int(len(ticks) * 0.9) :]
    top = ticks[: len(ticks) // 2]
    return float(bottom.mean()) > float(top.mean()) * 0.5
