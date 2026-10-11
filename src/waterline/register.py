"""Align a camera frame to the site reference so plate coordinates stay fixed.

The camera drifts a few pixels between frames and occasionally gets bumped. SIFT
features are kept only inside static_mask polygons (walls, rails, the plate board),
never on water or sky, so a flood cannot fake the alignment.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from waterline import sites
from waterline.models import SiteConfig
from waterline.settings import Settings


@dataclass(frozen=True)
class Registration:
    ref_name: str
    homography: np.ndarray  # frame pixels -> reference pixels
    inliers: int
    matches: int
    inlier_ratio: float
    reproj_rmse_px: float
    shift_px: float


@dataclass
class RefBank:
    """Per-reference SIFT features, computed once per site."""

    descriptors: dict[str, tuple[Sequence[cv2.KeyPoint], np.ndarray, tuple[int, int]]] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class _Fit:
    """One candidate homography against one reference image."""

    ref_name: str
    ref_size: tuple[int, int]
    frame_size: tuple[int, int]
    homography: np.ndarray
    matches: int
    inlier_mask: np.ndarray
    src: np.ndarray
    dst: np.ndarray


def _mask_for(site: SiteConfig, width: int, height: int) -> np.ndarray:
    mask = np.zeros((height, width), np.uint8)
    polys = [(np.asarray(poly) * (width, height)).astype(np.int32) for poly in site.static_mask]
    cv2.fillPoly(mask, polys, 255)
    return mask


def _detect(
    image: np.ndarray, mask: np.ndarray, sift: cv2.SIFT
) -> tuple[Sequence[cv2.KeyPoint], np.ndarray | None]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    keypoints, desc = sift.detectAndCompute(gray, mask)
    return keypoints, None if desc is None else np.asarray(desc)


def build_reference(site: SiteConfig, sites_dir: Path, settings: Settings) -> RefBank:
    bank = RefBank()
    sift = cv2.SIFT.create(nfeatures=settings.sift_features)
    mask_cache: dict[tuple[int, int], np.ndarray] = {}
    for name, filename in site.reference_frames.items():
        path = sites_dir / site.site_id / filename
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(f"reference image missing: {path}")
        h, w = image.shape[:2]
        mask = mask_cache.setdefault((w, h), _mask_for(site, w, h))
        keypoints, desc = _detect(image, mask, sift)
        if desc is None or len(keypoints) < settings.min_inliers:
            count = 0 if desc is None else len(desc)
            raise ValueError(f"too few features ({count}) in {path}")
        bank.descriptors[name] = (keypoints, desc, (w, h))
    return bank


def _match(frame_desc: np.ndarray, ref_desc: np.ndarray, ratio: float) -> list[cv2.DMatch]:
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(frame_desc, ref_desc, k=2)
    return [m for m, n in pairs if m.distance < ratio * n.distance]


def register(
    frame: np.ndarray,
    site: SiteConfig,
    bank: RefBank,
    settings: Settings,
    exclude: frozenset[str] = frozenset(),
) -> Registration | None:
    """Homography from frame pixels into the best-matching reference, or None."""
    h, w = frame.shape[:2]
    frame_kps, frame_desc = _detect(frame, _mask_for(site, w, h), cv2.SIFT.create())
    if frame_desc is None or len(frame_kps) < settings.min_inliers:
        return None

    best: Registration | None = None
    for ref_name, (ref_kps, ref_desc, ref_size) in bank.descriptors.items():
        if ref_name in exclude:
            continue
        matches = _match(frame_desc, ref_desc, settings.ratio_test)
        if len(matches) < settings.min_inliers:
            continue
        src = np.asarray([frame_kps[m.queryIdx].pt for m in matches], np.float32)
        dst = np.asarray([ref_kps[m.trainIdx].pt for m in matches], np.float32)
        homography, inlier_mask = cv2.findHomography(
            src, dst, cv2.RANSAC, settings.ransac_reproj_px
        )
        if (
            homography is None
            or inlier_mask is None
            or int(inlier_mask.sum()) < settings.min_inliers
        ):
            continue
        fit = _Fit(ref_name, ref_size, (w, h), homography, len(matches), inlier_mask, src, dst)
        candidate = _to_registration(fit, site)
        if best is None or candidate.inliers > best.inliers:
            best = candidate
    return best


def _to_registration(fit: _Fit, site: SiteConfig) -> Registration:
    keep = fit.inlier_mask.ravel() > 0
    projected = cv2.perspectiveTransform(fit.src[keep].reshape(-1, 1, 2), fit.homography)
    errors = np.linalg.norm(projected - fit.dst[keep].reshape(-1, 1, 2), axis=2)
    rmse = float(np.sqrt((errors**2).mean()))
    # A camera bump shows up when the reference plate quad, mapped into this frame,
    # lands away from where the site config expects it.
    ref_quad = np.asarray(sites.quad_pixels(site, *fit.ref_size), np.float32)
    mapped = cv2.perspectiveTransform(
        ref_quad.reshape(-1, 1, 2), np.linalg.inv(fit.homography)
    ).reshape(-1, 2)
    expected = np.asarray(sites.quad_pixels(site, *fit.frame_size), np.float32)
    return Registration(
        ref_name=fit.ref_name,
        homography=fit.homography,
        inliers=int(keep.sum()),
        matches=fit.matches,
        inlier_ratio=float(keep.sum()) / fit.matches,
        reproj_rmse_px=rmse,
        shift_px=float(np.abs(mapped - expected).mean()),
    )


def warp(frame: np.ndarray, registration: Registration, out_size: tuple[int, int]) -> np.ndarray:
    return cv2.warpPerspective(frame, registration.homography, out_size)
