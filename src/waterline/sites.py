"""Load and validate per-site configuration: reference frames, plate geometry, masks."""

import json
import math
import os
from pathlib import Path

from waterline.models import SiteConfig

SITES_DIR = Path(
    os.environ.get("WATERLINE_SITES_DIR") or Path(__file__).resolve().parents[2] / "sites"
)


def site_dir(site_id: str) -> Path:
    return SITES_DIR / site_id


def load_site(site_id: str, sites_dir: Path = SITES_DIR) -> SiteConfig:
    path = sites_dir / f"{site_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"no site config at {path}")
    return SiteConfig.model_validate(json.loads(path.read_text(encoding="utf-8")))


def list_sites(sites_dir: Path = SITES_DIR) -> list[str]:
    if not sites_dir.exists():
        return []
    return sorted(p.stem for p in sites_dir.glob("*.json"))


def quad_pixels(site: SiteConfig, width: float, height: float) -> list[tuple[float, float]]:
    """Plate corners stored as fractions of image size, scaled to real pixels."""
    return [(x * width, y * height) for x, y in site.plate_quad]


def validate_normalized(site: SiteConfig) -> list[str]:
    """Every normalized coordinate must sit inside the unit square."""
    problems = []
    for i, (x, y) in enumerate(site.plate_quad):
        if not (0 <= x <= 1 and 0 <= y <= 1):
            problems.append(f"plate_quad[{i}]={x:.3f},{y:.3f} outside image")
    for poly_i, poly in enumerate(site.static_mask):
        for j, (x, y) in enumerate(poly):
            if not (0 <= x <= 1 and 0 <= y <= 1):
                problems.append(f"static_mask[{poly_i}][{j}] outside image")
    if not site.plate_top_ft > site.plate_bottom_ft or not math.isfinite(site.plate_top_ft):
        problems.append("plate_top_ft must exceed plate_bottom_ft")
    return problems
