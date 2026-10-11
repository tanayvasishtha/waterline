"""Competition rule guard: OpenCV 5 must do the analysis, and every API we call must exist."""

import cv2
import pytest

# Every OpenCV function Waterline calls. OpenCV 5 moved several 4.x APIs (AKAZE is in contrib),
# so a missing name must fail here, not deep inside a pipeline run.
REQUIRED_APIS = [
    "SIFT_create",
    "BFMatcher",
    "findHomography",
    "warpPerspective",
    "perspectiveTransform",
    "getPerspectiveTransform",
    "cvtColor",
    "inRange",
    "createCLAHE",
    "inpaint",
    "Laplacian",
    "imdecode",
    "imencode",
    "dft",
    "matchTemplate",
    "GaussianBlur",
    "morphologyEx",
    "adaptiveThreshold",
]


def test_opencv_major_version_is_5() -> None:
    assert cv2.__version__.startswith("5."), f"OpenCV 5 required, found {cv2.__version__}"


@pytest.mark.parametrize("name", REQUIRED_APIS)
def test_required_api_exists(name: str) -> None:
    assert hasattr(cv2, name), f"cv2.{name} is missing from OpenCV {cv2.__version__}"
