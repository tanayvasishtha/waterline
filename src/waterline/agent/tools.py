"""Frame-level tools the agent invokes. Each maps one action to a concrete step."""

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from waterline import enhance
from waterline.models import AgentAction, FrameRef
from waterline.usgs import nims
from waterline.usgs.http import make_client

ENHANCE_OPS = {
    AgentAction.ENHANCE_CLAHE: "clahe",
    AgentAction.ENHANCE_GAMMA: "gamma",
    AgentAction.MASK_GLARE: "mask_glare",
}


class FrameNavigator:
    """Neighbors in the camera's archive. Wraps NIMS so tests can fake it."""

    def __init__(self, camera: nims.CameraInfo, cache: Path) -> None:
        self.camera = camera
        self.cache = cache

    def neighbor(self, when: datetime, direction: int, skip: int = 1) -> FrameRef | None:
        """The `skip`-th frame strictly after (direction>0) or before `when`."""
        client = make_client()
        if direction > 0:
            frames = nims.list_frames(client, self.camera, after=when, limit=skip + 2)
            later = sorted((f for f in frames if f.time > when), key=lambda f: f.time)
        else:
            frames = nims.list_frames(
                client, self.camera, before=when - timedelta(seconds=1), limit=skip + 2
            )
            later = sorted((f for f in frames if f.time < when), key=lambda f: f.time, reverse=True)
        return later[min(skip - 1, len(later) - 1)] if later else None

    def image(self, ref: FrameRef) -> np.ndarray:
        return nims.fetch_image(make_client(), ref, self.cache)


def enhance_image(action: AgentAction, image: np.ndarray) -> np.ndarray:
    return enhance.apply_op(image, ENHANCE_OPS[action])
