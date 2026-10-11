"""The camera's own reading history, for temporal plausibility.

Camera values only: the gage is never stored here, by construction. The last
accepted reading bounds how fast a plausible water surface can move.
"""

import math
from dataclasses import dataclass
from datetime import datetime

from waterline.models import Reading

MAX_GAP_H = 3.0


@dataclass
class CameraHistory:
    """The last accepted camera reading. Updated after each accepted read."""

    stage_ft: float | None = None
    sigma_ft: float | None = None
    time: datetime | None = None

    def update(self, reading: Reading) -> None:
        if reading.status == "accepted" and reading.stage_ft is not None:
            self.stage_ft = reading.stage_ft
            self.sigma_ft = reading.sigma_ft
            self.time = reading.time

    def jump_beyond(
        self,
        stage_ft: float,
        sigma_ft: float | None,
        t: datetime,
        rate_max_ft_per_h: float | None,
    ) -> float | None:
        """|stage - prev| beyond rate*dt + 3 combined sigmas, else None.

        Returns the actual jump in feet when the new reading is implausibly far
        from the last accepted one (within MAX_GAP_H); None when plausible or
        there is nothing to compare against.
        """
        if rate_max_ft_per_h is None or self.stage_ft is None or self.time is None:
            return None
        dt_h = (t - self.time).total_seconds() / 3600
        if dt_h <= 0 or dt_h > MAX_GAP_H:
            return None
        s = math.sqrt((sigma_ft or 0.0) ** 2 + (self.sigma_ft or 0.0) ** 2)
        jump = abs(stage_ft - self.stage_ft)
        return jump if jump > rate_max_ft_per_h * dt_h + 3.0 * s else None
