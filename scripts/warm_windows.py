"""Warm the image cache for the frozen test windows. Download only.

No perception, no reading of image content. Resumable via results/tmp/.
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

from waterline import eval as ev
from waterline import sites
from waterline.usgs import nims
from waterline.usgs.http import UsgsError, make_client

CKPT = Path("results/tmp/warm_done.json")


def main() -> None:
    client = make_client()
    data = json.loads(Path("results/test_windows.json").read_text())
    exclude = frozenset(str(f) for f in data.get("exclude_files", []))
    done = set(json.loads(CKPT.read_text())) if CKPT.exists() else set()
    for site_id, windows in data.items():
        if not isinstance(windows, list):
            continue
        site = sites.load_site(site_id)
        for w in windows:
            start = datetime.fromisoformat(str(w["start"]).replace("Z", "+00:00"))
            end = datetime.fromisoformat(str(w["end"]).replace("Z", "+00:00"))
            frames = ev.frame_list(site, start, end, int(w["every"]), exclude)
            got = skipped = failed = 0
            for ref in frames:
                if ref.filename in done:
                    skipped += 1
                    continue
                for attempt in range(8):
                    try:
                        nims.fetch_image(client, ref, Path("data/cache"))
                        done.add(ref.filename)
                        got += 1
                        break
                    except UsgsError as err:
                        if "rate limited" in str(err) and attempt < 7:
                            time.sleep(65)
                        else:
                            print(f"FAIL {ref.filename}: {err}", flush=True)
                            failed += 1
                            break
                CKPT.write_text(json.dumps(sorted(done)))
            print(
                f"{site_id} {w['start']}..{w['end']}: {len(frames)} wanted, "
                f"{got} fetched, {skipped} cached, {failed} failed",
                flush=True,
            )


if __name__ == "__main__":
    main()
    sys.exit(0)
