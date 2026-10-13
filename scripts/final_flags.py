"""Flags the frozen detector raised on the real (no-fault) final record.

Replays each site's final agent CSVs through the same FlagEngine the
injected-fault harness uses for its clean pass, one fresh engine per window
exactly like run_faults, and writes every flag to results/final_flags_{site}.json.
The counts must match the pooled fault reports' clean counts; a mismatch
stops the run, because it would mean the dashboard disagrees with the
published fault numbers.
"""

import json
import sys
from pathlib import Path

from waterline.faults import collect, load_rows
from waterline.settings import Settings

RESULTS = Path("results")


def main() -> int:
    rc = 0
    for site in ("hancock", "in_overland"):
        flags = []
        for csv_path in sorted(RESULTS.glob(f"final_eval_agent_{site}_*.csv")):
            label = csv_path.stem.removeprefix(f"final_eval_agent_{site}_")
            for _i, flag in collect(site, load_rows(csv_path), Settings()):
                entry = flag.model_dump(mode="json")
                entry["window"] = label
                flags.append(entry)
        out = RESULTS / f"final_flags_{site}.json"
        out.write_text(json.dumps(flags, indent=2))
        pooled = RESULTS / f"final_faults_{site}.json"
        expect = None
        if pooled.exists():
            expect = json.loads(pooled.read_text())["clean"]["flags"]
        status = "ok" if expect is None or len(flags) == expect else "MISMATCH"
        print(f"{site}: {len(flags)} flags (faults clean: {expect}) {status}")
        if status == "MISMATCH":
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
