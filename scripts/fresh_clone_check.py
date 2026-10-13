"""Fresh-clone smoke check: clone, sync, gate, dashboard, replay-compare.

Runs outside the repo. Steps:
1. clone origin/m2-perception into a temp dir
2. uv sync, then scripts/gate.py (ruff, format, mypy, pytest)
3. start the dashboard on a spare port, hit /api/sites and
   /api/final_summary?site=hancock&window=all, compare the summary values to
   the committed results/final_summary_hancock.json
4. trace-replay check on the first hancock window. The committed traces cover
   every final frame, so final_traces.py takes its compare-only path and
   re-validates each committed trace against the final CSVs (status, stage to
   1e-9, source frame) with no camera work and no network. Re-rendering traces
   from scratch needs data/cache/ (not committed) or NIMS access; on a clean
   clone the replay therefore proves the committed artifacts consistent,
   which is what the compare-only path does.

Run from anywhere: uv run python scripts/fresh_clone_check.py
"""

import json
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

REPO_URL = "https://github.com/tanayvasishtha/waterline.git"
BRANCH = "m2-perception"
PORT = 8791


def _sh(
    cmd: list[str], cwd: Path | None = None, timeout: int = 900
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, check=False
    )


def _port_free(port: int) -> bool:
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def _kill_tree(pid: int) -> None:
    """Terminate the spawned process and its children (uv -> waterline web)."""
    subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True, check=False)


def _dashboard_smoke(tmp: Path) -> bool:
    if not _port_free(PORT):
        print(f"port {PORT} in use; pick another")
        return False
    srv = subprocess.Popen(
        ["uv", "run", "waterline", "web", "--port", str(PORT)],
        cwd=tmp,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        base = f"http://localhost:{PORT}"
        for _ in range(30):
            try:
                urllib.request.urlopen(base + "/api/sites", timeout=2)
                break
            except Exception:
                time.sleep(1)
        sites = json.loads(urllib.request.urlopen(base + "/api/sites", timeout=5).read())
        print("sites:", sites)
        got = json.loads(
            urllib.request.urlopen(
                base + "/api/final_summary?site=hancock&window=all", timeout=5
            ).read()
        )
        want = json.loads((tmp / "results/final_summary_hancock.json").read_text())
        wa, ga = want["all"]["all"]["agent"], got["agent"]
        checks = [
            ("agent.mae_ft", wa["mae_ft"], ga["mae_ft"]),
            ("agent.accepted", wa["accepted"], ga["accepted"]),
            ("agent.pending", wa["pending"], ga["pending"]),
        ]
        ok = True
        for name, w, g in checks:
            match = w == g
            ok &= match
            print(f"  {name}: committed={w} api={g} {'OK' if match else 'MISMATCH'}")
        return ok
    finally:
        _kill_tree(srv.pid)


def _replay_compare(tmp: Path) -> bool:
    r = _sh(
        [
            "uv",
            "run",
            "python",
            "scripts/final_traces.py",
            "--worker",
            "hancock",
            "2026-04-14T00:00Z",
            "2026-05-28T00:00Z",
            "5",
        ],
        cwd=tmp,
        timeout=600,
    )
    out = (r.stdout + r.stderr).strip()
    print("\n".join(out.split("\n")[-6:]))
    return "RESULT 0" in out


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="waterline_clone_"))
    try:
        return _main(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _main(tmp: Path) -> int:
    print(f"CLONE -> {tmp}")
    r = _sh(["git", "clone", "--depth", "50", "-b", BRANCH, REPO_URL, str(tmp)])
    if r.returncode != 0:
        print(r.stderr[-2000:])
        return 1

    print("SYNC")
    r = _sh(["uv", "sync"], cwd=tmp)
    if r.returncode != 0:
        print(r.stderr[-2000:])
        return 1

    print("GATE")
    r = _sh(["uv", "run", "python", "scripts/gate.py"], cwd=tmp, timeout=1800)
    tail = (r.stdout + r.stderr).strip().split("\n")[-8:]
    print("\n".join(tail))
    if r.returncode != 0:
        return 1

    print("DASHBOARD SMOKE")
    if not _dashboard_smoke(tmp):
        return 1

    print("TRACE REPLAY (compare-only: committed traces vs committed CSVs)")
    if not _replay_compare(tmp):
        print("replay compare did not report RESULT 0")
        return 1

    print("FRESH-CLONE CHECK PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
