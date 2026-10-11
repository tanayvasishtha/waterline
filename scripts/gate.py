"""Pre-commit gate: lint, types, tests, prose and secret scans. Exits non-zero on any failure.

Usage: uv run python scripts/gate.py
"""

import re
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CHECKS = [
    ["ruff", "check", "."],
    ["ruff", "format", "--check", "."],
    ["mypy"],
    ["pytest", "-q"],
]

DASHES = re.compile("[\u2013\u2014]")
PLACEHOLDERS = re.compile(r"TODO|FIXME|lorem|example\.com", re.IGNORECASE)
SECRETS = re.compile(
    r"AKIA[0-9A-Z]{16}|aws_secret_access_key\s*=|-----BEGIN [A-Z ]*PRIVATE KEY-----"
)


def tracked_files() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout
    return [ROOT / line for line in out.splitlines() if line]


def scan(paths: Iterable[Path], pattern: re.Pattern[str], label: str) -> list[str]:
    hits = []
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, FileNotFoundError):
            continue
        for number, line in enumerate(text.splitlines(), 1):
            if pattern.search(line):
                hits.append(f"{label}: {path.relative_to(ROOT)}:{number}")
    return hits


def main() -> int:
    for command in CHECKS:
        if subprocess.run(["uv", "run", *command], cwd=ROOT, check=False).returncode != 0:
            print(f"gate failed: {' '.join(command)}")
            return 1

    files = tracked_files()
    prose_and_code = [p for p in files if p.suffix in {".md", ".py", ".html", ".js", ".css"}]
    source = [p for p in files if p.is_relative_to(ROOT / "src")]
    gate_file = Path(__file__).resolve()
    hits = (
        scan(prose_and_code, DASHES, "em or en dash")
        + scan(source, PLACEHOLDERS, "placeholder")
        + scan([p for p in files if p != gate_file], SECRETS, "possible secret")
    )
    for hit in hits:
        print(hit)
    if hits:
        print(f"gate failed: {len(hits)} scan hit(s)")
        return 1
    print("gate passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
