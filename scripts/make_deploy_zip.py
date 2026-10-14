"""Build dist/waterline-deploy.zip for the AWS CloudShell deploy.

The zip carries exactly what `docker build -f deploy/Dockerfile .` needs plus
deploy/deploy.sh: src/, sites/ (reference images included, no frame cache),
pyproject.toml, uv.lock, README.md, .dockerignore and deploy/. The recipient
uploads it in CloudShell, then runs:

    unzip -o waterline-deploy.zip -d waterline && cd waterline
    && bash deploy/deploy.sh

Run from the repo root: .venv/Scripts/python.exe scripts/make_deploy_zip.py
"""

import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dist" / "waterline-deploy.zip"

TOP_LEVEL_FILES = [
    "pyproject.toml",
    "uv.lock",
    "README.md",
    ".dockerignore",
]
TOP_LEVEL_DIRS = ["src", "sites", "deploy"]

SKIP_SUFFIXES = {".pyc", ".pyo"}
SKIP_DIRS = {"__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache"}


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for name in TOP_LEVEL_FILES:
            p = ROOT / name
            if not p.exists():
                print(f"missing required file {name}")
                return 1
            z.write(p, name)
            n += 1
        for top in TOP_LEVEL_DIRS:
            for p in sorted((ROOT / top).rglob("*")):
                if not p.is_file():
                    continue
                if p.suffix in SKIP_SUFFIXES or any(part in SKIP_DIRS for part in p.parts):
                    continue
                z.write(p, p.relative_to(ROOT).as_posix())
                n += 1
    size_mb = OUT.stat().st_size / 1e6
    print(f"{OUT}  {size_mb:.1f} MB  {n} files")
    if size_mb > 200:
        print("warning: zip is larger than expected; check what crept in")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
