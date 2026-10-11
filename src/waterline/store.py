"""Local JSON persistence. Same interface the DynamoDB backend will implement in M4."""

import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class JsonlStore:
    """Append-only JSON-lines store for readings, decisions and events."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, item: BaseModel) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(item.model_dump_json() + "\n")

    def read(self, model: type[T]) -> list[T]:
        if not self.path.exists():
            return []
        return [
            model.model_validate_json(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]


def write_json(path: Path, item: BaseModel) -> None:
    """Atomically write a single record (events, traces, manifests)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, delete=False, encoding="utf-8", suffix=".tmp"
    ) as fh:
        fh.write(item.model_dump_json(indent=2))
        tmp = Path(fh.name)
    tmp.replace(path)


def read_json(path: Path, model: type[T]) -> T:
    return model.model_validate_json(path.read_text(encoding="utf-8"))


def utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def event_dir(base: Path, site_id: str, when: datetime) -> Path:
    return base / site_id / when.strftime("%Y%m%dT%H%MZ")
