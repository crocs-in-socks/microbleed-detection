"""Atomic JSON and text persistence for orchestration artifacts."""

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def write_json_atomic(path: Path, data: dict[str, Any] | list[Any]) -> None:
    """Serialize ``data`` to ``path`` as JSON, replacing it atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)
        json.dump(data, temporary_file, indent=2, sort_keys=True)
        temporary_file.write("\n")
        temporary_file.flush()
        os.fsync(temporary_file.fileno())
    os.replace(temporary_path, path)


def read_json(path: Path) -> Any:
    """Read and parse a JSON document from ``path``."""
    with path.open(encoding="utf-8") as json_file:
        return json.load(json_file)


