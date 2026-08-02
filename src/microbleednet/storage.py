"""Atomic file writers shared across layers.

These primitives belong to the shared-contract layer (the package root, below
``core``/``pipelines``/``cli``) so that every layer — including root modules such
as :mod:`microbleednet.provenance` and :mod:`microbleednet.manifests` — can write
durable artifacts without reaching upward into ``core``.

Every durable JSON or text artifact is written by first serializing to a
temporary file in the target directory, flushing and syncing it, then atomically
replacing the target. A crash mid-write leaves the previous file intact rather
than a truncated one.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def write_json_atomic(path: Path, data: dict[str, Any] | list[Any]) -> None:
    """Serialize ``data`` to ``path`` as JSON, replacing it atomically.

    Accepts either a JSON object or a top-level array; both are valid JSON
    documents that callers need to persist.
    """
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


def write_text_atomic(path: Path, text: str) -> None:
    """Write ``text`` to ``path``, replacing it atomically.

    Used for reports (CSV, plots serialized as text) that advertise stage
    completion and must never be observed half-written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)
        temporary_file.write(text)
        temporary_file.flush()
        os.fsync(temporary_file.fileno())
    os.replace(temporary_path, path)
