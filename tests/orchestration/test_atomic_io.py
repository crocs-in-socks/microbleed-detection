import json
from pathlib import Path

import pytest

from microbleednet.orchestration.atomic_io import read_json, write_json_atomic


def test_write_json_atomic_creates_parent_and_stable_json(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "data.json"

    write_json_atomic(path, {"zebra": 1, "apple": "microbleed"})

    content = path.read_text(encoding="utf-8")
    assert content == '{\n  "apple": "microbleed",\n  "zebra": 1\n}\n'


def test_write_json_atomic_replaces_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "data.json"
    path.write_text("old content", encoding="utf-8")

    write_json_atomic(path, ["new", "content"])

    assert read_json(path) == ["new", "content"]
    assert list(tmp_path.iterdir()) == [path]


def test_read_json_rejects_malformed_document(tmp_path: Path) -> None:
    path = tmp_path / "data.json"
    path.write_text("{not valid json}", encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        read_json(path)
