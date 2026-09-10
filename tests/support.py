"""Explicitly imported builders shared by repository tests."""

import json
from pathlib import Path

from microbleednet.orchestration.configs import IndexDataConfig


def make_index_config(tmp_path: Path, **overrides: object) -> IndexDataConfig:
    """Build a valid index-data config with concise per-test overrides."""
    (tmp_path / "masks").mkdir(exist_ok=True)
    values: dict[str, object] = {
        "dataset_dir": tmp_path / "dataset",
        "input_dir": tmp_path,
        "volume_pattern": "{subject_id}_volume.nii.gz",
        "label_dir": tmp_path / "masks",
        "mask_pattern": "{subject_id}_mask.nii.gz",
        "source_id": "test-source",
    }
    values.update(overrides)
    return IndexDataConfig.model_validate(values)


def write_index_config(path: Path, **overrides: object) -> Path:
    """Write a flat index-data TOML config and return its path."""
    values: dict[str, object] = {
        "dataset_dir": path.parent / "dataset",
        "input_dir": path.parent,
        "volume_pattern": "{subject_id}_volume.nii.gz",
        "label_dir": path.parent / "masks",
        "mask_pattern": "{subject_id}_mask.nii.gz",
        "source_id": "test-source",
    }
    values.update(overrides)

    lines = []
    for key, value in values.items():
        if value is None:
            continue
        if isinstance(value, bool):
            literal = "true" if value else "false"
        elif isinstance(value, (str, Path)):
            literal = json.dumps(str(value))
        else:
            raise TypeError(f"unsupported config value: {value!r}")
        lines.append(f"{key} = {literal}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def create_volume_source(tmp_path: Path, name: str, subject_ids: list[str]) -> Path:
    """Create a source directory and matching empty mask files."""
    source_dir = tmp_path / name
    mask_dir = tmp_path / f"{name}_masks"
    source_dir.mkdir()
    mask_dir.mkdir(exist_ok=True)
    for subject_id in subject_ids:
        (source_dir / f"{subject_id}_volume.nii.gz").write_bytes(b"")
        (mask_dir / f"{subject_id}_mask.nii.gz").write_bytes(b"")
    return source_dir
