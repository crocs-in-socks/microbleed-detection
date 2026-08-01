import glob
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from natsort import natsorted

from ..core import utils
from . import constants


def execute(
    input_dir: Path,
    label_dir: Optional[Path],
    dataset_dir: Path,
    volume_pattern: str,
    mask_pattern: Optional[str],
    require_masks: bool = True,
) -> None:
    validate_pattern(volume_pattern)
    if mask_pattern is not None:
        validate_pattern(mask_pattern)
    if label_dir is None and require_masks:
        raise ValueError("a label directory is required for this manifest")
    if label_dir is not None and mask_pattern is None:
        raise ValueError("mask_pattern is required when label_dir is provided")

    dataset_dir.mkdir(parents=True, exist_ok=True)
    volume_paths = compute_paths(input_dir, volume_pattern)
    mask_paths = (
        compute_paths(label_dir, mask_pattern)
        if label_dir is not None and mask_pattern is not None
        else []
    )

    volume_subject_map = _build_subject_map(input_dir, volume_paths, volume_pattern)
    mask_subject_map = (
        _build_subject_map(label_dir, mask_paths, mask_pattern)
        if label_dir is not None and mask_pattern is not None
        else {}
    )

    volume_ids = set(volume_subject_map)
    mask_ids = set(mask_subject_map)
    unmatched_volumes = sorted(volume_ids - mask_ids)
    unmatched_masks = sorted(mask_ids - volume_ids)
    if require_masks and (unmatched_volumes or unmatched_masks):
        raise ValueError(
            "unmatched subjects: "
            f"volumes={unmatched_volumes}, masks={unmatched_masks}"
        )

    subjects = []
    for subject_id in natsorted(volume_subject_map):
        mask_path = mask_subject_map.get(subject_id)
        subjects.append(
            {
                "subject_id": subject_id,
                "volume_path": str(volume_subject_map[subject_id].resolve()),
                "mask_path": str(mask_path.resolve()) if mask_path else None,
            }
        )

    raw_manifest_data = {
        "stage": "raw",
        "created_on": datetime.now().isoformat(),
        "sources": [
            {
                "input_dir": str(input_dir.resolve()),
                "label_dir": str(label_dir.resolve()) if label_dir else None,
                "volume_pattern": volume_pattern,
                "mask_pattern": mask_pattern,
                "added_on": datetime.now().isoformat(),
            }
        ],
        "subjects": subjects,
        "unmatched_volumes": unmatched_volumes,
        "unmatched_masks": unmatched_masks,
    }

    utils.write_json_atomic(dataset_dir / constants.manifests.raw, raw_manifest_data)


def validate_pattern(pattern: str) -> None:
    placeholder = constants.index_data.subject_id_placeholder
    if pattern.count(placeholder) != 1:
        raise ValueError(
            "pattern must contain exactly one '{subject_id}' placeholder"
        )


def _build_subject_map(
    root_dir: Path,
    paths: list[Path],
    pattern: str,
) -> dict[str, Path]:
    subject_map: dict[str, Path] = {}
    for path in paths:
        subject_id = extract_subject_id(root_dir, path, pattern)
        if subject_id is None or not subject_id.strip():
            raise ValueError(f"path does not match pattern or has an empty ID: {path}")
        if subject_id in subject_map:
            raise ValueError(f"duplicate subject ID '{subject_id}' in {root_dir}")
        subject_map[subject_id] = path
    return subject_map


def compute_paths(dir: Path, pattern: str) -> list[Path]:
    validate_pattern(pattern)
    pattern_parts = pattern.split(constants.index_data.subject_id_placeholder)
    glob_pattern = "*".join(glob.escape(part) for part in pattern_parts)
    return natsorted(dir.rglob(glob_pattern))


def remove_overlap(
    paths_a: list[Path],
    paths_b: list[Path]
) -> tuple[list[Path], list[Path]]:
    overlap = set(paths_a) & set(paths_b)
    return (
        natsorted(set(paths_a) - overlap),
        natsorted(set(paths_b) - overlap),
    )


def extract_subject_id(root_dir: Path, path: Path, pattern: str) -> Optional[str]:
    validate_pattern(pattern)
    clean_path = path.relative_to(root_dir)

    pattern_parts = pattern.split(constants.index_data.subject_id_placeholder)
    escaped_parts = [re.escape(part) for part in pattern_parts]
    regex_pattern = "^" + "(.*?)".join(escaped_parts) + "$"
    match = re.match(regex_pattern, str(clean_path))
    return match.group(1) if match else None
