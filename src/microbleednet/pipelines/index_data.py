from pathlib import Path
from typing import Optional

import re
import json
import glob
from datetime import datetime
from natsort import natsorted

from . import constants


def execute(
    input_dir: Path,
    label_dir: Optional[Path],
    dataset_dir: Path,
    volume_pattern: str,
    mask_pattern: Optional[str]
) -> None:

    if label_dir is None:
        # Explicity set to None
        mask_pattern = None

    volume_paths = compute_paths(input_dir, volume_pattern)
    mask_paths = compute_paths(label_dir, mask_pattern) if mask_pattern is not None else list()

    if input_dir == label_dir:
        volume_paths, mask_paths = remove_overlap(volume_paths, mask_paths)

    volume_subject_map = {
        extract_subject_id(input_dir, path, volume_pattern): path
        for path in volume_paths
    }

    mask_subject_map = {}
    if len(mask_paths) > 0:
        mask_subject_map = {
            extract_subject_id(label_dir, path, mask_pattern): path
            for path in mask_paths
        }

    raw_manifest_path = dataset_dir / constants.manifests.raw
    raw_manifest_data = {"stage": "raw", "sources": [], "subjects": {}}
    if raw_manifest_path.exists():
        with open(raw_manifest_path, mode="r") as f:
            existing_data = json.load(f)
            raw_manifest_data["stage"] = existing_data.get("stage", "raw")
            raw_manifest_data["sources"] = existing_data.get("sources", [])
            # Map existing subjects by ID for deduplication
            raw_manifest_data["subjects"] = {s["subject_id"]: s for s in existing_data.get("subjects", [])}

    raw_manifest_data["sources"].append({
        "input_dir": str(input_dir.resolve()),
        "label_dir": str(label_dir.resolve()) if label_dir else None,
        "volume_pattern": volume_pattern,
        "mask_pattern": mask_pattern,
        "added_on": datetime.now().isoformat(),
    })

    for subject_id, volume_path in volume_subject_map.items():
        mask_path = mask_subject_map.get(subject_id)
        raw_manifest_data["subjects"][subject_id] = {
            "subject_id": subject_id,
            "volume_path": str(volume_path.resolve()),
            "mask_path": str(mask_path.resolve()) if mask_path else None
        }

    # Convert subjects back to a list
    raw_manifest_data["subjects"] = list(raw_manifest_data["subjects"].values())

    with open(raw_manifest_path, mode="w") as raw_manifest_file:
        json.dump(raw_manifest_data, raw_manifest_file)


def compute_paths(dir: Path, pattern: str) -> list[Path]:
    pattern_parts = pattern.split(constants.index_data.subject_id_placeholder)
    glob_pattern = "*".join(glob.escape(part) for part in pattern_parts)
    return natsorted(dir.rglob(glob_pattern))

def remove_overlap(
    paths_A: list[Path],
    paths_B: list[Path]
) -> tuple[list[Path], list[Path]]:
    paths_A = set(paths_A)
    paths_B = set(paths_B)

    if paths_A > paths_B:
        paths_A = paths_A - paths_B
    elif paths_B > paths_A:
        paths_B = paths_B - paths_A

    return natsorted(paths_A), natsorted(paths_B)


def extract_subject_id(root_dir: Path, path: Path, pattern: str) -> Optional[str]:
    clean_path = path.relative_to(root_dir)
    while clean_path.suffix:
        clean_path = clean_path.with_suffix("")

    clean_path_str = str(clean_path)

    pattern_parts = pattern.split(constants.index_data.subject_id_placeholder)
    escaped_parts = [re.escape(p) for p in pattern_parts]
    regex_pattern = "^" + "(.*?)".join(escaped_parts) + "$"
    match = re.match(regex_pattern, clean_path_str)

    if match:
        # group(1) returns the text caught by the first (.*?) placeholder
        return match.group(1)
    return None
