from pathlib import Path
import hashlib
import json
import os
import tempfile

import numpy as np

from microbleednet.core.transforms import patch

# No augmentation multiplier unless a caller asks for one.
_DEFAULT_AUGMENTATION_FACTOR = 1

def nonoverlapping_patcher(
    volume: np.ndarray,
    mask: np.ndarray,
    patch_size: int
) -> list:
    volume_patches = patch.get_nonoverlapping_patches(volume, patch_size)
    mask_patches = patch.get_nonoverlapping_patches(mask, patch_size)

    return [
        {
            "volume": volume_patch,
            "mask": mask_patch,
        }
        for volume_patch, mask_patch in zip(volume_patches, mask_patches)
    ]

def target_centered_patcher(
        volume: np.ndarray,
        mask: np.ndarray,
        target: np.ndarray,
        patch_size: int
) -> list:
    volume_records = patch.get_target_centered_patch_records(volume, target, patch_size)
    mask_records = patch.get_target_centered_patch_records(mask, target, patch_size)

    return [
        {
            "volume": volume_record[0],
            "mask": mask_record[0],
            "patch_bounds": volume_record[1],
            "candidate_id": volume_record[2],
        }
        for volume_record, mask_record in zip(volume_records, mask_records)
    ]

def materialize_patches(
    patches: list,
    patch_dir: Path,
    volume_identifier: str,
    augmentation_factor: int = _DEFAULT_AUGMENTATION_FACTOR,
):
    patch_dir.mkdir(parents=True, exist_ok=True)

    patch_metadata = []
    manifest = []

    for idx, patch_data in enumerate(patches):
        patch_path = patch_dir / f"patch_{volume_identifier}_{idx:06d}.npz"
        arrays = {key: value for key, value in patch_data.items() if isinstance(value, np.ndarray)}
        with tempfile.NamedTemporaryFile(dir=patch_dir, suffix=".npz", delete=False) as temporary_file:
            temporary_path = Path(temporary_file.name)
        try:
            np.savez_compressed(temporary_path, **arrays)
            os.replace(temporary_path, patch_path)
        finally:
            temporary_path.unlink(missing_ok=True)

        checksum = hashlib.sha256(patch_path.read_bytes()).hexdigest()
        has_microbleed = bool(np.any(patch_data["mask"] > 0))
        record = {
            "patch_path": str(patch_path.resolve()),
            "source_subject": volume_identifier,
            "patch_bounds": patch_data.get("patch_bounds"),
            "candidate_id": patch_data.get("candidate_id", idx),
            "candidate_probability": patch_data.get("candidate_probability"),
            "label": int(has_microbleed),
            "has_microbleed": has_microbleed,
            "augmentation_version": 0,
            "augmentation_factor": int(augmentation_factor),
            "checksum": checksum,
        }
        patch_metadata.append(record)
        manifest.append(record)

    manifest_path = patch_dir / f"manifest_{volume_identifier}.json"
    with tempfile.NamedTemporaryFile("w", dir=patch_dir, suffix=".json", delete=False) as temporary_file:
        json.dump(manifest, temporary_file, indent=2)
        temporary_manifest = Path(temporary_file.name)
    os.replace(temporary_manifest, manifest_path)

    return patch_metadata