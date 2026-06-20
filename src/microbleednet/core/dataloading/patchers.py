from pathlib import Path

import numpy as np

from .. import constants

from microbleednet.core.transforms import basic
from microbleednet.core.transforms import patch

def nonoverlapping_patcher(
    volume: np.ndarray,
    mask: np.ndarray,
    patch_size: int
) -> list:
    volume_patches = patch.get_nonoverlapping_patches(volume, patch_size)
    mask_patches = patch.get_nonoverlapping_patches(mask, patch_size)

    voxel_weights = basic.calculate_voxel_weights(mask)
    voxel_weights_patches = patch.get_nonoverlapping_patches(voxel_weights, patch_size)

    return [
        {
            "volume": volume_patch,
            "mask": mask_patch,
            "voxel_weights": voxel_weights_patch
        }
        for volume_patch, mask_patch, voxel_weights_patch in zip(volume_patches, mask_patches, voxel_weights_patches)
    ]

def target_centered_patcher(
        volume: np.ndarray,
        mask: np.ndarray,
        target: np.ndarray,
        patch_size: int
) -> list:
    volume_patches = patch.get_target_centered_patches(volume, target, patch_size)
    mask_patches = patch.get_target_centered_patches(mask, target, patch_size)

    return [
        {
            "volume": volume_patch,
            "mask": mask_patch
        }
        for volume_patch, mask_patch in zip(volume_patches, mask_patches)
    ]

def materialize_patches(
    patches: list,
    patch_dir: Path,
    volume_identifier: str,
    augmentation_factor: int = constants.dataloading.patchers.default.augmentation_factor,
):
    patch_dir.mkdir(parents=True, exist_ok=True)

    patch_metadata = []

    for idx, patch_data in enumerate(patches):
        patch_path = patch_dir / f"patch_{volume_identifier}_{idx:06d}.npz"
        np.savez_compressed(patch_path, **patch_data)

        has_microbleed = np.sum(patch_data['mask']) > 0

        patch_metadata.extend(
            {
                "patch_path": str(patch_path.resolve()),
                "has_microbleed": has_microbleed,
                "is_augmented": version != 0
            }
            for version in range(augmentation_factor)
        )

    return patch_metadata