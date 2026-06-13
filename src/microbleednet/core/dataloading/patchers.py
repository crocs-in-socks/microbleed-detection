import numpy as np

from microbleednet.core.transforms import basic
from microbleednet.core.transforms import patch

def nonoverlapping_patcher(
    volume: np.ndarray,
    mask: np.ndarray,
    patch_size: int
) -> list:
    volume_patches = patch.get_nonoverlapping_patches(volume, patch_size)
    mask_patches = patch.get_nonoverlapping_patches(volume, patch_size)

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
    patch_size: int
) -> list:
    volume_patches = patch.get_target_centered_patches(volume, mask, patch_size)
    mask_patches = patch.get_target_centered_patches(mask, mask, patch_size)

    return [
        {
            "volume": volume_patch,
            "mask": mask_patch
        }
        for volume_patch, mask_patch in zip(volume_patches, mask_patches)
    ]