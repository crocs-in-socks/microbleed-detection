import numpy as np
from skimage.measure import label
from skimage.measure import regionprops

from . import basic

def get_nonoverlapping_patches(volume: np.ndarray, patch_size: int) -> list:
    padding = [(0, (patch_size - s % patch_size) % patch_size) for s in volume.shape]

    volume = np.pad(volume, padding, mode='constant', constant_values=0)

    nx = volume.shape[0] // patch_size
    ny = volume.shape[1] // patch_size
    nz = volume.shape[2] // patch_size

    patches = []
    
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                start_x, end_x = x * patch_size, (x + 1) * patch_size
                start_y, end_y = y * patch_size, (y + 1) * patch_size
                start_z, end_z = z * patch_size, (z + 1) * patch_size

                bounding_box = (
                    (start_x, end_x),
                    (start_y, end_y),
                    (start_z, end_z)
                )
                
                patch = basic.apply_bounding_box(volume, bounding_box)
                patches.append(patch)
                
    return patches


def get_target_centered_patches(volume: np.ndarray, target: np.ndarray, patch_size: int) -> list:
    padding = [(patch_size // 2, patch_size // 2) for s in volume.shape]

    volume = np.pad(volume, padding, mode='constant', constant_values=0)
    target = np.pad(target, padding, mode='constant', constant_values=0)

    target = label(target)
    dist_props = regionprops(target)
    n_patches = len(dist_props)

    height, width, depth = target.shape
    patches = []

    for patch_idx in range(n_patches):
        centroid = dist_props[patch_idx].centroid
        start_x, end_x = max(0, int(np.round(centroid[0])) - patch_size // 2), min(height, int(np.round(centroid[0])) + patch_size // 2)
        start_y, end_y = max(0, int(np.round(centroid[1])) - patch_size // 2), min(width, int(np.round(centroid[1])) + patch_size // 2)
        start_z, end_z = max(0, int(np.round(centroid[2])) - patch_size // 2), min(depth, int(np.round(centroid[2])) + patch_size // 2)

        bounding_box = (
            (start_x, end_x),
            (start_y, end_y),
            (start_z, end_z)
        )

        patch = basic.apply_bounding_box(volume, bounding_box)
        patches.append(patch)
    
    return patches