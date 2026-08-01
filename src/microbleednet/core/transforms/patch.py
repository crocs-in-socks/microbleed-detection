import numpy as np
from skimage.measure import label
from skimage.measure import regionprops

from . import basic


def _validate_patch_size(patch_size: int) -> None:
    if not isinstance(patch_size, int) or patch_size <= 0:
        raise ValueError("patch_size must be a positive integer")


def _extract_fixed_patch(
    volume: np.ndarray,
    starts: tuple[int, int, int],
    patch_size: int,
) -> tuple[np.ndarray, tuple[tuple[int, int], tuple[int, int], tuple[int, int]]]:
    bounds = tuple((start, start + patch_size) for start in starts)
    result = np.zeros((patch_size, patch_size, patch_size), dtype=volume.dtype)
    source = tuple(slice(max(0, start), min(limit, size)) for (start, limit), size in zip(bounds, volume.shape))
    target = tuple(slice(max(0, -start), max(0, -start) + current.stop - current.start) for start, current in zip(starts, source))
    result[target] = volume[source]
    return result, bounds

def get_nonoverlapping_patches(volume: np.ndarray, patch_size: int) -> list:
    _validate_patch_size(patch_size)
    if volume.ndim != 3:
        raise ValueError("volume must be a 3D array")
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
    return [item[0] for item in get_target_centered_patch_records(volume, target, patch_size)]


def get_target_centered_patch_records(
    volume: np.ndarray, target: np.ndarray, patch_size: int
) -> list[tuple[np.ndarray, tuple[tuple[int, int], tuple[int, int], tuple[int, int]], int]]:
    _validate_patch_size(patch_size)
    if volume.ndim != 3 or target.shape != volume.shape:
        raise ValueError("volume and target must be matching 3D arrays")
    labeled_target = label(target > 0, connectivity=3)
    records = []
    for candidate_id, properties in enumerate(regionprops(labeled_target), start=1):
        center = tuple(int(np.round(value)) for value in properties.centroid)
        starts = tuple(center_axis - patch_size // 2 for center_axis in center)
        patch_data, bounds = _extract_fixed_patch(volume, starts, patch_size)
        records.append((patch_data, bounds, candidate_id))
    return records