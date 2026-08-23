import numpy as np

from . import volume_ops


def extract_centered_patches(
    volume: np.ndarray,
    centers: list[tuple[int, int, int]],
    patch_size: int,
) -> list[np.ndarray]:
    """Extract one ``patch_size``\\ ^3 patch centered on each voxel in ``centers``.

    The volume is zero-padded by ``patch_size // 2`` on every side once, so a
    center near the edge still yields a full-size patch (the out-of-bounds region
    reads as zeros). In the padded array the patch centered on ``center`` (given
    in source coordinates) starts at ``center`` and spans ``patch_size``, so each
    extraction is a plain slice with no per-patch bounds clamping. This is the
    shared entry point for the target-centered training path and inference, so
    the centering convention lives in one place. ``patch_size`` is a positive
    integer supplied by validated pipeline configuration.
    """
    half = patch_size // 2
    padded = np.pad(volume, half, mode="constant", constant_values=0)
    patches = []
    for center in centers:
        region = tuple(slice(axis, axis + patch_size) for axis in center)
        patches.append(padded[region])
    return patches


def get_nonoverlapping_patches(volume: np.ndarray, patch_size: int) -> list:
    """Extract nonoverlapping cubic patches using validated ``patch_size``."""
    if volume.ndim != 3:
        raise ValueError("volume must be a 3D array")
    padding = [(0, (patch_size - s % patch_size) % patch_size) for s in volume.shape]

    volume = np.pad(volume, padding, mode="constant", constant_values=0)

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

                bounding_box = ((start_x, end_x), (start_y, end_y), (start_z, end_z))

                patch = volume_ops.apply_bounding_box(volume, bounding_box)
                patches.append(patch)

    return patches


