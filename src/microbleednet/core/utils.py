import numpy as np
import nibabel as nib
from pathlib import Path


def load_volume(path: Path) -> nib.Nifti1Image:
    return nib.load(path)


def save_volume(volume: nib.Nifti1Image, path: Path) -> None:
    nib.save(volume, path)


def nifti_to_numpy(volume: nib.Nifti1Image) -> np.ndarray:
    return volume.get_fdata()


def numpy_to_nifti(array: np.ndarray, reference: nib.Nifti1Image | None = None) -> nib.Nifti1Image:
    if reference is None:
        return nib.Nifti1Image(array, np.eye(4), nib.Nifti1Header())
    return nib.Nifti1Image(array, reference.affine, reference.header)


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
                
                patch = volume[start_x:end_x, start_y:end_y, start_z:end_z]
                patches.append(patch)
                
    return patches