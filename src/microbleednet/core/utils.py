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

