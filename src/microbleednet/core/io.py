"""NIfTI, array, and checkpoint I/O used by core operations."""

from pathlib import Path
from typing import cast

import nibabel as nib
import numpy as np


def load_volume(path: Path | str) -> nib.Nifti1Image:
    return cast(nib.Nifti1Image, nib.load(path))


def save_volume(volume: nib.Nifti1Image, path: Path) -> None:
    nib.save(volume, path)


def nifti_to_numpy(volume: nib.Nifti1Image) -> np.ndarray:
    return volume.get_fdata()


def numpy_to_nifti(
    array: np.ndarray, reference: nib.Nifti1Image
) -> nib.Nifti1Image:
    return nib.Nifti1Image(array, reference.affine, reference.header)