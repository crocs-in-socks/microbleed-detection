import os
import subprocess
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import SimpleITK as sitk

from .. import io


def normalize_volume(volume: np.ndarray) -> np.ndarray:
    maximum = np.max(volume)
    if not np.isfinite(maximum) or maximum <= 0:
        raise ValueError("volume is empty or its maximum is not positive and finite")
    return volume / maximum


def invert_volume(volume: np.ndarray) -> np.ndarray:
    brain_mask = (volume > 0).astype(int)
    volume = np.max(volume) - volume
    volume = volume * brain_mask
    return volume


def tight_crop_volume(
    volume: np.ndarray,
) -> tuple[np.ndarray, tuple[tuple[int, int], tuple[int, int], tuple[int, int]]]:
    if volume.ndim != 3:
        raise ValueError("volume must be a 3D array")
    dim0_sum = np.sum(volume, axis=(1, 2))
    dim1_sum = np.sum(volume, axis=(0, 2))
    dim2_sum = np.sum(volume, axis=(0, 1))

    d0_start, d0_end = find_bounds_1d(dim0_sum)
    d1_start, d1_end = find_bounds_1d(dim1_sum)
    d2_start, d2_end = find_bounds_1d(dim2_sum)

    if not np.any(volume > 0):
        raise ValueError("cannot crop an empty volume")

    bounding_box: tuple[tuple[int, int], tuple[int, int], tuple[int, int]] = (
        (int(d0_start), int(d0_end + 1)),
        (int(d1_start), int(d1_end + 1)),
        (int(d2_start), int(d2_end + 1)),
    )

    cropped_volume = apply_bounding_box(volume, bounding_box)

    return cropped_volume, bounding_box


def apply_bounding_box(volume: np.ndarray, bounding_box: tuple) -> np.ndarray:
    (d0_start, d0_end), (d1_start, d1_end), (d2_start, d2_end) = bounding_box

    cropped_volume = volume[d0_start:d0_end, d1_start:d1_end, d2_start:d2_end]

    return cropped_volume


def find_bounds_1d(array: np.ndarray) -> tuple:
    nonzero_indices = np.flatnonzero(array > 0)
    if nonzero_indices.size == 0:
        return 0, 0

    first_index = nonzero_indices[0]
    last_index = nonzero_indices[-1]

    return first_index, last_index


def reorient_to_std(volume: nib.Nifti1Image) -> nib.Nifti1Image:
    return nib.as_closest_canonical(volume)


def crop_affine(
    affine: np.ndarray,
    crop_start: tuple[int, int, int],
) -> np.ndarray:
    translation = np.eye(4)
    translation[:3, 3] = crop_start
    return affine @ translation


def extract_brain(volume: nib.Nifti1Image) -> nib.Nifti1Image:
    fsldir = Path(os.getenv("FSLDIR", ""))
    if not fsldir.is_dir():
        raise EnvironmentError(
            "Valid FSLDIR environment variable is not set. "
            "Set it using 'export FSLDIR=/path/to/fsl'."
        )

    with tempfile.TemporaryDirectory(prefix="microbleednet-fsl-bet-") as temp_dir:
        temp_dir = Path(temp_dir)
        input_path = temp_dir / "pre_bet.nii.gz"
        output_path = temp_dir / "post_bet.nii.gz"

        # Save to disk just for BET
        io.save_volume(volume, input_path)
        subprocess.run(
            [str(fsldir / "bin" / "bet"), str(input_path), str(output_path)], check=True
        )

        # Some potentially over-defensive programming:
        # Load back into memory immediately and let tempdir delete the files
        output_volume = io.load_volume(output_path)
        # Force load data into memory so we don't rely on the deleted temp file
        output_data = io.nifti_to_numpy(output_volume)
        loaded_volume = io.numpy_to_nifti(output_data, output_volume)

        return loaded_volume


def bias_field_correct_n4(volume: nib.Nifti1Image) -> nib.Nifti1Image:
    volume_data = io.nifti_to_numpy(volume).astype(float)

    # Transpose for SimpleITK coordinate system
    sitk_volume = sitk.GetImageFromArray(volume_data.T)

    # Create a mask of the brain tissue to ignore empty space
    mask_image = sitk_volume > 0

    # Run highly optimized N4 correction
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrected_sitk = corrector.Execute(sitk_volume, mask_image)

    # Convert back to NumPy and nibabel
    corrected_data = sitk.GetArrayFromImage(corrected_sitk).T
    corrected_nifti = io.numpy_to_nifti(corrected_data, volume)

    return corrected_nifti
