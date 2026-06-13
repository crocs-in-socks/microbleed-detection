import os
import tempfile
import subprocess
import numpy as np
import nibabel as nib
from pathlib import Path
import SimpleITK as sitk

from scipy.ndimage import gaussian_filter

from .. import utils


def normalize_volume(volume: np.ndarray) -> np.ndarray:
    volume = volume - np.min(volume)
    if np.max(volume) > 0:
        volume = volume / np.max(volume)
    return volume


def invert_volume(volume: np.ndarray) -> np.ndarray:
    brain_mask = (volume > 0).astype(int)
    volume = np.max(volume) - volume
    volume = volume * brain_mask
    return volume


def tight_crop_volume(volume: np.ndarray) -> tuple[np.ndarray, tuple]:
    dim0_sum = np.sum(volume, axis=(1, 2))
    dim1_sum = np.sum(volume, axis=(0, 2))
    dim2_sum = np.sum(volume, axis=(0, 1))

    d0_start, d0_end = find_bounds_1d(dim0_sum)
    d1_start, d1_end = find_bounds_1d(dim1_sum)
    d2_start, d2_end = find_bounds_1d(dim2_sum)

    bounding_box = (
        (int(d0_start), int(d0_end + 1)),
        (int(d1_start), int(d1_end + 1)),
        (int(d2_start), int(d2_end + 1)),
    )

    cropped_volume = apply_bounding_box(volume, bounding_box)

    return cropped_volume, bounding_box


def apply_bounding_box(volume: np.ndarray, bounding_box: tuple) -> np.ndarray:
    (d0_start, d0_end), (d1_start, d1_end), (d2_start, d2_end) = bounding_box

    cropped_volume = volume[
        d0_start:d0_end, 
        d1_start:d1_end, 
        d2_start:d2_end
    ]

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


def extract_brain(volume: nib.Nifti1Image) -> nib.Nifti1Image:
    fsldir = Path(os.getenv("FSLDIR", ""))
    if not fsldir.is_dir():
        raise EnvironmentError("Valid FSLDIR environment variable is not set. Set it using 'export FSLDIR=/path/to/fsl'.")

    with tempfile.TemporaryDirectory(prefix="microbleednet-fsl-bet-") as temp_dir:
        temp_dir = Path(temp_dir)
        input_path = temp_dir / "pre_bet.nii.gz"
        output_path = temp_dir / "post_bet.nii.gz"

        # Save to disk just for BET
        utils.save_volume(volume, input_path)
        subprocess.run(
            [str(fsldir / "bin" / "bet"), str(input_path), str(output_path)], check=True
        )

        # Some potentially over-defensive programming:
        # Load back into memory immediately and let tempdir delete the files
        output_volume = utils.load_volume(output_path)
        # Force load data into memory so we don't rely on the deleted temp file
        volume = utils.nifti_to_numpy(output_volume)
        volume = utils.numpy_to_nifti(volume, output_volume)

        return volume


def bias_field_correct(volume: nib.Nifti1Image) -> nib.Nifti1Image:
    volume_data = utils.nifti_to_numpy(volume).astype(float)

    # Transpose for SimpleITK coordinate system
    sitk_volume = sitk.GetImageFromArray(volume_data.T)

    # Create a mask of the brain tissue to ignore empty space
    mask_image = sitk_volume > 0

    # Run highly optimized N4 correction
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrected_sitk = corrector.Execute(sitk_volume, mask_image)

    # Convert back to NumPy and nibabel
    corrected_data = sitk.GetArrayFromImage(corrected_sitk).T
    corrected_nifti = utils.numpy_to_nifti(corrected_data, volume)

    return corrected_nifti

def calculate_voxel_weights(volume: np.ndarray) -> np.ndarray:
    return gaussian_filter(volume, 1.2) * 10

def _fsl_process(
    volume: nib.Nifti1Image,
    reorient_to_std: bool,
    extract_brain: bool,
    bias_field_correct: bool,
    verbose: bool,
) -> nib.Nifti1Image:
    """
    Preprocess a NIfTI volume using FSL tools.

    Args:
        volume: The input NIfTI volume.
        reorient_to_std: Whether to reorient the volume to standard orientation.
        extract_brain: Whether to extract the brain from the volume.
        bias_field_correct: Whether to correct for bias field inhomogeneities.
        verbose: Enable verbose output.

    Returns:
        The preprocessed NIfTI image.
    """

    fsldir = os.getenv("FSLDIR")
    if not fsldir:
        raise EnvironmentError("FSLDIR environment variable is not set.")

    fsldir = Path(fsldir)

    with tempfile.TemporaryDirectory(prefix="microbleednet-fsl-process-") as temp_dir:
        temp_dir = Path(temp_dir)
        temp_volume_path = temp_dir / "temp_volume.nii.gz"
        nib.save(volume, temp_volume_path)

        if reorient_to_std:
            reoriented_path = temp_dir / "reoriented.nii.gz"
            subprocess.run(
                [
                    str(fsldir / "bin" / "fslreorient2std"),
                    str(temp_volume_path),
                    str(reoriented_path),
                ],
                check=True,
            )
            temp_volume_path = reoriented_path

            if verbose:
                print(f"Reoriented volume saved to: {reoriented_path}")

        if extract_brain:
            brain_extracted_path = temp_dir / "brain_extracted.nii.gz"
            subprocess.run(
                [
                    str(fsldir / "bin" / "bet"),
                    str(temp_volume_path),
                    str(brain_extracted_path),
                ],
                check=True,
            )
            temp_volume_path = brain_extracted_path

            if verbose:
                print(f"Brain extracted volume saved to: {brain_extracted_path}")

        if bias_field_correct:
            bias_corrected_path = temp_volume_path
            while bias_corrected_path.suffix:
                bias_corrected_path = bias_corrected_path.with_suffix("")
            bias_corrected_path = bias_corrected_path.with_name(
                bias_corrected_path.name + "_restore.nii.gz"
            )

            subprocess.run(
                [str(fsldir / "bin" / "fast"), "-B", "--nopve", str(temp_volume_path)],
                check=True,
            )
            temp_volume_path = bias_corrected_path

            if verbose:
                print(f"Bias corrected volume saved to: {bias_corrected_path}")

        processed_volume = nib.load(temp_volume_path)
        processed_volume = nib.Nifti1Image(
            processed_volume.get_fdata(),
            processed_volume.affine,
            processed_volume.header,
        )
        return processed_volume
