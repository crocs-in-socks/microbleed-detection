import os
import tempfile
import subprocess
import numpy as np
import nibabel as nib
from pathlib import Path
import SimpleITK as sitk

from microbleednet.core.utils import *


def fsl_process(
    volume: nib.Nifti1Image,
    reorient_to_std: bool,
    extract_brain: bool,
    bias_field_correct: bool,
    verbose: bool,
) -> nib.Nifti1Image:

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


def reorient_to_std(volume: nib.Nifti1Image) -> nib.Nifti1Image:
    return nib.as_closest_canonical(volume)


def extract_brain(volume: nib.Nifti1Image) -> nib.Nifti1Image:
    fsldir = Path(os.getenv("FSLDIR", ""))
    if not fsldir.is_dir():
        raise EnvironmentError("Valid FSLDIR environment variable is not set.")

    with tempfile.TemporaryDirectory(prefix="microbleednet-fsl-bet-") as temp_dir:
        temp_dir = Path(temp_dir)
        input_path = temp_dir / "pre_bet.nii.gz"
        output_path = temp_dir / "post_bet.nii.gz"

        nib.save(volume, input_path)
        subprocess.run(
            [str(fsldir / "bin" / "bet"), str(input_path), str(output_path)], check=True
        )

        # Load back into memory immediately and let tempdir delete the files
        volume = nib.load(output_path)
        # Force load data into memory so we don't rely on the deleted temp file
        volume = nib.Nifti1Image(volume.get_fdata(), volume.affine, volume.header)

        return volume


def bias_field_correct(volume: nib.Nifti1Image) -> nib.Nifti1Image:
    volume = nifti_to_numpy(volume).astype(float)
    sitk_volume = sitk.GetImageFromArray(volume.T)
    mask_image = sitk_volume > 0
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrected_sitk = corrector.Execute(sitk_volume, mask_image)
    corrected_data = sitk.GetArrayFromImage(corrected_sitk).T
    volume = nib.Nifti1Image(corrected_data, volume.affine, volume.header)

    return volume
