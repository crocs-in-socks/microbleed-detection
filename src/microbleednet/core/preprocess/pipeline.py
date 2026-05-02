from typing import Optional

import nibabel as nib

from microbleednet.core.utils import *
from microbleednet.core.preprocess import frst
from microbleednet.core.preprocess import standardize
from microbleednet.core.preprocess import inpaint_vessels

def run(
    volume: nib.Nifti1Image,
    label: Optional[nib.Nifti1Image],
    reorient_to_std: bool,
    extract_brain: bool,
    bias_field_correct: bool,
) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:

    if reorient_to_std:
        volume = standardize.reorient_to_std(volume)
    if extract_brain:
        volume = standardize.extract_brain(volume)
    if bias_field_correct:
        volume = standardize.bias_field_correct(volume)

    volume = nifti_to_numpy(volume).astype(float)
    volume = inpaint_vessels.apply(volume)
    frst_volume = frst.apply(volume)

    # Invert volume
    volume = normalize_volume(volume)
    volume = invert_volume(volume)

    volume, bounding_box = tight_crop_volume(volume)
    frst_volume = frst_volume[bounding_box]

    if label is not None:
        label = nifti_to_numpy(label).astype(int)
        label = label[bounding_box]
    
    return volume, frst_volume, label
