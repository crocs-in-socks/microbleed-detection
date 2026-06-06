from typing import Optional

import numpy as np
import nibabel as nib

from microbleednet.core import utils
from microbleednet.core import transforms


def run(
    volume: nib.Nifti1Image,
    mask: Optional[nib.Nifti1Image],
    reorient_to_std: bool,
    extract_brain: bool,
    bias_field_correct: bool,
    invert_volume: bool,
    inpaint_vessels: bool
) -> tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:

    if reorient_to_std:
        volume = transforms.basic.reorient_to_std(volume)
        print("reoriented volume")
        if mask is not None:
            mask = transforms.basic.reorient_to_std(mask)
            print("reoriented mask")
            
    if extract_brain:
        volume = transforms.basic.extract_brain(volume)
        print("extracted brain")

    if bias_field_correct:
        volume = transforms.basic.bias_field_correct(volume)
        print("bias field corrected")

    volume = utils.nifti_to_numpy(volume).astype(float)
    volume = transforms.basic.normalize_volume(volume)
    print("normalized volume")

    if invert_volume:
        volume = transforms.basic.invert_volume(volume)
        print("inverted volume")

    volume, bounding_box = transforms.basic.tight_crop_volume(volume)
    print("tightly cropped volume")
    if mask is not None:
        mask = utils.nifti_to_numpy(mask).astype(int)
        mask = transforms.basic.apply_bounding_box(mask, bounding_box)
        print("applied bounding box to mask")

    if inpaint_vessels:
        volume = transforms.inpaint_vessels.apply(volume)
        print("inpainted vessels")

    return volume, mask, bounding_box

