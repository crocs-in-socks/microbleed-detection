from typing import Optional

import torch
import numpy as np
import torch.nn as nn
import nibabel as nib
from torch.amp import autocast

from microbleednet.core import utils
from microbleednet.core import transforms
from microbleednet.core.transforms import frst


def preprocess(
    volume: nib.Nifti1Image,
    mask: Optional[nib.Nifti1Image],
    reorient_to_std: bool,
    extract_brain: bool,
    bias_field_correct: bool,
    invert_volume: bool,
    inpaint_vessels: bool
) -> dict[np.ndarray, Optional[np.ndarray], np.ndarray]:

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

    return {
        "volume": volume,
        "mask": mask,
        "bounding_box": bounding_box
    }

def infer(
    model: nn.Module,
    device: torch.device,
    volume: np.ndarray
):
    volume = np.expand_dims(volume, axis=(0, 1)) # Shape: (1, 1, H, W, D)
    volume = torch.from_numpy(volume).float().to(device)

    volume_frst = frst.apply(volume)
    volume = torch.cat((volume, volume_frst), dim=1)

    model = model.to(device)
    model.eval()

    use_amp = (device.type == "cuda")
    amp_dtype = torch.float16 if use_amp else torch.bfloat16

    with torch.no_grad():
        with autocast(device_type=device.type, dtype=amp_dtype):
            logits = model(volume)
    
    return logits