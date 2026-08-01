from typing import Optional

import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
from torch.amp import autocast
from nibabel.orientations import io_orientation, ornt_transform, apply_orientation

from microbleednet.core import utils
from microbleednet.core import transforms
from microbleednet.core.transforms import frst
from microbleednet.records import CropTransform, ImageGeometry, PreprocessResult


def preprocess(
    volume: nib.Nifti1Image,
    mask: Optional[nib.Nifti1Image],
    canonical_orientation: bool,
    extract_brain: bool,
    bias_field_correction: bool,
    invert_volume: bool,
    inpaint_vessels: bool,
) -> PreprocessResult:
    source_shape = tuple(int(value) for value in volume.shape)
    source_affine = volume.affine.copy()
    source_orientation = io_orientation(source_affine)
    if canonical_orientation:
        canonical_volume = transforms.basic.reorient_to_std(volume)
        target_orientation = io_orientation(canonical_volume.affine)
        orientation_transform = ornt_transform(source_orientation, target_orientation)
    else:
        canonical_volume = volume
        orientation_transform = np.array(
            [[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]], dtype=np.float64
        )

    if mask is not None:
        if not np.allclose(mask.affine, source_affine):
            raise ValueError("image and mask affines do not match")
        if canonical_orientation:
            mask = transforms.basic.reorient_to_std(mask)
        if mask.shape != canonical_volume.shape:
            raise ValueError("reoriented image and mask shapes do not match")

    volume = canonical_volume
    if extract_brain:
        volume = transforms.basic.extract_brain(volume)

    if bias_field_correction:
        volume = transforms.basic.bias_field_correct_n4(volume)

    volume = utils.nifti_to_numpy(volume).astype(np.float32)
    volume = transforms.basic.normalize_volume(volume)

    if invert_volume:
        volume = transforms.basic.invert_volume(volume)

    volume, bounding_box = transforms.basic.tight_crop_volume(volume)
    crop_start = tuple(bounds[0] for bounds in bounding_box)
    crop_stop = tuple(bounds[1] for bounds in bounding_box)
    if mask is not None:
        mask = utils.nifti_to_numpy(mask).astype(np.int16)
        mask = transforms.basic.apply_bounding_box(mask, bounding_box)

    if inpaint_vessels:
        volume = transforms.inpaint_vessels.apply(volume)

    canonical_affine = canonical_volume.affine
    cropped_affine = transforms.basic.crop_affine(canonical_affine, crop_start)
    geometry = ImageGeometry(
        shape=tuple(int(value) for value in volume.shape),
        affine=cropped_affine,
        spacing=tuple(float(value) for value in nib.affines.voxel_sizes(cropped_affine)),
    )
    transform = CropTransform(
        source_shape=source_shape,
        canonical_shape=tuple(int(value) for value in canonical_volume.shape),
        crop_start=crop_start,
        crop_stop=crop_stop,
        source_affine=source_affine,
        canonical_affine=canonical_affine,
        orientation_transform=orientation_transform,
    )
    return PreprocessResult(volume, mask, geometry, transform)


def restore_to_source(array: np.ndarray, transform: CropTransform) -> nib.Nifti1Image:
    if tuple(array.shape) != tuple(
        stop - start for start, stop in zip(transform.crop_start, transform.crop_stop)
    ):
        raise ValueError("array shape does not match the crop transform")

    canonical = np.zeros(transform.canonical_shape, dtype=array.dtype)
    slices = tuple(
        slice(start, stop)
        for start, stop in zip(transform.crop_start, transform.crop_stop)
    )
    canonical[slices] = array
    inverse_orientation = np.empty_like(transform.orientation_transform)
    for source_axis, (canonical_axis, flip) in enumerate(transform.orientation_transform):
        inverse_orientation[int(canonical_axis)] = (source_axis, flip)
    source = apply_orientation(canonical, inverse_orientation)
    return nib.Nifti1Image(source, transform.source_affine)

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

    with torch.no_grad():
        if device.type == "cuda":
            with autocast(device_type=device.type, dtype=torch.float16):
                logits = model(volume)
        else:
            logits = model(volume)
    
    return logits