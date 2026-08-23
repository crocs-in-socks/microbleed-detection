from typing import Optional, cast

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from nibabel.affines import voxel_sizes
from nibabel.orientations import apply_orientation, io_orientation, ornt_transform
from torch.amp.autocast_mode import autocast

from microbleednet.core import io
from microbleednet.core.datamodels import (
    CropTransform,
    FloatArray,
    ImageGeometry,
    IntArray,
    Modality,
    PreprocessResult,
)
from microbleednet.core.transforms import frst, inpaint_vessels, volume_ops


def preprocess(
    volume: nib.Nifti1Image,
    mask: Optional[nib.Nifti1Image],
    modality: Modality,
) -> PreprocessResult:
    # cast() here is a static type-checker hint, not a runtime conversion: it
    # narrows nibabel's loosely-typed shape (tuple[int, ...]) and optional affine
    # (ndarray | None) to the fixed types CropTransform declares. The values are
    # not coerced -- the real 3-D / finite-affine invariant is enforced when this
    # feeds CropTransform below, whose __post_init__ validates shape and affine.
    source_shape = cast(tuple[int, int, int], tuple(int(v) for v in volume.shape))
    source_affine = cast(FloatArray, volume.affine).copy()
    source_orientation = io_orientation(source_affine)
    canonical_volume = volume_ops.reorient_to_std(volume)
    target_orientation = io_orientation(cast(FloatArray, canonical_volume.affine))
    orientation_transform = ornt_transform(source_orientation, target_orientation)

    if mask is not None:
        if not np.allclose(cast(FloatArray, mask.affine), source_affine):
            raise ValueError("image and mask affines do not match")
        mask = volume_ops.reorient_to_std(mask)
        if mask.shape != canonical_volume.shape:
            raise ValueError("reoriented image and mask shapes do not match")

    processed_volume = volume_ops.extract_brain(canonical_volume)
    if modality in {"T2*-GRE", "SWI"}:
        processed_volume = volume_ops.bias_field_correct_n4(processed_volume)

    volume_array = io.nifti_to_numpy(processed_volume).astype(np.float32)
    volume_array = volume_ops.normalize_volume(volume_array)

    if modality in {"T2*-GRE", "SWI"}:
        volume_array = volume_ops.invert_volume(volume_array)

    volume_array, bounding_box = volume_ops.tight_crop_volume(volume_array)
    crop_start = cast(tuple[int, int, int], tuple(b[0] for b in bounding_box))
    crop_stop = cast(tuple[int, int, int], tuple(b[1] for b in bounding_box))
    mask_array: Optional[IntArray] = None
    if mask is not None:
        mask_array = io.nifti_to_numpy(mask).astype(np.int16)
        mask_array = volume_ops.apply_bounding_box(mask_array, bounding_box)

    volume_array = inpaint_vessels.apply(volume_array)

    canonical_affine = cast(FloatArray, canonical_volume.affine)
    cropped_affine = volume_ops.crop_affine(canonical_affine, crop_start)
    geometry = ImageGeometry(
        shape=cast(tuple[int, int, int], tuple(int(v) for v in volume_array.shape)),
        affine=cropped_affine,
        spacing=cast(
            tuple[float, float, float],
            tuple(float(value) for value in voxel_sizes(cropped_affine)),
        ),
    )
    transform = CropTransform(
        source_shape=source_shape,
        canonical_shape=cast(
            tuple[int, int, int], tuple(int(v) for v in canonical_volume.shape)
        ),
        crop_start=crop_start,
        crop_stop=crop_stop,
        source_affine=source_affine,
        canonical_affine=canonical_affine,
        orientation_transform=cast(IntArray, orientation_transform),
    )
    return PreprocessResult(volume_array, mask_array, geometry, transform)


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
    for source_axis, (canonical_axis, flip) in enumerate(
        transform.orientation_transform
    ):
        inverse_orientation[int(canonical_axis)] = (source_axis, flip)
    source = apply_orientation(canonical, inverse_orientation)
    return nib.Nifti1Image(source, transform.source_affine)


def infer(model: nn.Module, device: torch.device, volume: np.ndarray):
    volume_array = np.expand_dims(volume, axis=(0, 1))  # Shape: (1, 1, H, W, D)
    volume_tensor = torch.from_numpy(volume_array).float().to(device)

    volume_tensor = frst.prepend_frst_channel(volume_tensor)

    model = model.to(device)
    model.eval()

    with torch.no_grad():
        if device.type == "cuda":
            with autocast(device_type=device.type, dtype=torch.float16):
                logits = model(volume_tensor)
        else:
            logits = model(volume_tensor)

    return logits



