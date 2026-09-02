from typing import Optional, cast

import nibabel as nib
import numpy as np

from .. import io
from ..datamodels import (
    FloatArray,
    IntArray,
    Modality,
    PreprocessResult,
    Shape3D,
)
from ..transforms import inpaint_vessels, volume_ops


def preprocess(
    volume: nib.Nifti1Image,
    mask: Optional[nib.Nifti1Image],
    modality: Modality,
) -> PreprocessResult:
    canonical_volume = volume_ops.reorient_to_canonical(volume)

    if mask is not None:
        if not np.allclose(
            cast(FloatArray, mask.affine), cast(FloatArray, volume.affine)
        ):
            raise ValueError("image and mask affines do not match")
        mask = volume_ops.reorient_to_canonical(mask)
        if mask.shape != canonical_volume.shape:
            raise ValueError("reoriented image and mask shapes do not match")

    processed_volume = volume_ops.extract_brain(canonical_volume)
    if modality in {"T2*-GRE", "SWI"}:
        processed_volume = volume_ops.bias_field_correct_n4(processed_volume)

    volume_array = io.nifti_to_numpy(processed_volume)
    volume_array = volume_ops.normalize_volume(volume_array)

    if modality in {"T2*-GRE", "SWI"}:
        volume_array = volume_ops.invert_volume(volume_array)

    volume_array, bounding_box = volume_ops.tight_crop_volume(volume_array)
    mask_array: IntArray | None = None
    if mask is not None:
        mask_array = io.nifti_to_numpy(mask).astype(np.uint8)
        mask_array = volume_ops.apply_bounding_box(mask_array, bounding_box)

    volume_array = inpaint_vessels.apply(volume_array)

    crop_start = cast(Shape3D, tuple(b[0] for b in bounding_box))
    canonical_affine = cast(FloatArray, canonical_volume.affine)
    cropped_affine = volume_ops.adjust_affine_for_crop(canonical_affine, crop_start)

    return PreprocessResult(volume_array, mask_array, cropped_affine)
