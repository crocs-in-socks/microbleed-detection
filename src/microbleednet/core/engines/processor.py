from typing import Optional, cast

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from nibabel.affines import voxel_sizes
from nibabel.orientations import apply_orientation, io_orientation, ornt_transform
from skimage.measure import label, regionprops
from torch.amp.autocast_mode import autocast

from microbleednet.core import transforms, utils
from microbleednet.core.transforms import frst
from microbleednet.records import (
    CropTransform,
    FloatArray,
    ImageGeometry,
    IntArray,
    PreprocessResult,
)


def preprocess(
    volume: nib.Nifti1Image,
    mask: Optional[nib.Nifti1Image],
    canonical_orientation: bool,
    extract_brain: bool,
    bias_field_correction: bool,
    invert_volume: bool,
    inpaint_vessels: bool,
) -> PreprocessResult:
    source_shape = cast(tuple[int, int, int], tuple(int(v) for v in volume.shape))
    source_affine = cast(FloatArray, volume.affine).copy()
    source_orientation = io_orientation(source_affine)
    if canonical_orientation:
        canonical_volume = transforms.basic.reorient_to_std(volume)
        target_orientation = io_orientation(cast(FloatArray, canonical_volume.affine))
        orientation_transform = ornt_transform(source_orientation, target_orientation)
    else:
        canonical_volume = volume
        orientation_transform = np.array(
            [[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]], dtype=np.float64
        )

    if mask is not None:
        if not np.allclose(cast(FloatArray, mask.affine), source_affine):
            raise ValueError("image and mask affines do not match")
        if canonical_orientation:
            mask = transforms.basic.reorient_to_std(mask)
        if mask.shape != canonical_volume.shape:
            raise ValueError("reoriented image and mask shapes do not match")

    processed_volume = canonical_volume
    if extract_brain:
        processed_volume = transforms.basic.extract_brain(processed_volume)

    if bias_field_correction:
        processed_volume = transforms.basic.bias_field_correct_n4(processed_volume)

    volume_array = utils.nifti_to_numpy(processed_volume).astype(np.float32)
    volume_array = transforms.basic.normalize_volume(volume_array)

    if invert_volume:
        volume_array = transforms.basic.invert_volume(volume_array)

    volume_array, bounding_box = transforms.basic.tight_crop_volume(volume_array)
    crop_start = cast(tuple[int, int, int], tuple(b[0] for b in bounding_box))
    crop_stop = cast(tuple[int, int, int], tuple(b[1] for b in bounding_box))
    mask_array: Optional[IntArray] = None
    if mask is not None:
        mask_array = utils.nifti_to_numpy(mask).astype(np.int16)
        mask_array = transforms.basic.apply_bounding_box(mask_array, bounding_box)

    if inpaint_vessels:
        volume_array = transforms.inpaint_vessels.apply(volume_array)

    canonical_affine = cast(FloatArray, canonical_volume.affine)
    cropped_affine = transforms.basic.crop_affine(canonical_affine, crop_start)
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


def positive_class_probability(logits: torch.Tensor) -> np.ndarray:
    """Softmax over the class axis, returning the positive-class channel as numpy.

    Every consumer of a two-class output wants the same thing: the per-voxel
    probability of the foreground (index 1) class. Input shape (Batch, 2, ...);
    output shape (Batch, ...).
    """
    return F.softmax(logits, dim=1)[:, 1].cpu().numpy()


def label_candidates(
    candidate_mask: np.ndarray, probability: np.ndarray
) -> tuple[np.ndarray, list, dict[int, float]]:
    """Label connected candidate components and score each by mean probability.

    Runs 26-connected labeling (``connectivity=3``) over ``candidate_mask`` and
    returns the label volume, the ``skimage`` regions, and a mapping from each
    region label to the mean ``probability`` over that component's voxels.
    Callers own the threshold that produced ``candidate_mask`` — this is the
    shared labeling-and-scoring step that follows it, with no behavior of its
    own.
    """
    candidate_labels = cast(np.ndarray, label(candidate_mask, connectivity=3))
    regions = regionprops(candidate_labels)
    mean_probabilities: dict[int, float] = {
        int(region.label): float(probability[candidate_labels == region.label].mean())
        for region in regions
    }
    return candidate_labels, regions, mean_probabilities
