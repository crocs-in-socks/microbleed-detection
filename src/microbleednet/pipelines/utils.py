import gc
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from skimage.measure import label, regionprops

from ..core import utils as core_utils
from ..core.common.models import CandidateDetector
from ..core.dataloading import patchers as core_patchers
from ..core.engines import processor as core_processor
from ..manifests import PreprocessedSubject


def delete_model(model):
    del model
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def collect_patches(subjects: list, subject_patcher: Callable, patcher_parameters: dict):
    patches = []
    for subject in subjects:
        patches.extend(subject_patcher(subject, **patcher_parameters))
    return patches


def _load_array(path: str) -> np.ndarray:
    """Load a NIfTI volume at ``path`` and return it as a numpy array."""
    return core_utils.nifti_to_numpy(core_utils.load_volume(Path(path)))

def patch_subject_non_overlapping(subject: PreprocessedSubject, patch_dir: Path, patch_size: int, augmentation_factor: int):
    subject_id = subject.subject_id
    if subject.mask_path is None:
        raise ValueError(f"training subject has no mask: {subject_id}")

    volume = _load_array(subject.volume_path)
    mask = _load_array(subject.mask_path)

    patches = core_patchers.nonoverlapping_patcher(volume, mask, patch_size)
    patches = core_patchers.materialize_patches(patches, patch_dir, subject_id, augmentation_factor)

    return patches


def patch_subject_target_centered(subject: PreprocessedSubject, patch_dir: Path, patch_size: int, augmentation_factor: int, model: CandidateDetector, device: torch.device, threshold: float):
    subject_id = subject.subject_id
    if subject.mask_path is None:
        raise ValueError(f"training subject has no mask: {subject_id}")

    volume = _load_array(subject.volume_path)
    mask = _load_array(subject.mask_path)

    logits = core_processor.infer(model, device, volume)
    output = core_processor.positive_class_probability(logits)[0]  # drop batch axis
    candidate_mask = (output > threshold).astype(np.uint8)
    candidate_labels = label(candidate_mask, connectivity=3)
    candidate_probabilities = {
        region.label: float(output[candidate_labels == region.label].mean())
        for region in regionprops(candidate_labels)
    }

    patches = core_patchers.target_centered_patcher(volume, mask, candidate_mask, patch_size)
    for patch_data in patches:
        candidate_id = patch_data["candidate_id"]
        patch_data["candidate_probability"] = candidate_probabilities[candidate_id]
    patches = core_patchers.materialize_patches(patches, patch_dir, subject_id, augmentation_factor)

    return patches