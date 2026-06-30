from pathlib import Path

import torch
import torch.nn.functional as F

from ..core import utils as core_utils
from ..core.common.models import CandidateDetector 
from ..core.engines import processor as core_processor
from ..core.dataloading import patchers as core_patchers


def collect_patches(subjects: list, subject_patcher: function, patcher_parameters: dict):
    patches = []
    for subject in subjects:
        patches.extend(subject_patcher(subject, **patcher_parameters))
    return patches

def patch_subject_non_overlapping(subject: dict, patch_dir: Path, patch_size: int, augmentation_factor: int):
    subject_id = subject["subject_id"]
    volume_path = subject["volume_path"]
    mask_path = subject["mask_path"]

    volume = core_utils.nifti_to_numpy(core_utils.load_volume(volume_path))
    mask = core_utils.nifti_to_numpy(core_utils.load_volume(mask_path))

    patches = core_patchers.nonoverlapping_patcher(volume, mask, patch_size)
    patches = core_patchers.materialize_patches(patches, patch_dir, subject_id, augmentation_factor)

    return patches


def patch_subject_target_centered(subject: dict, patch_dir: Path, patch_size: int, augmentation_factor: int, model: CandidateDetector, device: torch.device, threshold: float):
    subject_id = subject["subject_id"]
    volume_path = subject["volume_path"]
    mask_path = subject["mask_path"]

    volume = core_utils.nifti_to_numpy(core_utils.load_volume(volume_path))
    mask = core_utils.nifti_to_numpy(core_utils.load_volume(mask_path))

    logits = core_processor.infer(model, device, volume)
    output = F.softmax(logits, dim=1)
    output = output.cpu().numpy()[0, 1] # 0 to remove batch, and index 1 for output channel
    output = (output > threshold).astype(int)

    patches = core_patchers.target_centered_patcher(volume, mask, patch_size)
    patches = core_patchers.materialize_patches(patches, patch_dir, subject_id, augmentation_factor)

    return patches