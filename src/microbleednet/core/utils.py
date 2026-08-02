import logging
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn


logger = logging.getLogger(__name__)

CHECKPOINT_FORMAT_VERSION = 1


def unwrap_model(model: nn.Module) -> nn.Module:
    return model._orig_mod if hasattr(model, "_orig_mod") else model


def load_volume(path: Path) -> nib.Nifti1Image:
    return nib.load(path)


def save_volume(volume: nib.Nifti1Image, path: Path) -> None:
    nib.save(volume, path)


def nifti_to_numpy(volume: nib.Nifti1Image) -> np.ndarray:
    return volume.get_fdata()


def numpy_to_nifti(array: np.ndarray, reference: nib.Nifti1Image | None = None) -> nib.Nifti1Image:
    if reference is None:
        return nib.Nifti1Image(array, np.eye(4), nib.Nifti1Header())
    return nib.Nifti1Image(array, reference.affine, reference.header)

def load_model_weights(model: nn.Module, device: torch.device, checkpoint_path: Path):
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path.resolve()}")

    logger.info("Loading weights from: %s.", checkpoint_path.resolve())

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    missing, unexpected = unwrap_model(model).load_state_dict(state_dict, strict=False)

    if missing or unexpected:
        raise RuntimeError(f"checkpoint keys do not match; missing={missing}, unexpected={unexpected}")

    return checkpoint


def initialize_teacher_from_detector(detector: nn.Module, teacher: nn.Module) -> None:
    detector_state = unwrap_model(detector).state_dict()
    teacher_state = unwrap_model(teacher).state_dict()
    transferable = {
        key: value
        for key, value in detector_state.items()
        if key.startswith(("feature_extractor.", "segmentor."))
    }
    missing = [key for key in transferable if key not in teacher_state]
    if missing:
        raise RuntimeError(f"detector-to-teacher keys missing in teacher: {missing}")
    teacher_state.update(transferable)
    unwrap_model(teacher).load_state_dict(teacher_state, strict=True)