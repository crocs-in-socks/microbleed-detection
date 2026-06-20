import numpy as np
import nibabel as nib
from pathlib import Path

import torch
import torch.nn as nn


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
        print(f"No checkpoint found at {checkpoint_path.resolve()}.")
        return None

    print(f"Loading weights from: {checkpoint_path.resolve()}.")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if missing or unexpected:
        print("Note: Some keys did not match perfectly.")

    return checkpoint