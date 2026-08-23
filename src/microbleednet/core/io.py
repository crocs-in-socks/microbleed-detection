"""NIfTI, array, and checkpoint I/O used by core operations."""

import logging
import os
import tempfile
from pathlib import Path
from typing import Any, cast

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn

from .utils import unwrap_model

logger = logging.getLogger(__name__)

CHECKPOINT_FORMAT_VERSION = 1


def load_volume(path: Path) -> nib.Nifti1Image:
    return cast(nib.Nifti1Image, nib.load(path))


def save_volume(volume: nib.Nifti1Image, path: Path) -> None:
    nib.save(volume, path)


def nifti_to_numpy(volume: nib.Nifti1Image) -> np.ndarray:
    return volume.get_fdata()


def numpy_to_nifti(
    array: np.ndarray, reference: nib.Nifti1Image | None = None
) -> nib.Nifti1Image:
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
        raise RuntimeError(
            f"checkpoint keys do not match; missing={missing}, unexpected={unexpected}"
        )
    return checkpoint


def save_array_atomic(array: np.ndarray, path: Path) -> None:
    """Write a NumPy array without exposing a partial file to readers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, suffix=".npy", delete=False
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)
        np.save(temporary_file, array)
        temporary_file.flush()
        os.fsync(temporary_file.fileno())
    os.replace(temporary_path, path)


def load_array_mmap(path: str | Path) -> np.ndarray:
    """Open a NumPy array for memory-mapped, read-only access."""
    return cast(np.ndarray, np.load(path, mmap_mode="r"))


def save_checkpoint_atomic(state: dict[str, Any], path: Path) -> None:
    """Write a Torch checkpoint without exposing a partial file to readers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, suffix=".pth", delete=False
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)
        torch.save(state, temporary_file)
        temporary_file.flush()
        os.fsync(temporary_file.fileno())
    os.replace(temporary_path, path)