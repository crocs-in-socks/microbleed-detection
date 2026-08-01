from pathlib import Path

import nibabel as nib
import numpy as np
import pytest


@pytest.fixture
def synthetic_affine() -> np.ndarray:
    return np.diag([1.0, 1.0, 1.0, 1.0])


@pytest.fixture
def synthetic_image() -> np.ndarray:
    image = np.zeros((8, 8, 8), dtype=np.float32)
    image[2:6, 2:6, 2:6] = 1.0
    image[3, 3, 3] = 2.0
    return image


@pytest.fixture
def synthetic_mask() -> np.ndarray:
    mask = np.zeros((8, 8, 8), dtype=np.uint8)
    mask[3, 3, 3] = 1
    return mask


@pytest.fixture
def boundary_mask() -> np.ndarray:
    mask = np.zeros((8, 8, 8), dtype=np.uint8)
    mask[0, 0, 0] = 1
    return mask


@pytest.fixture
def anisotropic_spacing() -> tuple[float, float, float]:
    return (1.0, 2.0, 3.0)


@pytest.fixture
def synthetic_nifti_pair(
    tmp_path: Path,
    synthetic_image: np.ndarray,
    synthetic_mask: np.ndarray,
    synthetic_affine: np.ndarray,
) -> tuple[Path, Path]:
    image_path = tmp_path / "subject01_T2star.nii.gz"
    mask_path = tmp_path / "subject01_mask.nii.gz"
    nib.save(nib.Nifti1Image(synthetic_image, synthetic_affine), image_path)
    nib.save(nib.Nifti1Image(synthetic_mask, synthetic_affine), mask_path)
    return image_path, mask_path