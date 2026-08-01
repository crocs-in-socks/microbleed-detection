import numpy as np
import pytest

from microbleednet.core.dataloading.patchers import target_centered_patcher
from microbleednet.core.transforms.patch import (
    get_nonoverlapping_patches,
    get_target_centered_patches,
)


@pytest.mark.parametrize("center", [(0, 0, 0), (0, 3, 3), (3, 0, 3), (3, 3, 0), (3, 3, 3)])
def test_boundary_candidates_have_fixed_shape_and_center(center: tuple[int, int, int]) -> None:
    volume = np.zeros((7, 7, 7), dtype=np.float32)
    target = np.zeros_like(volume, dtype=np.uint8)
    volume[center] = 1
    target[center] = 1

    patch = get_target_centered_patches(volume, target, 4)[0]

    assert patch.shape == (4, 4, 4)
    assert patch[2, 2, 2] == 1


@pytest.mark.parametrize("patch_size", [24, 48, 5])
def test_nonoverlapping_patches_have_requested_shape(patch_size: int) -> None:
    volume = np.ones((7, 8, 9), dtype=np.float32)

    patches = get_nonoverlapping_patches(volume, patch_size)

    assert patches
    assert all(item.shape == (patch_size, patch_size, patch_size) for item in patches)


def test_multiple_candidates_produce_one_patch_each() -> None:
    volume = np.ones((8, 8, 8), dtype=np.float32)
    target = np.zeros_like(volume, dtype=np.uint8)
    target[1, 1, 1] = 1
    target[6, 6, 6] = 1

    patches = get_target_centered_patches(volume, target, 4)

    assert len(patches) == 2


def test_partial_reference_overlap_is_positive() -> None:
    volume = np.ones((8, 8, 8), dtype=np.float32)
    candidate = np.zeros_like(volume, dtype=np.uint8)
    reference = np.zeros_like(volume, dtype=np.uint8)
    candidate[3, 3, 3] = 1
    reference[3, 3, 3] = 1
    reference[4, 3, 3] = 1

    patches = target_centered_patcher(volume, reference, candidate, 4)

    assert len(patches) == 1
    assert np.any(patches[0]["mask"] > 0)


def test_empty_target_produces_no_candidate_patches() -> None:
    volume = np.ones((8, 8, 8), dtype=np.float32)
    target = np.zeros_like(volume, dtype=np.uint8)

    assert get_target_centered_patches(volume, target, 4) == []


def test_invalid_patch_size_fails() -> None:
    volume = np.ones((3, 3, 3))

    with pytest.raises(ValueError, match="positive"):
        get_nonoverlapping_patches(volume, 0)