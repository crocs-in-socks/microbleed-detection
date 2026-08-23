from pathlib import Path

import numpy as np
import pytest
import torch

from microbleednet.core.dataloading.patchers import (
    materialize_patches,
    target_centered_patcher,
)
from microbleednet.core.datamodels import ExtractedPatch, ModelArchitecture
from microbleednet.core.postprocessing.components import (
    component_centers,
    describe_components,
)
from microbleednet.core.transforms.patch import (
    extract_centered_patches,
    get_nonoverlapping_patches,
)


def _component_centers(mask: np.ndarray) -> list[tuple[int, int, int]]:
    _, components = describe_components(mask)
    return component_centers(components)


@pytest.mark.parametrize(
    "center", [(0, 0, 0), (0, 3, 3), (3, 0, 3), (3, 3, 0), (3, 3, 3)]
)
def test_boundary_candidates_have_fixed_shape_and_center(
    center: tuple[int, int, int],
) -> None:
    volume = np.zeros((7, 7, 7), dtype=np.float32)
    target = np.zeros_like(volume, dtype=np.uint8)
    volume[center] = 1
    target[center] = 1

    patch = extract_centered_patches(volume, _component_centers(target), 4)[0]

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

    patches = extract_centered_patches(volume, _component_centers(target), 4)

    assert len(patches) == 2


def test_partial_reference_overlap_is_positive() -> None:
    volume = np.ones((8, 8, 8), dtype=np.float32)
    candidate = np.zeros_like(volume, dtype=np.uint8)
    reference = np.zeros_like(volume, dtype=np.uint8)
    candidate[3, 3, 3] = 1
    reference[3, 3, 3] = 1
    reference[4, 3, 3] = 1

    patches = target_centered_patcher(
        volume, reference, _component_centers(candidate), 4
    )

    assert len(patches) == 1
    assert np.any(patches[0].mask > 0)


def test_empty_target_produces_no_candidate_patches() -> None:
    volume = np.ones((8, 8, 8), dtype=np.float32)
    target = np.zeros_like(volume, dtype=np.uint8)

    assert extract_centered_patches(volume, _component_centers(target), 4) == []


def test_extract_centered_patches_centers_on_each_voxel() -> None:
    # The shared entry point centers a size^3 patch on each requested voxel; the
    # target voxel lands at the patch center (offset half from the corner).
    volume = np.zeros((8, 8, 8), dtype=np.float32)
    volume[4, 5, 6] = 1

    patches = extract_centered_patches(volume, [(4, 5, 6)], 4)

    assert len(patches) == 1
    assert patches[0].shape == (4, 4, 4)
    assert patches[0][2, 2, 2] == 1


def test_extract_centered_patches_pads_edge_candidates() -> None:
    # A center at the volume corner still yields a full-size patch; the
    # out-of-bounds region reads as zeros.
    volume = np.ones((8, 8, 8), dtype=np.float32)

    patch = extract_centered_patches(volume, [(0, 0, 0)], 4)[0]

    assert patch.shape == (4, 4, 4)
    assert patch[2, 2, 2] == 1  # the corner voxel, at the patch center
    assert patch[0, 0, 0] == 0  # padded out-of-bounds corner


def test_target_centered_patcher_loads_its_detector_lazily_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from microbleednet.orchestration import patching

    detector = object()
    load_calls = []
    monkeypatch.setattr(patching, "CandidateDetector", lambda _architecture: detector)
    monkeypatch.setattr(
        patching.core_io,
        "load_model_weights",
        lambda model, device, checkpoint_path: load_calls.append(
            (model, device, checkpoint_path)
        ),
    )
    monkeypatch.setattr(
        patching.core_processor, "infer", lambda *_args: np.empty(0)
    )
    monkeypatch.setattr(
        patching.core_utils,
        "microbleed_probability",
        lambda _logits: np.ones((1, 4, 4, 4), dtype=np.float32),
    )

    patcher = patching.TargetCenteredPatcher(
        2,
        architecture=ModelArchitecture(initial_channels=1, output_classes=2),
        checkpoint_path=tmp_path / "detector.pth",
        device=torch.device("cpu"),
        threshold=0.5,
    )
    volume = np.zeros((4, 4, 4), dtype=np.float32)
    mask = np.zeros_like(volume, dtype=np.uint8)

    assert patcher.detector is None
    patcher._extract(volume, mask)
    patcher._extract(volume, mask)

    assert patcher.detector is detector
    assert load_calls == [(detector, torch.device("cpu"), tmp_path / "detector.pth")]


def _extracted_patches(count: int) -> list[ExtractedPatch]:
    return [
        ExtractedPatch(
            volume=np.full((4, 4, 4), index, dtype=np.float32),
            mask=np.zeros((4, 4, 4), dtype=np.uint8),
        )
        for index in range(count)
    ]


def test_materialize_inflates_records_without_duplicating_files(
    tmp_path: Path,
) -> None:
    # augmentation_factor inflates the training set by referencing each stored
    # patch factor times, so the record list grows but the file count does not.
    patches = _extracted_patches(3)

    records = materialize_patches(patches, tmp_path, "subject", augmentation_factor=10)

    assert len(records) == 30
    # A subject's patches are stacked into exactly two .npy files (volume, mask).
    assert len(list(tmp_path.glob("*.npy"))) == 2
    # The 30 records index into those two files; each stored patch is referenced
    # 10 times, so there are still only 3 distinct patch indices.
    assert len({record.patch_index for record in records}) == 3


def test_materialize_records_point_at_stored_patches(tmp_path: Path) -> None:
    # Each record's (path, index) must resolve to the patch it represents.
    records = materialize_patches(_extracted_patches(3), tmp_path, "subject")

    assert len(records) == 3
    for expected_index, record in enumerate(records):
        assert record.patch_index == expected_index
        volume = np.load(record.volume_path, mmap_mode="r")[record.patch_index]
        # _extracted_patches fills patch i with the constant value i.
        assert np.all(volume == expected_index)


def test_materialize_empty_writes_nothing(tmp_path: Path) -> None:
    assert materialize_patches([], tmp_path, "subject") == []
    assert list(tmp_path.glob("*.npy")) == []
