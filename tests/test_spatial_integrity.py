import nibabel as nib
import numpy as np
import pytest
from nibabel.affines import apply_affine

from microbleednet.core.engines import processor
from microbleednet.core.engines.processor import preprocess, restore_to_source

# Reorient only: the other stages need external tools or would alter geometry in
# ways these spatial round-trip tests do not exercise.
_TEST_MODALITY = "QSM"


@pytest.fixture
def fixed_preprocessing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(processor.volume_ops, "extract_brain", lambda volume: volume)
    monkeypatch.setattr(
        processor.volume_ops, "bias_field_correct_n4", lambda volume: volume
    )
    monkeypatch.setattr(processor.inpaint_vessels, "apply", lambda volume: volume)


def test_preprocess_round_trip_preserves_source_geometry(
    fixed_preprocessing: None,
) -> None:
    affine = np.array(
        [[0, -2, 0, 20], [1, 0, 0, 30], [0, 0, 3, 40], [0, 0, 0, 1]],
        dtype=float,
    )
    image = np.zeros((5, 6, 4), dtype=np.float32)
    image[1:4, 2:5, 1:3] = 1
    image[2, 3, 1] = 2
    mask = np.zeros_like(image, dtype=np.uint8)
    mask[2, 3, 1] = 1

    result = preprocess(
        nib.Nifti1Image(image, affine),
        nib.Nifti1Image(mask, affine),
            _TEST_MODALITY,
    )
    assert result.mask is not None
    restored_image = restore_to_source(result.image, result.transform)
    restored_mask = restore_to_source(result.mask, result.transform)

    assert restored_image.shape == image.shape
    assert restored_mask.shape == mask.shape
    assert restored_image.affine is not None
    assert restored_mask.affine is not None
    assert np.allclose(restored_image.affine, affine)
    assert np.allclose(restored_mask.affine, affine)
    assert np.array_equal(restored_mask.get_fdata(), mask)

    source_world = apply_affine(affine, (2, 3, 1))
    restored_index = np.unravel_index(
        np.argmax(restored_image.get_fdata()), restored_image.shape
    )
    restored_world = apply_affine(restored_image.affine, restored_index)
    assert np.allclose(restored_world, source_world)


def test_preprocess_rejects_empty_input(fixed_preprocessing: None) -> None:
    image = np.zeros((4, 4, 4), dtype=np.float32)
    image_nii = nib.Nifti1Image(image, np.eye(4))

    with pytest.raises(ValueError, match="empty"):
        preprocess(image_nii, None, _TEST_MODALITY)


@pytest.mark.parametrize("modality, expected_calls", [("QSM", 0), ("T2*-GRE", 1)])
def test_bias_correction_matches_paper_modality_branch(
    monkeypatch: pytest.MonkeyPatch,
    modality: str,
    expected_calls: int,
) -> None:
    calls = 0

    monkeypatch.setattr(processor.volume_ops, "extract_brain", lambda volume: volume)
    monkeypatch.setattr(processor.inpaint_vessels, "apply", lambda volume: volume)

    def bias_correct(volume: nib.Nifti1Image) -> nib.Nifti1Image:
        nonlocal calls
        calls += 1
        return volume

    monkeypatch.setattr(processor.volume_ops, "bias_field_correct_n4", bias_correct)
    image = nib.Nifti1Image(
        np.arange(64, dtype=np.float32).reshape(4, 4, 4) + 1.0,
        np.eye(4),
    )

    preprocess(image, None, modality)

    assert calls == expected_calls


def test_preprocess_rejects_mismatched_mask_affine(
    fixed_preprocessing: None,
) -> None:
    image = np.ones((4, 4, 4), dtype=np.float32)
    mask = np.ones_like(image, dtype=np.uint8)
    image_nii = nib.Nifti1Image(image, np.eye(4))
    mask_nii = nib.Nifti1Image(mask, np.diag([1, 1, 1, 2]))

    with pytest.raises(ValueError, match="affines"):
        preprocess(image_nii, mask_nii, _TEST_MODALITY)
