from unittest.mock import Mock

import nibabel as nib
import numpy as np
import pytest

from microbleednet.core.engines import processor


def _volume(shape: tuple[int, int, int] = (2, 2, 2), affine=None) -> nib.Nifti1Image:
    return nib.Nifti1Image(np.ones(shape), np.eye(4) if affine is None else affine)


def _stub_processing_steps(monkeypatch) -> tuple[Mock, Mock]:
    monkeypatch.setattr(
        processor.volume_ops, "reorient_to_canonical", lambda value: value
    )
    monkeypatch.setattr(processor.volume_ops, "extract_brain", lambda value: value)
    monkeypatch.setattr(processor.volume_ops, "normalize_volume", lambda value: value)
    monkeypatch.setattr(
        processor.volume_ops,
        "tight_crop_volume",
        lambda value: (value, ((0, 2), (0, 2), (0, 2))),
    )
    monkeypatch.setattr(processor.inpaint_vessels, "apply", lambda value: value)
    bias_correct = Mock(side_effect=lambda value: value)
    invert = Mock(side_effect=lambda value: value)
    monkeypatch.setattr(processor.volume_ops, "bias_field_correct_n4", bias_correct)
    monkeypatch.setattr(processor.volume_ops, "invert_volume", invert)
    return bias_correct, invert


def test_preprocess_qsm_skips_contrast_operations(monkeypatch) -> None:
    bias_correct, invert = _stub_processing_steps(monkeypatch)
    mask = nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.uint8), np.eye(4))

    result = processor.preprocess(_volume(), mask, "QSM")

    assert result.mask.dtype == np.uint8
    np.testing.assert_array_equal(result.image, np.ones((2, 2, 2)))
    np.testing.assert_array_equal(result.affine, np.eye(4))
    bias_correct.assert_not_called()
    invert.assert_not_called()


def test_preprocess_swi_processes_and_crops_mask(monkeypatch) -> None:
    bias_correct, invert = _stub_processing_steps(monkeypatch)
    mask = nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.uint8), np.eye(4))

    result = processor.preprocess(_volume(), mask, "SWI")

    assert result.mask is not None
    np.testing.assert_array_equal(result.mask, np.ones((2, 2, 2), dtype=int))
    assert result.mask.dtype == np.uint8
    bias_correct.assert_called_once()
    invert.assert_called_once()


def test_preprocess_rejects_mask_with_different_affine() -> None:
    mask = _volume(affine=np.diag([2.0, 1.0, 1.0, 1.0]))

    with pytest.raises(ValueError, match="affines do not match"):
        processor.preprocess(_volume(), mask, "QSM")


def test_preprocess_rejects_reoriented_mask_with_different_shape() -> None:
    with pytest.raises(ValueError, match="shapes do not match"):
        processor.preprocess(_volume(), _volume((1, 2, 2)), "QSM")