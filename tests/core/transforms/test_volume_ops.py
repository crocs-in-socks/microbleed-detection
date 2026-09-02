import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from microbleednet.core.transforms import volume_ops


def test_normalize_volume_scales_by_positive_maximum() -> None:
    normalized = volume_ops.normalize_volume(np.array([[[0.0, 2.0, 4.0]]]))

    np.testing.assert_array_equal(normalized, np.array([[[0.0, 0.5, 1.0]]]))


@pytest.mark.parametrize("maximum", [0.0, np.nan])
def test_normalize_volume_rejects_invalid_maximum(maximum: float) -> None:
    with pytest.raises(ValueError, match="maximum is not positive and finite"):
        volume_ops.normalize_volume(np.array([[[maximum]]]))


def test_invert_volume_preserves_zero_background() -> None:
    volume = np.array([[[0.0, 1.0, 3.0]]])

    inverted = volume_ops.invert_volume(volume)

    np.testing.assert_array_equal(inverted, np.array([[[0.0, 2.0, 0.0]]]))


def test_tight_crop_volume_returns_positive_extent() -> None:
    volume = np.zeros((4, 5, 6))
    volume[1:3, 2:4, 3:5] = 1

    cropped, bounding_box = volume_ops.tight_crop_volume(volume)

    assert bounding_box == ((1, 3), (2, 4), (3, 5))
    np.testing.assert_array_equal(cropped, np.ones((2, 2, 2)))


def test_tight_crop_volume_rejects_empty_volume() -> None:
    with pytest.raises(ValueError, match="cannot crop an empty volume"):
        volume_ops.tight_crop_volume(np.zeros((2, 2, 2)))


def test_reorient_to_canonical_flips_negative_axis() -> None:
    volume = nib.Nifti1Image(
        np.arange(8, dtype=float).reshape((2, 2, 2)), np.diag([-1, 1, 1, 1])
    )

    canonical = volume_ops.reorient_to_canonical(volume)

    assert nib.aff2axcodes(canonical.affine) == ("R", "A", "S")


def test_adjust_affine_for_crop_translates_in_voxel_coordinates() -> None:
    affine = np.diag([2.0, 3.0, 4.0, 1.0])

    adjusted = volume_ops.adjust_affine_for_crop(affine, (1, 2, 3))

    np.testing.assert_array_equal(adjusted[:3, 3], np.array([2.0, 6.0, 12.0]))


def test_extract_brain_requires_valid_fsldir(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("FSLDIR", str(tmp_path / "missing"))
    volume = nib.Nifti1Image(np.ones((2, 2, 2)), np.eye(4))

    with pytest.raises(EnvironmentError, match="Valid FSLDIR"):
        volume_ops.extract_brain(volume)


def test_extract_brain_runs_bet_and_materializes_output(
    tmp_path: Path, monkeypatch
) -> None:
    fsldir = tmp_path / "fsl"
    (fsldir / "bin").mkdir(parents=True)
    monkeypatch.setenv("FSLDIR", str(fsldir))
    calls = []

    def fake_run(command: list[str], check: bool) -> None:
        calls.append((command, check))
        shutil.copyfile(command[1], command[2])

    monkeypatch.setattr(volume_ops.subprocess, "run", fake_run)
    volume = nib.Nifti1Image(
        np.arange(8, dtype=float).reshape((2, 2, 2)), np.eye(4)
    )

    extracted = volume_ops.extract_brain(volume)

    assert calls[0][0][0] == str(fsldir / "bin" / "bet")
    assert calls[0][1] is True
    np.testing.assert_array_equal(extracted.get_fdata(), volume.get_fdata())


def test_bias_field_correct_n4_preserves_nifti_geometry(monkeypatch) -> None:
    class FakeCorrector:
        def Execute(self, volume: np.ndarray, mask: np.ndarray) -> np.ndarray:
            np.testing.assert_array_equal(mask, volume > 0)
            return volume + 1

    monkeypatch.setattr(volume_ops.sitk, "GetImageFromArray", lambda array: array)
    monkeypatch.setattr(
        volume_ops.sitk, "N4BiasFieldCorrectionImageFilter", FakeCorrector
    )
    monkeypatch.setattr(volume_ops.sitk, "GetArrayFromImage", lambda array: array)
    affine = np.diag([2.0, 3.0, 4.0, 1.0])
    volume = nib.Nifti1Image(np.ones((2, 2, 2)), affine)

    corrected = volume_ops.bias_field_correct_n4(volume)

    np.testing.assert_array_equal(corrected.get_fdata(), np.full((2, 2, 2), 2.0))
    np.testing.assert_array_equal(corrected.affine, affine)