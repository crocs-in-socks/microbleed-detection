from pathlib import Path

import nibabel as nib
import numpy as np

from microbleednet.core import io


def test_numpy_to_nifti_preserves_reference_geometry() -> None:
    reference = nib.Nifti1Image(np.zeros((2, 2, 2)), np.diag([2, 3, 4, 1]))
    array = np.ones((2, 2, 2))

    volume = io.numpy_to_nifti(array, reference)

    np.testing.assert_array_equal(volume.affine, reference.affine)
    assert volume.header.get_data_shape() == array.shape


def test_save_and_load_volume_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "volume.nii.gz"
    volume = nib.Nifti1Image(np.ones((2, 2, 2)), np.eye(4))

    io.save_volume(volume, path)
    loaded = io.load_volume(path)

    np.testing.assert_array_equal(io.nifti_to_numpy(loaded), np.ones((2, 2, 2)))