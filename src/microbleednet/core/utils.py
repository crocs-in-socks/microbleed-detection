import numpy as np
import nibabel as nib


def nifti_to_numpy(volume: nib.Nifti1Image) -> np.ndarray:
    return volume.get_fdata()


def numpy_to_nifti(array: np.ndarray, reference: nib.Nifti1Image) -> nib.Nifti1Image:
    return nib.Nifti1Image(array, reference.affine, reference.header)


def normalize_volume(volume: np.ndarray) -> np.ndarray:
    volume = volume - np.min(volume)
    if np.max(volume) > 0:
        volume = volume / np.max(volume)
    return volume


def invert_volume(volume: np.ndarray) -> np.ndarray:
    brain_mask = (volume > 0).astype(int)
    volume = np.max(volume) - volume
    volume = volume * brain_mask
    return volume


def tight_crop_volume(volume: np.ndarray) -> tuple[np.ndarray, tuple]:

    dim0_sum = np.sum(volume, axis=(1, 2))
    dim1_sum = np.sum(volume, axis=(0, 2))
    dim2_sum = np.sum(volume, axis=(0, 1))

    d0_start, d0_end = find_bounds_1d(dim0_sum)
    d1_start, d1_end = find_bounds_1d(dim1_sum)
    d2_start, d2_end = find_bounds_1d(dim2_sum)

    bounding_box = (
        slice(d0_start, d0_end + 1),
        slice(d1_start, d1_end + 1),
        slice(d2_start, d2_end + 1),
    )

    cropped_volume = volume[bounding_box]

    return cropped_volume, bounding_box


def find_bounds_1d(array: np.ndarray) -> tuple:
    nonzero_indices = np.flatnonzero(array > 0)
    if nonzero_indices.size == 0:
        return 0, 0

    first_index = nonzero_indices[0]
    last_index = nonzero_indices[-1]

    return first_index, last_index
