import numpy as np
from scipy.ndimage import gaussian_filter

from microbleednet.core.datamodels import AugmentationConfig


def translate_array(array: np.ndarray, offset_x: int, offset_y: int) -> np.ndarray:
    """Shift the first two axes without wrapping values across boundaries."""
    shifted = np.zeros_like(array)
    source_x_start = max(0, -offset_x)
    source_x_stop = min(array.shape[0], array.shape[0] - offset_x)
    source_y_start = max(0, -offset_y)
    source_y_stop = min(array.shape[1], array.shape[1] - offset_y)
    target_x_start = max(0, offset_x)
    target_y_start = max(0, offset_y)
    shifted[
        target_x_start : target_x_start + source_x_stop - source_x_start,
        target_y_start : target_y_start + source_y_stop - source_y_start,
        ...,
    ] = array[source_x_start:source_x_stop, source_y_start:source_y_stop, ...]
    return shifted


def translate(
    *volumes,
    augmentation: AugmentationConfig,
    rng: np.random.Generator | None = None,
    **kwargs,
):
    """
    Translation: x-offset: [-15, 15], y-offset: [-15, 15] voxels
    Applied to ALL provided volumes equally.
    (kwargs swallows 'intensity_indices' passed by the main loop)
    """
    rng = rng or np.random.default_rng(0)
    low, high = augmentation.translation_offset_range
    offset_x = int(rng.integers(low, high + 1))
    offset_y = int(rng.integers(low, high + 1))

    translated_volumes = tuple(
        translate_array(vol, offset_x, offset_y) for vol in volumes
    )
    translated_volumes = tuple(
        (translated > 0).astype(volume.dtype, copy=False)
        if index in kwargs.get("mask_indices", ())
        else translated
        for index, (translated, volume) in enumerate(zip(translated_volumes, volumes))
    )

    return translated_volumes


def add_noise(
    *volumes,
    augmentation: AugmentationConfig,
    intensity_indices=(0,),
    rng: np.random.Generator | None = None,
    **kwargs,
):
    """
    Random noise injection: Distribution - Gaussian, mu = 0, sigma^2 = [0.01, 0.04]
    Applied ONLY to the volumes specified by intensity_indices.
    """
    rng = rng or np.random.default_rng(0)
    variance = rng.uniform(*augmentation.noise_variance_range)

    result = list(volumes)

    for idx in intensity_indices:
        result[idx] = result[idx] + rng.normal(0, np.sqrt(variance), result[idx].shape)

    return tuple(result)


def blur(
    *volumes,
    augmentation: AugmentationConfig,
    intensity_indices=(0,),
    rng: np.random.Generator | None = None,
    **kwargs,
):
    """
    Gaussian filtering: sigma = [0.1, 0.2] voxels
    Applied ONLY to the volumes specified by intensity_indices.
    """
    rng = rng or np.random.default_rng(0)
    sigma = rng.uniform(*augmentation.blur_sigma_range)

    result = list(volumes)

    for idx in intensity_indices:
        result[idx] = gaussian_filter(result[idx], sigma)

    return tuple(result)


def augment(
    *volumes,
    augmentation: AugmentationConfig | None = None,
    intensity_indices=(0,),
    mask_indices=(),
    rng: np.random.Generator | None = None,
):
    """
    Applies a random combination of transformations to an arbitrary number of volumes.

    Args:
        *volumes: Any number of numpy arrays (e.g., image, label, weights)
        augmentation: Paper-parameterized ranges (translation/noise/blur). Defaults
            to the paper values via AugmentationConfig().
        intensity_indices: Tuple of integers indicating which volumes get
            blur/noise. Defaults to (0,), meaning only the first volume is
            altered.
    """
    augmentation = augmentation or AugmentationConfig()
    available_transformations = {
        "translate": translate,
        "noise": add_noise,
        "blur": blur,
    }

    rng = rng or np.random.default_rng(0)
    transformation_names = list(available_transformations)
    count = int(rng.integers(1, len(transformation_names) + 1))
    transformations = [
        available_transformations[name]
        for name in rng.choice(transformation_names, size=count, replace=False)
    ]

    transformed_volumes = volumes

    for func in transformations:
        transformed_volumes = func(
            *transformed_volumes,
            augmentation=augmentation,
            intensity_indices=intensity_indices,
            mask_indices=mask_indices,
            rng=rng,
        )

    return transformed_volumes
