import numpy as np
from skimage.measure import label
from skimage.filters import frangi
from scipy.ndimage import convolve
from sklearn.cluster import KMeans
from skimage.measure import regionprops
from scipy.ndimage import binary_dilation
from skimage.feature import structure_tensor
from skimage.feature import structure_tensor_eigenvalues

from joblib import delayed
from joblib import Parallel

from .. import utils

# Frangi vesselness filter parameters (see skimage.filters.frangi). Structural
# tuning of the vessel detector; not exposed through the config models.
_FRANGI_SIGMAS = (0.5, 1.2, 0.2)
_FRANGI_ALPHA = 0.9
_FRANGI_BETA = 20
_FRANGI_BLACK_RIDGES = False

# KMeans clustering + region filtering that turn the vesselness map into a mask.
_CLUSTERER_N_CLUSTERS = 2
_CLUSTERER_RANDOM_STATE = 42
_MINIMUM_VESSEL_ECCENTRICITY = 0.9
_MAXIMUM_VESSEL_SOLIDITY = 0.5


def apply(volume: np.ndarray) -> np.ndarray:
    """
    Inpaint vessels in the given volume.
    """
    vessel_mask = get_volume_vessel_mask(volume)
    vessel_mask = binary_dilation(vessel_mask, iterations=1)

    inpainted_volume = inpaint_with_neighborhood_mean(volume, vessel_mask)

    return inpainted_volume


def get_volume_vessel_mask(volume: np.ndarray) -> np.ndarray:
    """Get vessel mask using Frangi filter and KMeans clustering in parallel"""

    _, _, depth = volume.shape
    vessel_mask = np.zeros_like(volume)

    parallel_generator = Parallel(n_jobs=-1, return_as="generator")(
        delayed(get_slice_vessel_mask)(volume[:, :, slice_idx])
        for slice_idx in range(depth)
    )

    for slice_idx, vessel_mask_slice in enumerate(parallel_generator):
        vessel_mask[:, :, slice_idx] = vessel_mask_slice

    return vessel_mask


def get_slice_vessel_mask(image_slice: np.ndarray) -> np.ndarray:

    brain_mask = (image_slice > 0).astype(int)

    if not np.any(brain_mask) or np.min(image_slice) == np.max(image_slice):
        return np.zeros_like(image_slice)

    frangi_slice = frangi(
        image_slice,
        sigmas=_FRANGI_SIGMAS,
        alpha=_FRANGI_ALPHA,
        beta=_FRANGI_BETA,
        black_ridges=_FRANGI_BLACK_RIDGES,
    )
    frangi_slice = frangi_slice * brain_mask

    linearity = get_linearity_measure(image_slice)
    linearity = linearity * brain_mask
    linearity -= np.min(linearity)
    if np.max(linearity) > 0:
        linearity = linearity / np.max(linearity)

    slice_features = np.stack([frangi_slice.ravel(), linearity.ravel()], axis=1)
    clusterer = KMeans(
        n_clusters=_CLUSTERER_N_CLUSTERS,
        random_state=_CLUSTERER_RANDOM_STATE,
    ).fit(slice_features)
    clusters = clusterer.labels_

    # Assuming that the number of pixels in vessels is less than other pixels, we label clusters
    vessel_cluster_label = 1 if (clusters == 1).sum() < (clusters == 0).sum() else 0

    vessel_mask = np.reshape(clusters == vessel_cluster_label, slice.shape)
    vessel_mask = label(vessel_mask)
    vessel_mask_props = regionprops(vessel_mask)

    valid_vessel_regions = [
        prop.label
        for prop in vessel_mask_props
        if not (
            prop.eccentricity < _MINIMUM_VESSEL_ECCENTRICITY
            and prop.solidity > _MAXIMUM_VESSEL_SOLIDITY
        )
    ]
    
    vessel_mask = np.isin(vessel_mask, valid_vessel_regions).astype(int)
    return vessel_mask

def get_linearity_measure(slice: np.ndarray) -> np.ndarray:

    Ixx, Ixy, Iyy = structure_tensor(slice)
    eigenvalues = structure_tensor_eigenvalues((Ixx, Ixy, Iyy))

    lambda1 = eigenvalues[0]
    lambda2 = eigenvalues[1]
    linearity_measure = np.absolute(lambda1 - lambda2) / 2

    return linearity_measure


def inpaint_with_neighborhood_mean(volume: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Inpaint point with mean of 26-connected neighbourhood"""

    inpainted_volume = volume.copy()
    working_mask = mask > 0
    kernel = np.ones((3, 3, 3))
    kernel[1, 1, 1] = 0

    while working_mask.sum() > 0:
        previous_mask_sum = working_mask.sum()
        valid_mask = (1 - working_mask).astype(int)

        neighbour_sum = convolve(inpainted_volume * valid_mask, kernel, mode="constant", cval=0)
        neighbour_count = convolve(valid_mask, kernel, mode="constant", cval=0)

        # Identify voxels that are BOTH currently masked AND have at least one valid neighbor
        update_mask = working_mask & (neighbour_count > 0)
        inpainted_volume[update_mask] = (neighbour_sum[update_mask] / neighbour_count[update_mask])

        working_mask[update_mask] = False
        resolved_voxels = previous_mask_sum - working_mask.sum()

        # Infinite loop safety: Break if the mask size didn't shrink
        if resolved_voxels == 0:
            break

    if np.any(working_mask):
        raise ValueError("vessel mask contains unresolved voxels after inpainting")
    return inpainted_volume