import numpy as np
from skimage.measure import label
from skimage.filters import frangi
from scipy.ndimage import convolve
from sklearn.cluster import KMeans
from skimage.measure import regionprops
from skimage.feature import structure_tensor
from skimage.feature import structure_tensor_eigenvalues

from joblib import Parallel, delayed

from microbleednet.constants import FRANGI_FILTER_SIGMAS
from microbleednet.constants import FRANGI_FILTER_ALPHA
from microbleednet.constants import FRANGI_FILTER_BETA
from microbleednet.constants import FRANGI_FILTER_BLACK_RIDGES
from microbleednet.constants import VESSEL_INPAINTING_CLUSTERER_N_CLUSTERS
from microbleednet.constants import VESSEL_INPAINTING_CLUSTERER_RANDOM_STATE
from microbleednet.constants import VESSEL_INPAINTING_MINIMUM_VESSEL_ECCENTRICITY
from microbleednet.constants import VESSEL_INPAINTING_MAXIMUM_VESSEL_SOLIDITY


def apply(volume: np.ndarray) -> np.ndarray:
    vessel_mask = get_volume_vessel_mask(volume)
    inpainted_volume = inpaint_with_neighborhood_mean(volume, vessel_mask)
    return inpainted_volume


def get_volume_vessel_mask(volume: np.ndarray) -> np.ndarray:
    _, _, depth = volume.shape
    vessel_mask = np.zeros_like(volume)

    parallel_generator = Parallel(n_jobs=-1, return_as="generator")(
        delayed(get_slice_vessel_mask)(volume[:, :, slice_idx])
        for slice_idx in range(depth)
    )

    for slice_idx, vessel_mask_slice in enumerate(parallel_generator):
        vessel_mask[:, :, slice_idx] = vessel_mask_slice

    return vessel_mask


def get_slice_vessel_mask(slice: np.ndarray) -> None:

    brain_mask = (slice > 0).astype(int)

    if np.min(slice) == np.max(slice):
        # slice is empty, so skip but still advance the progress bar
        return np.zeros_like(slice)

    frangi_slice = frangi(
        slice,
        sigmas=FRANGI_FILTER_SIGMAS,
        alpha=FRANGI_FILTER_ALPHA,
        beta=FRANGI_FILTER_BETA,
        black_ridges=FRANGI_FILTER_BLACK_RIDGES,
    )
    frangi_slice = frangi_slice * brain_mask

    linearity = get_linearity_measure(slice)
    linearity = linearity * brain_mask
    linearity -= np.min(linearity)
    if np.max(linearity) > 0:
        linearity = linearity / np.max(linearity)

    slice_features = np.stack([frangi_slice.ravel(), linearity.ravel()], axis=1)
    clusterer = KMeans(
        n_clusters=VESSEL_INPAINTING_CLUSTERER_N_CLUSTERS,
        random_state=VESSEL_INPAINTING_CLUSTERER_RANDOM_STATE,
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
            prop.eccentricity < VESSEL_INPAINTING_MINIMUM_VESSEL_ECCENTRICITY
            and prop.solidity > VESSEL_INPAINTING_MAXIMUM_VESSEL_SOLIDITY
        )
    ]
    
    vessel_mask = np.isin(vessel_mask, valid_vessel_regions).astype(int)
    return vessel_mask

def get_linearity_measure(slice: np.ndarray) -> np.ndarray:

    Ixx, Ixy, Iyy = structure_tensor(slice)
    eigenvalues = structure_tensor_eigenvalues((Ixx, Ixy, Iyy))

    lambda1 = eigenvalues[0]
    lambda2 = eigenvalues[1]
    linearity_measure = np.absolute((lambda1 - lambda2) / 2)

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
        inpainted_volume[update_mask] = (
            neighbour_sum[update_mask] / neighbour_count[update_mask]
        )

        working_mask[update_mask] = False
        resolved_voxels = previous_mask_sum - working_mask.sum()

        # Infinite loop safety: Break if the mask size didn't shrink
        if resolved_voxels == 0:
            print(f"[yellow]Warning: Inpainting stopped early. {working_mask.sum()} voxels remain completely isolated.")
            break

    return inpainted_volume