from typing import Any, cast

import numpy as np
from skimage.measure import label, regionprops


def describe_components(
    mask: np.ndarray,
    source_subject: str = "",
    probability: np.ndarray | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Label 3D components and return their masks and geometric descriptions.

    When ``probability`` is supplied, it must share ``mask``'s shape and each
    component description includes its mean probability.
    """
    if mask.ndim != 3:
        raise ValueError("component mask must be 3D")
    if probability is not None and probability.shape != mask.shape:
        raise ValueError("probability must match component mask shape")
    labeled = cast(np.ndarray, label(mask > 0, connectivity=3))
    components = []
    for region in regionprops(labeled):
        component_mask = labeled == region.label
        component = {
            "component_id": int(region.label),
            "source_subject": source_subject,
            "voxel_count": int(region.area),
            "centroid": [float(value) for value in region.centroid],
            "bounding_box": [
                [int(region.bbox[index]), int(region.bbox[index + 3])]
                for index in range(3)
            ],
            "label_mask": component_mask,
        }
        if probability is not None:
            component["mean_probability"] = float(probability[component_mask].mean())
        components.append(component)
    return labeled, components


def component_centers(
    components: list[dict[str, Any]],
) -> list[tuple[int, int, int]]:
    """Round component centroids to voxel centers for patch extraction."""
    centers: list[tuple[int, int, int]] = []
    for component in components:
        centroid = component["centroid"]
        centers.append(
            (
                int(round(centroid[0])),
                int(round(centroid[1])),
                int(round(centroid[2])),
            )
        )
    return centers
