from typing import Any

import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.measure import label, regionprops

from .components import label_prediction_components


def _projection_eccentricity(component_mask: np.ndarray) -> float:
    projections = [component_mask.max(axis=axis).astype(np.uint8) for axis in range(3)]
    projection = max(projections, key=lambda value: int(value.sum()))
    projected_labels = label(projection, connectivity=2)
    projected_regions = regionprops(projected_labels)
    if not projected_regions:
        return 1.0
    return float(max(region.eccentricity for region in projected_regions))


def filter_components(
    prediction_mask: np.ndarray,
    spacing: tuple[float, float, float],
    brain_mask: np.ndarray,
    minimum_volume_mm3: float = 2.5,
    maximum_eccentricity: float = 0.2,
    minimum_boundary_distance_voxels: float = 0.0,
    source_subject: str = "",
) -> list[dict[str, Any]]:
    if prediction_mask.shape != brain_mask.shape or prediction_mask.ndim != 3:
        raise ValueError("prediction_mask and brain_mask must be matching 3D arrays")
    if len(spacing) != 3 or any(value <= 0 for value in spacing):
        raise ValueError("spacing must contain three positive values")
    if minimum_volume_mm3 < 0 or minimum_boundary_distance_voxels < 0:
        raise ValueError("filter thresholds must be non-negative")

    components = label_prediction_components(prediction_mask, source_subject)
    brain = brain_mask > 0
    distance = distance_transform_edt(brain)
    assert distance is not None
    voxel_volume = float(np.prod(spacing))
    results = []
    for component in components:
        mask = component.pop("label_mask")
        volume_mm3 = component["voxel_count"] * voxel_volume
        eccentricity = _projection_eccentricity(mask)
        boundary_distance = float(distance[mask].min()) if np.any(mask) else 0.0
        reasons = []
        if volume_mm3 < minimum_volume_mm3:
            reasons.append("volume")
        if eccentricity > maximum_eccentricity:
            reasons.append("eccentricity")
        if boundary_distance < minimum_boundary_distance_voxels:
            reasons.append("boundary_distance")
        component.update(
            {
                "volume_mm3": volume_mm3,
                "eccentricity": eccentricity,
                "boundary_distance_voxels": boundary_distance,
                "accepted": not reasons,
                "rejection_reasons": reasons,
            }
        )
        results.append(component)
    return results
