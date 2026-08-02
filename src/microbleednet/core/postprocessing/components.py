from typing import Any

import numpy as np
from skimage.measure import label, regionprops


def label_prediction_components(
    mask: np.ndarray, source_subject: str = ""
) -> list[dict[str, Any]]:
    if mask.ndim != 3:
        raise ValueError("prediction mask must be 3D")
    labeled = label(mask > 0, connectivity=3)
    components = []
    for region in regionprops(labeled):
        components.append(
            {
                "component_id": int(region.label),
                "source_subject": source_subject,
                "voxel_count": int(region.area),
                "centroid": [float(value) for value in region.centroid],
                "bounding_box": [
                    [int(region.bbox[index]), int(region.bbox[index + 3])]
                    for index in range(3)
                ],
                "label_mask": labeled == region.label,
            }
        )
    return components
