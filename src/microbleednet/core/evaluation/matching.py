from typing import Any, cast

import numpy as np
from scipy.optimize import linear_sum_assignment
from skimage.measure import label


def match_components(
    prediction_mask: np.ndarray, reference_mask: np.ndarray
) -> dict[str, Any]:
    """Match 26-connected prediction and reference components by voxel overlap."""
    if prediction_mask.shape != reference_mask.shape or prediction_mask.ndim != 3:
        raise ValueError(
            "prediction_mask and reference_mask must be matching 3D arrays"
        )

    prediction_labels = cast(np.ndarray, label(prediction_mask > 0, connectivity=3))
    reference_labels = cast(np.ndarray, label(reference_mask > 0, connectivity=3))
    prediction_count = int(prediction_labels.max())
    reference_count = int(reference_labels.max())
    overlap = np.zeros((prediction_count, reference_count), dtype=np.int64)
    if prediction_count and reference_count:
        overlapping_voxels = (prediction_labels > 0) & (reference_labels > 0)
        np.add.at(
            overlap,
            (
                prediction_labels[overlapping_voxels] - 1,
                reference_labels[overlapping_voxels] - 1,
            ),
            1,
        )
        rows, columns = linear_sum_assignment(overlap, maximize=True)
    else:
        rows, columns = np.array([], dtype=int), np.array([], dtype=int)

    matched = []
    for row, column in zip(rows, columns):
        count = int(overlap[row, column])
        if count:
            matched.append(
                {
                    "prediction_id": int(row + 1),
                    "reference_id": int(column + 1),
                    "overlap_voxels": count,
                }
            )
    matched_prediction_ids = {item["prediction_id"] for item in matched}
    matched_reference_ids = {item["reference_id"] for item in matched}
    return {
        "matched": matched,
        "true_positives": len(matched),
        "false_negatives": reference_count - len(matched_reference_ids),
        "false_positives": prediction_count - len(matched_prediction_ids),
        "prediction_count": prediction_count,
        "reference_count": reference_count,
    }
