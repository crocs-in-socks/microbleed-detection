import numpy as np

from microbleednet.core.evaluation.froc import sweep_thresholds
from microbleednet.core.evaluation.matching import match_components
from microbleednet.core.evaluation.metrics import aggregate_metrics


def _mask() -> np.ndarray:
    return np.zeros((10, 10, 10), dtype=np.uint8)


def test_exact_and_partial_overlap_match_once() -> None:
    prediction = _mask()
    reference = _mask()
    prediction[2:4, 2:4, 2:4] = 1
    reference[3:5, 3:5, 3:5] = 1
    result = match_components(prediction, reference)
    assert result["true_positives"] == 1
    assert result["matched"][0]["overlap_voxels"] == 1


def test_fragmented_prediction_and_merged_reference_are_counted_auditably() -> None:
    prediction = _mask()
    reference = _mask()
    prediction[2:3, 2:3, 2:3] = 1
    prediction[2:3, 4:5, 2:3] = 1
    reference[2:3, 2:5, 2:3] = 1
    fragmented = match_components(prediction, reference)
    assert (fragmented["true_positives"], fragmented["false_positives"]) == (1, 1)

    prediction = _mask()
    reference = _mask()
    prediction[2:3, 2:5, 2:3] = 1
    reference[2:3, 2:3, 2:3] = 1
    reference[2:3, 4:5, 2:3] = 1
    merged = match_components(prediction, reference)
    assert (merged["true_positives"], merged["false_negatives"]) == (1, 1)


def test_empty_cases_and_cohort_metrics_use_all_subjects() -> None:
    empty = _mask()
    result = match_components(empty, empty)
    assert result["true_positives"] == result["false_positives"] == result["false_negatives"] == 0
    metrics = aggregate_metrics([
        {"true_positives": 1, "false_negatives": 1, "false_positives": 1},
        {"true_positives": 0, "false_negatives": 0, "false_positives": 2},
    ])
    assert metrics["cluster_tpr"] == 0.5
    assert metrics["cluster_precision"] == 1 / 4
    assert metrics["false_positives_per_subject"] == 1.5
    assert aggregate_metrics([])["false_positives_per_subject"] == 0.0


def test_froc_points_are_ordered_by_false_positives_per_subject() -> None:
    def evaluate(threshold: float):
        if threshold == 0.1:
            return [{"true_positives": 1, "false_negatives": 0, "false_positives": 2}]
        return [{"true_positives": 1, "false_negatives": 0, "false_positives": 0}]

    points = sweep_thresholds([0.1, 0.9], evaluate)
    assert [point["threshold"] for point in points] == [0.9, 0.1]
