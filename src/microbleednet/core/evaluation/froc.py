from typing import Any, Callable, Iterable

from .metrics import aggregate_metrics


def sweep_thresholds(
    thresholds: Iterable[float],
    evaluate_at_threshold: Callable[[float], Iterable[dict[str, Any]]],
) -> list[dict[str, Any]]:
    points = []
    for threshold in thresholds:
        subject_matches = list(evaluate_at_threshold(float(threshold)))
        metrics = aggregate_metrics(subject_matches)
        points.append({"threshold": float(threshold), **metrics})
    return sorted(
        points,
        key=lambda point: (point["false_positives_per_subject"], point["threshold"]),
    )
