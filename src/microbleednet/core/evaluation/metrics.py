from typing import Any, Iterable


def _safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def aggregate_metrics(subject_matches: Iterable[dict[str, Any]], subject_count: int | None = None) -> dict[str, Any]:
    matches = list(subject_matches)
    count = len(matches) if subject_count is None else subject_count
    if count < 0:
        raise ValueError("subject_count must be non-negative")
    true_positives = sum(int(item["true_positives"]) for item in matches)
    false_negatives = sum(int(item["false_negatives"]) for item in matches)
    false_positives = sum(int(item["false_positives"]) for item in matches)
    return {
        "true_positives": true_positives,
        "false_negatives": false_negatives,
        "false_positives": false_positives,
        "subject_count": count,
        "cluster_tpr": _safe_ratio(true_positives, true_positives + false_negatives),
        "cluster_precision": _safe_ratio(true_positives, true_positives + false_positives),
        "false_positives_per_subject": _safe_ratio(false_positives, count),
    }
