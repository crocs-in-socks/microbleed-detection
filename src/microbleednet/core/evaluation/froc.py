import csv
import io
from pathlib import Path
from typing import Any, Callable, Iterable

from microbleednet import storage

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


def write_froc(points: list[dict[str, Any]], json_path: Path, csv_path: Path) -> None:
    fieldnames = [
        "threshold",
        "true_positives",
        "false_negatives",
        "false_positives",
        "subject_count",
        "cluster_tpr",
        "cluster_precision",
        "false_positives_per_subject",
    ]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(points)
    # Write the JSON payload first, then the CSV report last: a reader that sees
    # the CSV can rely on the machine-readable points already being present.
    storage.write_json_atomic(json_path, points)
    storage.write_text_atomic(csv_path, buffer.getvalue())
