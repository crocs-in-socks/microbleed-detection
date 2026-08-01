import csv
import json
from pathlib import Path
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
    return sorted(points, key=lambda point: (point["false_positives_per_subject"], point["threshold"]))


def write_froc(points: list[dict[str, Any]], json_path: Path, csv_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(points, indent=2) + "\n", encoding="utf-8")
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
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(points)
