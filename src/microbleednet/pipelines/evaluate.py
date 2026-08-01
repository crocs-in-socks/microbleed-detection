import csv
import logging
from pathlib import Path
from typing import Any, Iterable

from ..core import utils
from ..core.evaluation.matching import match_components
from ..core.evaluation.metrics import aggregate_metrics

logger = logging.getLogger(__name__)


def evaluate_subjects(
    subjects: Iterable[dict[str, Any]],
    output_dir: Path,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    subject_results = []
    for subject in subjects:
        prediction = utils.nifti_to_numpy(utils.load_volume(Path(subject["prediction_path"])))
        reference = utils.nifti_to_numpy(utils.load_volume(Path(subject["reference_path"])))
        if prediction.shape != reference.shape:
            raise ValueError(f"prediction and reference shapes differ for {subject['subject_id']}")
        result = match_components(prediction > 0, reference > 0)
        result["subject_id"] = subject["subject_id"]
        subject_results.append(result)

    aggregate = aggregate_metrics(subject_results)
    report = {"metadata": metadata or {}, "subjects": subject_results, "aggregate": aggregate}
    utils.write_json_atomic(output_dir / "evaluation.json", report)
    with (output_dir / "subject_metrics.csv").open("w", newline="", encoding="utf-8") as csv_file:
        fieldnames = ["subject_id", "true_positives", "false_negatives", "false_positives", "prediction_count", "reference_count"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: result[key] for key in fieldnames} for result in subject_results)
    with (output_dir / "aggregate_metrics.csv").open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(aggregate))
        writer.writeheader()
        writer.writerow(aggregate)

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        plt = None
        logger.info(
            "matplotlib not installed; skipping the counts plot. "
            "Install the 'plots' extra to enable it."
        )
    if plt is not None:
        figure, axis = plt.subplots()
        axis.bar(["TP", "FN", "FP"], [aggregate["true_positives"], aggregate["false_negatives"], aggregate["false_positives"]])
        axis.set_title("Component evaluation counts")
        figure.savefig(output_dir / "component_counts.png", bbox_inches="tight")
        plt.close(figure)
    return report


def execute(*args, **kwargs) -> dict[str, Any]:
    return evaluate_subjects(*args, **kwargs)
