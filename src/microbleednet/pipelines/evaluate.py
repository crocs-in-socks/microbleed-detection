import csv
import io
import logging
from pathlib import Path
from typing import Any, Iterable

from .. import storage
from ..config import EvaluationSubject
from ..core import utils
from ..core.evaluation.froc import sweep_thresholds, write_froc
from ..core.evaluation.matching import match_components
from ..core.evaluation.metrics import aggregate_metrics

logger = logging.getLogger(__name__)


def _write_froc(
    subjects: list[EvaluationSubject],
    references: list[Any],
    thresholds: tuple[float, ...],
    output_dir: Path,
) -> None:
    """Sweep FROC thresholds over each subject's probability map and write it.

    At each threshold every subject's probability map is binarized and matched
    against its reference, so the sweep reuses the same component matching as the
    single-threshold metrics.
    """
    probabilities = [
        utils.nifti_to_numpy(utils.load_volume(subject.probability_path))
        for subject in subjects
        if subject.probability_path is not None
    ]

    def evaluate_at_threshold(threshold: float) -> list[dict[str, Any]]:
        return [
            match_components(probability > threshold, reference > 0)
            for probability, reference in zip(probabilities, references)
        ]

    points = sweep_thresholds(thresholds, evaluate_at_threshold)
    write_froc(points, output_dir / "froc.json", output_dir / "froc.csv")


def evaluate_subjects(
    subjects: Iterable[EvaluationSubject],
    output_dir: Path,
    metadata: dict[str, Any] | None = None,
    froc_thresholds: tuple[float, ...] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    subjects = list(subjects)
    subject_results = []
    references = []
    for subject in subjects:
        prediction = utils.nifti_to_numpy(utils.load_volume(subject.prediction_path))
        reference = utils.nifti_to_numpy(utils.load_volume(subject.reference_path))
        if prediction.shape != reference.shape:
            raise ValueError(f"prediction and reference shapes differ for {subject.subject_id}")
        result = match_components(prediction > 0, reference > 0)
        result["subject_id"] = subject.subject_id
        subject_results.append(result)
        references.append(reference)

    aggregate = aggregate_metrics(subject_results)
    report = {"metadata": metadata or {}, "subjects": subject_results, "aggregate": aggregate}

    subject_fieldnames = ["subject_id", "true_positives", "false_negatives", "false_positives", "prediction_count", "reference_count"]
    subject_buffer = io.StringIO()
    subject_writer = csv.DictWriter(subject_buffer, fieldnames=subject_fieldnames)
    subject_writer.writeheader()
    subject_writer.writerows({key: result[key] for key in subject_fieldnames} for result in subject_results)

    aggregate_buffer = io.StringIO()
    aggregate_writer = csv.DictWriter(aggregate_buffer, fieldnames=list(aggregate))
    aggregate_writer.writeheader()
    aggregate_writer.writerow(aggregate)

    # Publish data first, then the report that advertises completion last.
    storage.write_json_atomic(output_dir / "evaluation.json", report)
    storage.write_text_atomic(output_dir / "subject_metrics.csv", subject_buffer.getvalue())
    storage.write_text_atomic(output_dir / "aggregate_metrics.csv", aggregate_buffer.getvalue())

    if froc_thresholds is not None:
        _write_froc(subjects, references, froc_thresholds, output_dir)

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
