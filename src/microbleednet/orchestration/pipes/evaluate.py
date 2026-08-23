import csv
import io
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch

from ...core import io as core_io
from ...core.common.models import CandidateDetector, CandidateDiscriminatorStudent
from ...core.evaluation.froc import sweep_thresholds
from ...core.evaluation.matching import match_components
from ...core.evaluation.metrics import aggregate_metrics
from .. import atomic_io, provenance
from ..configs import EvaluateConfig
from ..layouts import DatasetLayout, ExperimentLayout
from ..manifests import RawDatasetManifest, read_manifest
from . import infer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _ScoredSubject:
    """One subject's source-space prediction paired with its reference mask.

    Assembled by the inference loop and consumed by the scoring pass; it is the
    internal counterpart to the paths a caller used to supply by hand.
    """

    subject_id: str
    prediction_path: Path
    reference_path: Path
    probability_path: Path


def _resolve_checkpoints(config: EvaluateConfig) -> tuple[Path, Path]:
    """Resolve the detector and student checkpoints for evaluation.

    Explicit checkpoint paths win; otherwise the best checkpoints are resolved
    from ``experiment_dir`` exactly as ``train`` writes them. The config
    validator guarantees one of the two is fully specified.
    """
    if config.detector_checkpoint is not None and config.student_checkpoint is not None:
        return config.detector_checkpoint, config.student_checkpoint

    assert config.experiment_dir is not None  # guaranteed by EvaluateConfig
    layout = ExperimentLayout()
    detector = config.experiment_dir / layout.best_checkpoint("detector")
    student = config.experiment_dir / layout.best_checkpoint("student")
    return detector, student


def _predict_dataset(
    config: EvaluateConfig, device: torch.device
) -> list[_ScoredSubject]:
    """Run inference over every indexed subject that carries a reference mask.

    The models are loaded once and reused across the dataset. Predictions land
    under ``output_dir/predictions/<subject_id>`` and are paired with the raw
    manifest's source-space mask as the reference.
    """
    layout = DatasetLayout()
    manifest = read_manifest(
        config.dataset_dir / layout.raw_manifest, RawDatasetManifest
    )
    detector_checkpoint, student_checkpoint = _resolve_checkpoints(config)
    detector = CandidateDetector(config.detector.architecture)
    student = CandidateDiscriminatorStudent(config.student.architecture)
    core_io.load_model_weights(detector, device, detector_checkpoint)
    core_io.load_model_weights(student, device, student_checkpoint)
    detector = detector.to(device).eval()
    student = student.to(device).eval()

    scored: list[_ScoredSubject] = []
    skipped: list[str] = []
    predictions_dir = config.output_dir / "predictions"
    for subject in manifest.subjects:
        if subject.mask_path is None:
            skipped.append(subject.subject_id)
            continue
        summary = infer.predict_and_write(
            detector,
            student,
            volume_path=Path(subject.volume_path),
            output_dir=predictions_dir / subject.subject_id,
            detector_config=config.detector,
            student_config=config.student,
            preprocessing=config.preprocessing,
            device=device,
            patch_batch_size=config.patch_batch_size,
            subject_id=subject.subject_id,
            postprocessing=config.postprocessing,
        )
        scored.append(
            _ScoredSubject(
                subject_id=subject.subject_id,
                prediction_path=summary.mask_path,
                reference_path=Path(subject.mask_path),
                probability_path=summary.probability_path,
            )
        )

    if skipped:
        logger.warning(
            "skipped %d subjects without a reference mask: %s",
            len(skipped),
            skipped,
        )
    if not scored:
        raise ValueError(
            "no subjects with reference masks to evaluate in "
            f"{config.dataset_dir / layout.raw_manifest}"
        )
    return scored


def _write_froc(
    subjects: list[_ScoredSubject],
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
        core_io.nifti_to_numpy(core_io.load_volume(subject.probability_path))
        for subject in subjects
    ]

    def evaluate_at_threshold(threshold: float) -> list[dict[str, Any]]:
        return [
            match_components(probability > threshold, reference > 0)
            for probability, reference in zip(probabilities, references)
        ]

    points = sweep_thresholds(thresholds, evaluate_at_threshold)
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
    atomic_io.write_json_atomic(output_dir / "froc.json", points)
    atomic_io.write_text_atomic(output_dir / "froc.csv", buffer.getvalue())


def _score_subjects(
    subjects: Iterable[_ScoredSubject],
    output_dir: Path,
    metadata: dict[str, Any] | None = None,
    froc_thresholds: tuple[float, ...] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    subjects = list(subjects)
    subject_results = []
    references = []
    for subject in subjects:
        prediction = core_io.nifti_to_numpy(
            core_io.load_volume(subject.prediction_path)
        )
        reference = core_io.nifti_to_numpy(
            core_io.load_volume(subject.reference_path)
        )
        if prediction.shape != reference.shape:
            raise ValueError(
                f"prediction and reference shapes differ for {subject.subject_id}"
            )
        result = match_components(prediction > 0, reference > 0)
        result["subject_id"] = subject.subject_id
        subject_results.append(result)
        references.append(reference)

    aggregate = aggregate_metrics(subject_results)
    report = {
        "metadata": metadata or {},
        "subjects": subject_results,
        "aggregate": aggregate,
    }

    subject_fieldnames = [
        "subject_id",
        "true_positives",
        "false_negatives",
        "false_positives",
        "prediction_count",
        "reference_count",
    ]
    subject_buffer = io.StringIO()
    subject_writer = csv.DictWriter(subject_buffer, fieldnames=subject_fieldnames)
    subject_writer.writeheader()
    subject_writer.writerows(
        {key: result[key] for key in subject_fieldnames} for result in subject_results
    )

    aggregate_buffer = io.StringIO()
    aggregate_writer = csv.DictWriter(aggregate_buffer, fieldnames=list(aggregate))
    aggregate_writer.writeheader()
    aggregate_writer.writerow(aggregate)

    # Publish data first, then the report that advertises completion last.
    atomic_io.write_json_atomic(output_dir / "evaluation.json", report)
    atomic_io.write_text_atomic(
        output_dir / "subject_metrics.csv", subject_buffer.getvalue()
    )
    atomic_io.write_text_atomic(
        output_dir / "aggregate_metrics.csv", aggregate_buffer.getvalue()
    )

    if froc_thresholds is not None:
        _write_froc(subjects, references, froc_thresholds, output_dir)

    try:
        import matplotlib.pyplot as plt  # pyright: ignore[reportMissingImports]
    except ImportError:
        plt = None
        logger.info(
            "matplotlib not installed; skipping the counts plot. "
            "Install the 'plots' extra to enable it."
        )
    if plt is not None:
        figure, axis = plt.subplots()
        axis.bar(
            ["TP", "FN", "FP"],
            [
                aggregate["true_positives"],
                aggregate["false_negatives"],
                aggregate["false_positives"],
            ],
        )
        axis.set_title("Component evaluation counts")
        figure.savefig(output_dir / "component_counts.png", bbox_inches="tight")
        plt.close(figure)
    return report


def execute(config: EvaluateConfig) -> dict[str, Any]:
    device = torch.device(config.device)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Seed and record provenance before any prediction is written, so every
    # evaluation directory can be traced back to its config, seed, and revision.
    provenance.seed_everything(config.seed)
    provenance.write_provenance(
        config.output_dir, config, seed=config.seed, device=device
    )

    scored = _predict_dataset(config, device)
    return _score_subjects(
        scored,
        config.output_dir,
        config.metadata,
        config.froc_thresholds,
    )
