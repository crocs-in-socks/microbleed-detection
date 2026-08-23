import csv
import io
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from skimage.measure import label

from ...core import io as core_io
from ...core import utils as core_utils
from ...core.common.models import CandidateDetector, CandidateDiscriminatorStudent
from ...core.datamodels import Modality
from ...core.engines import processor
from ...core.postprocessing.components import component_centers, describe_components
from ...core.postprocessing.filters import filter_components
from ...core.transforms import frst
from ...core.transforms.patch import extract_centered_patches
from .. import atomic_io, provenance
from ..configs import DetectorConfig, InferConfig, PostprocessingConfig, StudentConfig
from ..records import PredictionSummary


def _student_probabilities(
    student, patches: list[np.ndarray], device: torch.device, batch_size: int
) -> np.ndarray:
    if not patches:
        return np.empty(0, dtype=np.float32)
    probabilities = []
    with torch.no_grad():
        for start in range(0, len(patches), batch_size):
            batch = (
                torch.from_numpy(np.stack(patches[start : start + batch_size]))
                .float()
                .unsqueeze(1)
                .to(device)
            )
            batch = frst.prepend_frst_channel(batch)
            logits = student(batch)
            probabilities.append(core_utils.microbleed_probability(logits))
    return np.concatenate(probabilities)


def _apply_postprocessing(
    component_mask: np.ndarray,
    probability_map: np.ndarray,
    brain_mask: np.ndarray,
    spacing: tuple[float, float, float],
    postprocessing: PostprocessingConfig,
    subject_id: str,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Morphologically filter the student-accepted mask before it is written.

    Rejected components are removed from both the mask and the probability map so
    the written outputs reflect only accepted lesions. The per-component filter
    verdicts (volume, eccentricity, boundary distance) are returned for the
    record so a reader can audit why any candidate was dropped.
    """
    filtered = filter_components(
        component_mask,
        spacing,
        brain_mask,
        minimum_volume_mm3=postprocessing.minimum_volume_mm3,
        maximum_eccentricity=postprocessing.maximum_eccentricity,
        minimum_boundary_distance_mm=postprocessing.minimum_boundary_distance_mm,
        source_subject=subject_id,
    )
    component_labels = label(component_mask > 0, connectivity=3)
    accepted_mask = np.zeros_like(component_mask)
    for component in filtered:
        if component["accepted"]:
            accepted_mask[component_labels == component["component_id"]] = 1
    accepted_probability = np.where(accepted_mask > 0, probability_map, 0.0).astype(
        probability_map.dtype
    )
    return accepted_mask, accepted_probability, filtered


def predict_and_write(
    detector,
    student,
    volume_path: Path,
    output_dir: Path,
    detector_config: DetectorConfig,
    student_config: StudentConfig,
    preprocessing: Modality,
    device: torch.device = torch.device("cpu"),
    patch_batch_size: int = 8,
    subject_id: str | None = None,
    postprocessing: PostprocessingConfig | None = None,
) -> PredictionSummary:
    """Predict one volume with already-loaded models.

    Evaluation calls this after loading the detector and student once, then
    reuses those models across the whole dataset.
    """
    detector_threshold = detector_config.probability_threshold
    student_threshold = student_config.probability_threshold
    subject_id = subject_id or Path(volume_path).name.split(".")[0]
    image = core_io.load_volume(volume_path)
    processed = processor.preprocess(image, None, preprocessing)

    detector_logits = processor.infer(detector, device, processed.image)
    detector_probability = core_utils.microbleed_probability(detector_logits)[0]
    candidate_mask = detector_probability >= detector_threshold
    candidate_labels, candidate_records = describe_components(
        candidate_mask, subject_id, detector_probability
    )
    for record in candidate_records:
        record["candidate_id"] = record.pop("component_id")
        record["detector_probability"] = record.pop("mean_probability")
    candidate_patch_size = student_config.patch_size
    candidate_ids = [record["candidate_id"] for record in candidate_records]
    patches = extract_centered_patches(
        processed.image, component_centers(candidate_records), candidate_patch_size
    )

    student_probabilities = _student_probabilities(
        student, patches, device, patch_batch_size
    )
    component_mask = np.zeros(processed.image.shape, dtype=np.uint8)
    probability_map = np.zeros(processed.image.shape, dtype=np.float32)
    by_id = {record["candidate_id"]: record for record in candidate_records}
    for candidate_id, student_probability in zip(
        candidate_ids, student_probabilities
    ):
        record = by_id[candidate_id]
        record["student_probability"] = float(student_probability)
        record["accepted"] = bool(student_probability >= student_threshold)
        if record["accepted"]:
            component_mask[candidate_labels == candidate_id] = 1
            probability_map[candidate_labels == candidate_id] = student_probability

    if postprocessing is not None:
        # The extracted-brain support is the brain mask for boundary distance.
        brain_mask = (processed.image != 0).astype(np.uint8)
        component_mask, probability_map, _ = _apply_postprocessing(
            component_mask,
            probability_map,
            brain_mask,
            processed.geometry.spacing,
            postprocessing,
            subject_id,
        )
        # A candidate survives postprocessing iff its voxels remain in the final
        # accepted mask. Recording this per candidate keeps the components file
        # the single audit trail: student-accepted candidates dropped by the
        # morphological filter are flagged here.
        for record in candidate_records:
            candidate_voxels = candidate_labels == record["candidate_id"]
            student_accepted = record.get("accepted", False)
            record["postprocessing_accepted"] = bool(
                student_accepted and np.any(component_mask[candidate_voxels])
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    mask_image = processor.restore_to_source(component_mask, processed.transform)
    probability_image = processor.restore_to_source(
        probability_map, processed.transform
    )
    mask_path = output_dir / f"{subject_id}_prediction.nii.gz"
    probability_path = output_dir / f"{subject_id}_probability.nii.gz"
    core_io.save_volume(mask_image, mask_path)
    core_io.save_volume(probability_image, probability_path)
    record_path = output_dir / f"{subject_id}_components.json"
    csv_path = output_dir / f"{subject_id}_components.csv"
    csv_fieldnames = [
        "candidate_id",
        "detector_probability",
        "student_probability",
        "accepted",
    ]
    if postprocessing is not None:
        csv_fieldnames.append("postprocessing_accepted")
    csv_buffer = io.StringIO()
    csv_writer = csv.DictWriter(csv_buffer, fieldnames=csv_fieldnames)
    csv_writer.writeheader()
    for record in candidate_records:
        csv_writer.writerow({key: record.get(key) for key in csv_fieldnames})
    # Publish the component JSON first, then the CSV report last.
    components_payload = {"subject_id": subject_id, "components": candidate_records}
    atomic_io.write_json_atomic(record_path, components_payload)
    atomic_io.write_text_atomic(csv_path, csv_buffer.getvalue())
    return PredictionSummary(
        subject_id=subject_id,
        mask_path=mask_path,
        probability_path=probability_path,
        components_path=record_path,
        component_count=len(candidate_records),
    )


def execute(config: InferConfig) -> PredictionSummary:
    device = torch.device(config.device)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Seed and record provenance before any prediction is written, so every
    # output directory can be traced back to its config, seed, and revision.
    provenance.seed_everything(config.seed)
    provenance.write_provenance(
        config.output_dir, config, seed=config.seed, device=device
    )

    detector = CandidateDetector(config.detector.architecture)
    student = CandidateDiscriminatorStudent(config.student.architecture)
    core_io.load_model_weights(detector, device, config.detector_checkpoint)
    core_io.load_model_weights(student, device, config.student_checkpoint)
    return predict_and_write(
        detector.to(device).eval(),
        student.to(device).eval(),
        volume_path=config.volume_path,
        output_dir=config.output_dir,
        detector_config=config.detector,
        student_config=config.student,
        preprocessing=config.preprocessing,
        device=device,
        patch_batch_size=config.patch_batch_size,
        subject_id=config.subject_id,
        postprocessing=config.postprocessing,
    )
