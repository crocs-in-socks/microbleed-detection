import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from skimage.measure import label, regionprops

from .. import provenance
from ..config import (
    DetectorConfig,
    InferCommandConfig,
    PreprocessingConfig,
    StudentConfig,
)
from ..core import utils
from ..core.common.models import CandidateDetector, CandidateDiscriminatorStudent
from ..core.engines import processor
from ..core.transforms import frst
from ..core.transforms.patch import _extract_fixed_patch
from ..records import PredictionSummary


def _load_models(
    detector_config: DetectorConfig,
    student_config: StudentConfig,
    detector_checkpoint: Path,
    student_checkpoint: Path,
    device: torch.device,
):
    detector = CandidateDetector(
        input_channels=detector_config.input_channels,
        output_classes=detector_config.output_classes,
        initial_channels=detector_config.initial_channels,
    )
    student = CandidateDiscriminatorStudent(
        input_channels=student_config.input_channels,
        output_classes=student_config.output_classes,
        initial_channels=student_config.initial_channels,
        dropout_rate=student_config.dropout_rate,
    )
    utils.load_model_weights(detector, device, detector_checkpoint)
    utils.load_model_weights(student, device, student_checkpoint)
    return detector.to(device).eval(), student.to(device).eval()


def _candidate_records(candidate_mask: np.ndarray, probability: np.ndarray, subject_id: str) -> tuple[list[dict], np.ndarray]:
    records = []
    candidates = label(candidate_mask, connectivity=3)
    for region in regionprops(candidates):
        records.append({
            "candidate_id": int(region.label),
            "source_subject": subject_id,
            "voxel_count": int(region.area),
            "centroid": [float(value) for value in region.centroid],
            "bounding_box": [[int(region.bbox[index]), int(region.bbox[index + 3])] for index in range(3)],
            "detector_probability": float(probability[candidates == region.label].mean()),
        })
    return records, candidates


def _student_probabilities(student, patches: list[np.ndarray], device: torch.device, batch_size: int) -> np.ndarray:
    if not patches:
        return np.empty(0, dtype=np.float32)
    probabilities = []
    with torch.no_grad():
        for start in range(0, len(patches), batch_size):
            batch = torch.from_numpy(np.stack(patches[start:start + batch_size])).float().unsqueeze(1).to(device)
            batch = torch.cat((batch, frst.apply(batch)), dim=1)
            logits = student(batch)
            probabilities.append(F.softmax(logits, dim=1)[:, 1].cpu().numpy())
    return np.concatenate(probabilities)


def predict_volume(
    volume_path: Path,
    output_dir: Path,
    detector_config: DetectorConfig,
    student_config: StudentConfig,
    detector_checkpoint: Path,
    student_checkpoint: Path,
    preprocessing: PreprocessingConfig,
    device: torch.device = torch.device("cpu"),
    patch_batch_size: int = 8,
    subject_id: str | None = None,
) -> PredictionSummary:
    if patch_batch_size <= 0:
        raise ValueError("patch_batch_size must be positive")
    detector_threshold = detector_config.probability_threshold
    student_threshold = student_config.probability_threshold
    subject_id = subject_id or Path(volume_path).name.split(".")[0]
    image = utils.load_volume(volume_path)
    processed = processor.preprocess(image, None, **preprocessing.model_dump())
    detector, student = _load_models(
        detector_config, student_config, detector_checkpoint, student_checkpoint, device
    )

    detector_logits = processor.infer(detector, device, processed.image)
    detector_probability = F.softmax(detector_logits, dim=1).cpu().numpy()[0, 1]
    candidate_mask = detector_probability >= detector_threshold
    candidate_records, candidate_labels = _candidate_records(candidate_mask, detector_probability, subject_id)
    patch_records = []
    patches = []
    for region in regionprops(candidate_labels):
        center = tuple(int(round(value)) for value in region.centroid)
        starts = tuple(axis - 12 for axis in center)
        patch, bounds = _extract_fixed_patch(processed.image, starts, 24)
        patches.append(patch)
        patch_records.append((int(region.label), bounds))

    student_probabilities = _student_probabilities(student, patches, device, patch_batch_size)
    component_mask = np.zeros(processed.image.shape, dtype=np.uint8)
    probability_map = np.zeros(processed.image.shape, dtype=np.float32)
    by_id = {record["candidate_id"]: record for record in candidate_records}
    for (candidate_id, bounds), student_probability in zip(patch_records, student_probabilities):
        record = by_id[candidate_id]
        record["student_probability"] = float(student_probability)
        record["accepted"] = bool(student_probability >= student_threshold)
        if record["accepted"]:
            component_mask[candidate_labels == candidate_id] = 1
            probability_map[candidate_labels == candidate_id] = student_probability

    output_dir.mkdir(parents=True, exist_ok=True)
    mask_image = processor.restore_to_source(component_mask, processed.transform)
    probability_image = processor.restore_to_source(probability_map, processed.transform)
    mask_path = output_dir / f"{subject_id}_prediction.nii.gz"
    probability_path = output_dir / f"{subject_id}_probability.nii.gz"
    utils.save_volume(mask_image, mask_path)
    utils.save_volume(probability_image, probability_path)
    record_path = output_dir / f"{subject_id}_components.json"
    utils.write_json_atomic(record_path, {"subject_id": subject_id, "components": candidate_records})
    csv_path = output_dir / f"{subject_id}_components.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["candidate_id", "detector_probability", "student_probability", "accepted"])
        writer.writeheader()
        for record in candidate_records:
            writer.writerow({key: record.get(key) for key in writer.fieldnames})
    return PredictionSummary(
        subject_id=subject_id,
        mask_path=mask_path,
        probability_path=probability_path,
        components_path=record_path,
        component_count=len(candidate_records),
    )


def execute(config: InferCommandConfig) -> PredictionSummary:
    device = torch.device(config.device)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Seed and record provenance before any prediction is written, so every
    # output directory can be traced back to its config, seed, and revision.
    provenance.seed_everything(config.seed)
    provenance.write_provenance(
        config.output_dir, config, seed=config.seed, device=device
    )

    return predict_volume(
        volume_path=config.volume_path,
        output_dir=config.output_dir,
        detector_config=config.detector,
        student_config=config.student,
        detector_checkpoint=config.detector_checkpoint,
        student_checkpoint=config.student_checkpoint,
        preprocessing=config.preprocessing,
        device=device,
        patch_batch_size=config.patch_batch_size,
        subject_id=config.subject_id,
    )
