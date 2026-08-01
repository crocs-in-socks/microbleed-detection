import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from skimage.measure import label, regionprops

from ..core import utils
from ..core.common.models import CandidateDetector, CandidateDiscriminatorStudent
from ..core.engines import processor
from ..core.transforms import frst
from ..core.transforms.patch import _extract_fixed_patch


def _load_models(model_parameters: dict, detector_checkpoint: Path, student_checkpoint: Path, device: torch.device):
    detector = CandidateDetector(**model_parameters)
    student = CandidateDiscriminatorStudent(**model_parameters)
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
    model_parameters: dict,
    detector_checkpoint: Path,
    student_checkpoint: Path,
    preprocess_parameters: dict,
    device: torch.device = torch.device("cpu"),
    detector_threshold: float = 0.5,
    student_threshold: float = 0.5,
    patch_batch_size: int = 8,
    subject_id: str | None = None,
) -> dict:
    if patch_batch_size <= 0:
        raise ValueError("patch_batch_size must be positive")
    subject_id = subject_id or Path(volume_path).name.split(".")[0]
    image = utils.load_volume(volume_path)
    processed = processor.preprocess(image, None, **preprocess_parameters)
    detector, student = _load_models(model_parameters, detector_checkpoint, student_checkpoint, device)

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
    return {"subject_id": subject_id, "mask_path": str(mask_path), "probability_path": str(probability_path), "components_path": str(record_path), "component_count": len(candidate_records)}


def execute(*args, **kwargs) -> dict:
    return predict_volume(*args, **kwargs)
