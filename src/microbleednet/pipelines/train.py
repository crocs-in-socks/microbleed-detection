from copy import deepcopy
from pathlib import Path

import json
import torch
from torch.utils.data import DataLoader
from torch.utils.data import BatchSampler
from torch.utils.data import SequentialSampler
from sklearn.model_selection import train_test_split

from ..core import utils as core_utils
from ..core import constants as core_constants
from ..core.engines.trainers import Trainer
from ..core.common.tasks import SegmentationTask
from ..core.common.tasks import SegmentationClassificationTask
from ..core.common.tasks import KnowledgeDistillationClassificationTask
from ..core.common.models import CandidateDetector
from ..core.common.models import CandidateDiscriminatorTeacher
from ..core.common.models import CandidateDiscriminatorStudent
from ..core.dataloading.samplers import EqualBatchSampler
from ..core.dataloading.datasets import SegmentationPatchDataset
from ..core.dataloading.datasets import SegmentationClassificationPatchDataset
from ..core.dataloading.datasets import ClassificationPatchDataset

from . import constants
from . import utils

"""
Example parameter dictionaries for `execute(...)` (all required keys with sample values).

datasplit_parameters = {
    # Must include either `test_size` or `train_size` for sklearn.model_selection.train_test_split
    "test_size": 0.2,
    "random_state": 42,
    "shuffle": True
}

patcher_parameters = {
    # Required by `utils.patch_subject_non_overlapping(subject, patch_dir, patch_size, augmentation_factor)`
    "patch_dir": Path("data/patches"),    # Path where materialized patches will be written
    "patch_size": 64,                      # int: size of cubic patch edge
    "augmentation_factor": 8               # int: how many augmented variants to produce per patch
}

dataset_parameters = {
    # Passed to `SegmentationPatchDataset(patches, **dataset_parameters)`
    "perform_augmentation": True           # bool
}

sampler_parameters = {
    # Used by EqualBatchSampler(patches, **sampler_parameters) and BatchSampler(..., **sampler_parameters)
    "batch_size": 8                        # int
}

dataloader_parameters = {
    # Optional DataLoader kwargs; safe defaults shown for cross-platform use
    "num_workers": 0,
    "pin_memory": False
}

model_parameters = {
    # Required by CandidateDetector(in_channels, n_classes, initial_channels)
    "in_channels": 1,
    "n_classes": 2,
    "initial_channels": 16
}

trainer_parameters = {
    # These keys satisfy Trainer(...) constructor
    "device": __import__("torch").device("cpu"),
    "optimizer_parameters": {"lr": 1e-4, "weight_decay": 1e-5, "clip_norm": 1.0},
    "scheduler_parameters": {"milestones": [10, 20], "gamma": 0.1},
    "checkpoint_dir": Path("checkpoints"),
}

fit_parameters = {
    "n_epochs": 30,
    # Optional Trainer.fit args
    "checkpoint_path": None,
    "weights_only": False,
    "compile_model": False
}

Notes:
- `datasplit_parameters` must provide `test_size` or `train_size` for `train_test_split`.
- `patcher_parameters` must contain the keyword args expected by the chosen patcher function
  (`patch_dir`, `patch_size`, `augmentation_factor` for `patch_subject_non_overlapping`).
"""

def _validate_subjects(subjects: list[dict]) -> None:
    missing = [subject.get("subject_id", "unknown") for subject in subjects if not subject.get("mask_path")]
    if missing:
        raise ValueError(f"training requires masks for subjects: {missing}")


def _write_stage_manifest(path: Path, stage: str, status: str, **details) -> None:
    payload = {"stage": stage, "status": status, **details}
    core_utils.write_json_atomic(path, payload)


def _stage_parameters(parameters: dict, experiment_dir: Path, stage: str) -> dict:
    copied = deepcopy(parameters)
    stage_root = experiment_dir / "train" / stage
    copied.setdefault("patcher_parameters", {})
    copied.setdefault("dataset_parameters", {})
    copied.setdefault("sampler_parameters", {"batch_size": 1})
    copied.setdefault("dataloader_parameters", {"num_workers": 0, "pin_memory": False})
    copied.setdefault("task_parameters", {})
    copied.setdefault("fit_parameters", {})
    copied.setdefault("model_parameters", {})
    copied.setdefault("trainer_parameters", {})
    copied["patcher_parameters"]["train_patch_dir"] = stage_root / "patches" / "train"
    copied["patcher_parameters"]["validation_patch_dir"] = stage_root / "patches" / "validation"
    copied["trainer_parameters"]["checkpoint_dir"] = stage_root / "checkpoints"
    return copied


def _build_loaders(train_subjects, validation_subjects, parameters, patcher, dataset_class):
    patcher_parameters = parameters["patcher_parameters"]
    train_parameters = {key: value for key, value in patcher_parameters.items() if key not in {"train_patch_dir", "validation_patch_dir"}}
    train_parameters["patch_dir"] = patcher_parameters["train_patch_dir"]
    train_parameters["augmentation_factor"] = parameters.get("train_augmentation_factor", train_parameters.get("augmentation_factor", 1))
    validation_parameters = dict(train_parameters)
    validation_parameters["patch_dir"] = patcher_parameters["validation_patch_dir"]
    validation_parameters["augmentation_factor"] = 1

    train_patches = utils.collect_patches(train_subjects, patcher, train_parameters)
    validation_patches = utils.collect_patches(validation_subjects, patcher, validation_parameters)
    dataset_parameters = dict(parameters["dataset_parameters"])
    dataset_parameters["perform_augmentation"] = True
    train_dataset = dataset_class(train_patches, **dataset_parameters)
    validation_dataset = dataset_class(validation_patches, perform_augmentation=False)
    train_sampler = EqualBatchSampler(train_patches, **parameters["sampler_parameters"])
    validation_sampler = BatchSampler(SequentialSampler(validation_dataset), **parameters["sampler_parameters"])
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, **parameters["dataloader_parameters"])
    validation_loader = DataLoader(validation_dataset, batch_sampler=validation_sampler, **parameters["dataloader_parameters"])
    return train_loader, validation_loader


def _run_stage(stage: str, train_subjects, validation_subjects, parameters, model, task, patcher, dataset_class, experiment_dir: Path):
    stage_parameters = parameters
    manifest_path = experiment_dir / "manifests" / f"{stage}.json"
    _write_stage_manifest(manifest_path, stage, "running")
    try:
        train_loader, validation_loader = _build_loaders(
            train_subjects, validation_subjects, stage_parameters, patcher, dataset_class
        )
        fit_parameters = dict(stage_parameters["fit_parameters"])
        compile_model = fit_parameters.pop("compile_model", None)
        trainer_parameters = dict(stage_parameters["trainer_parameters"])
        if compile_model is not None:
            trainer_parameters["compile_model"] = compile_model
        trainer = Trainer(model, task, **trainer_parameters)
        trainer.fit(train_loader, validation_loader, **fit_parameters)
        _write_stage_manifest(
            manifest_path,
            stage,
            "complete",
            checkpoint_dir=str(stage_parameters["trainer_parameters"]["checkpoint_dir"].resolve()),
            train_patch_dir=str(stage_parameters["patcher_parameters"]["train_patch_dir"].resolve()),
            validation_patch_dir=str(stage_parameters["patcher_parameters"]["validation_patch_dir"].resolve()),
            records=trainer.epoch_records,
        )
    except Exception as error:
        _write_stage_manifest(manifest_path, stage, "failed", error=str(error))
        raise


def train_detector(train_subjects, validation_subjects, parameters, experiment_dir: Path) -> Path:
    stage_parameters = _stage_parameters(parameters, experiment_dir, "detector")
    stage_parameters["model_parameters"]["in_channels"] = 2
    stage_parameters["model_parameters"].setdefault("initial_channels", 64)
    model = CandidateDetector(**stage_parameters["model_parameters"])
    task = SegmentationTask()
    _run_stage("detector", train_subjects, validation_subjects, stage_parameters, model, task, utils.patch_subject_non_overlapping, SegmentationPatchDataset, experiment_dir)
    return stage_parameters["trainer_parameters"]["checkpoint_dir"] / core_constants.engines.trainers.best_checkpoint_path


def train_teacher(train_subjects, validation_subjects, parameters, detector_checkpoint: Path, experiment_dir: Path) -> Path:
    stage_parameters = _stage_parameters(parameters, experiment_dir, "teacher")
    stage_parameters["model_parameters"]["in_channels"] = 2
    detector = CandidateDetector(**stage_parameters["model_parameters"])
    core_utils.load_model_weights(detector, stage_parameters["trainer_parameters"]["device"], detector_checkpoint)
    teacher = CandidateDiscriminatorTeacher(**stage_parameters["model_parameters"])
    core_utils.initialize_teacher_from_detector(detector, teacher)
    stage_parameters["patcher_parameters"]["model"] = detector
    stage_parameters["patcher_parameters"]["device"] = stage_parameters["trainer_parameters"]["device"]
    stage_parameters["patcher_parameters"]["threshold"] = constants.train.default.detector_threshold
    stage_parameters["patcher_parameters"]["patch_size"] = 24
    stage_parameters["train_augmentation_factor"] = 5
    _run_stage("teacher", train_subjects, validation_subjects, stage_parameters, teacher, SegmentationClassificationTask(), utils.patch_subject_target_centered, SegmentationClassificationPatchDataset, experiment_dir)
    core_utils.delete_model(detector)
    return stage_parameters["trainer_parameters"]["checkpoint_dir"] / core_constants.engines.trainers.best_checkpoint_path


def train_student(train_subjects, validation_subjects, parameters, detector_checkpoint: Path, teacher_checkpoint: Path, experiment_dir: Path) -> Path:
    stage_parameters = _stage_parameters(parameters, experiment_dir, "student")
    stage_parameters["model_parameters"]["in_channels"] = 2
    detector = CandidateDetector(**stage_parameters["model_parameters"])
    teacher = CandidateDiscriminatorTeacher(**stage_parameters["model_parameters"])
    device = stage_parameters["trainer_parameters"]["device"]
    core_utils.load_model_weights(detector, device, detector_checkpoint)
    core_utils.load_model_weights(teacher, device, teacher_checkpoint)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    stage_parameters["patcher_parameters"].update({"model": detector, "device": device, "threshold": constants.train.default.detector_threshold, "patch_size": 24})
    stage_parameters["train_augmentation_factor"] = 5
    student = CandidateDiscriminatorStudent(**stage_parameters["model_parameters"])
    stage_parameters["task_parameters"] = {"teacher_model": teacher}
    _run_stage("student", train_subjects, validation_subjects, stage_parameters, student, KnowledgeDistillationClassificationTask(teacher), utils.patch_subject_target_centered, ClassificationPatchDataset, experiment_dir)
    core_utils.delete_model(detector)
    core_utils.delete_model(teacher)
    return stage_parameters["trainer_parameters"]["checkpoint_dir"] / core_constants.engines.trainers.best_checkpoint_path


def execute(
    dataset_dir: Path,
    experiment_dir: Path,
    datasplit_parameters: dict,
    detector_parameters: dict,
    discriminator_teacher_parameters: dict,
    discriminator_student_parameters: dict
) -> None:
    preprocessed_manifest_path = dataset_dir / constants.manifests.preprocessed
    with open(preprocessed_manifest_path, "r") as preprocessed_manifest_file:
        preprocessed_manifest_content = json.load(preprocessed_manifest_file)
        preprocessed_subjects = preprocessed_manifest_content.get("subjects", [])

    _validate_subjects(preprocessed_subjects)
    train_subjects, test_subjects = train_test_split(preprocessed_subjects, **datasplit_parameters)

    # set up experiment directory
    experiment_dir.mkdir(parents=True, exist_ok=True)

    detector_checkpoint = train_detector(train_subjects, test_subjects, detector_parameters, experiment_dir)
    teacher_checkpoint = train_teacher(train_subjects, test_subjects, discriminator_teacher_parameters, detector_checkpoint, experiment_dir)
    train_student(train_subjects, test_subjects, discriminator_student_parameters, detector_checkpoint, teacher_checkpoint, experiment_dir)