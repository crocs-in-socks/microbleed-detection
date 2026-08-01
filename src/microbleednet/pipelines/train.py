import json
import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import BatchSampler, DataLoader, SequentialSampler

from .. import provenance
from ..config import (
    DetectorConfig,
    ModelConfig,
    StudentConfig,
    TeacherConfig,
    TrainCommandConfig,
    TrainerConfig,
)
from ..core import constants as core_constants
from ..core import utils as core_utils
from ..core.common.models import (
    CandidateDetector,
    CandidateDiscriminatorStudent,
    CandidateDiscriminatorTeacher,
)
from ..core.common.tasks import (
    KnowledgeDistillationClassificationTask,
    SegmentationClassificationTask,
    SegmentationTask,
)
from ..core.dataloading.datasets import (
    ClassificationPatchDataset,
    SegmentationClassificationPatchDataset,
    SegmentationPatchDataset,
)
from ..core.dataloading.samplers import EqualBatchSampler
from . import constants, utils

logger = logging.getLogger(__name__)

# Every training stage feeds the model two input channels: the preprocessed
# volume plus its FRST transform, concatenated inside the task's training step.
_STAGE_INPUT_CHANNELS = 2

# The teacher and student operate on candidate-centered patches, which the
# classifier head requires to be 24^3 (see Classifier.expected_features).
_CANDIDATE_PATCH_SIZE = 24


@dataclass(frozen=True)
class _StageRuntime:
    """Derived paths and device for one training stage.

    This is the mutable-at-construction counterpart to the immutable config:
    it carries values computed from ``experiment_dir`` and the resolved device,
    which cannot live inside a frozen Pydantic config.
    """

    stage: str
    device: torch.device
    train_patch_dir: Path
    validation_patch_dir: Path
    checkpoint_dir: Path
    manifest_path: Path
    best_checkpoint: Path


def _stage_runtime(
    experiment_dir: Path, device: torch.device, stage: str
) -> _StageRuntime:
    stage_root = experiment_dir / "train" / stage
    checkpoint_dir = stage_root / "checkpoints"
    best_checkpoint_name = core_constants.engines.trainers.best_checkpoint_path
    return _StageRuntime(
        stage=stage,
        device=device,
        train_patch_dir=stage_root / "patches" / "train",
        validation_patch_dir=stage_root / "patches" / "validation",
        checkpoint_dir=checkpoint_dir,
        manifest_path=experiment_dir / "manifests" / f"{stage}.json",
        best_checkpoint=checkpoint_dir / best_checkpoint_name,
    )


def _model_kwargs(model_config: ModelConfig) -> dict:
    """Shared 3D U-Net constructor kwargs, with paper-fixed input channels.

    Every stage trains on two input channels regardless of what the config
    file declares, so this overrides ``input_channels`` to the paper value.
    """
    return {
        "input_channels": _STAGE_INPUT_CHANNELS,
        "output_classes": model_config.output_classes,
        "initial_channels": model_config.initial_channels,
    }


def _validate_subjects(subjects: list[dict]) -> None:
    missing = [
        subject.get("subject_id", "unknown")
        for subject in subjects
        if not subject.get("mask_path")
    ]
    if missing:
        raise ValueError(f"training requires masks for subjects: {missing}")


def _write_stage_manifest(path: Path, stage: str, status: str, **details) -> None:
    payload = {"stage": stage, "status": status, **details}
    core_utils.write_json_atomic(path, payload)


def _optimizer_parameters(trainer_config: TrainerConfig) -> dict:
    return {
        "lr": trainer_config.learning_rate,
        "eps": trainer_config.adam_epsilon,
        "clip_norm": core_constants.engines.trainers.default.clip_norm,
    }


def _scheduler_parameters(trainer_config: TrainerConfig) -> dict:
    return {
        "gamma": trainer_config.learning_rate_factor,
        "step_size": trainer_config.learning_rate_period,
    }


def _build_loaders(
    train_subjects,
    validation_subjects,
    runtime: _StageRuntime,
    trainer_config: TrainerConfig,
    command_config: TrainCommandConfig,
    patcher,
    patcher_parameters: dict,
    dataset_class,
    augmentation_factor: int,
):
    train_parameters = {
        **patcher_parameters,
        "patch_dir": runtime.train_patch_dir,
        "augmentation_factor": augmentation_factor,
    }
    validation_parameters = {
        **patcher_parameters,
        "patch_dir": runtime.validation_patch_dir,
        "augmentation_factor": 1,
    }

    train_patches = utils.collect_patches(train_subjects, patcher, train_parameters)
    validation_patches = utils.collect_patches(
        validation_subjects, patcher, validation_parameters
    )

    train_dataset = dataset_class(
        train_patches,
        perform_augmentation=True,
        augmentation=command_config.augmentation,
    )
    validation_dataset = dataset_class(
        validation_patches, perform_augmentation=False
    )

    train_sampler = EqualBatchSampler(
        train_patches, batch_size=trainer_config.batch_size
    )
    validation_sampler = BatchSampler(
        SequentialSampler(validation_dataset),
        batch_size=trainer_config.batch_size,
        drop_last=False,
    )

    loader_kwargs = {
        "num_workers": command_config.num_workers,
        "pin_memory": command_config.pin_memory,
    }
    train_loader = DataLoader(
        train_dataset, batch_sampler=train_sampler, **loader_kwargs
    )
    validation_loader = DataLoader(
        validation_dataset, batch_sampler=validation_sampler, **loader_kwargs
    )
    return train_loader, validation_loader


def _run_stage(
    runtime: _StageRuntime,
    train_subjects,
    validation_subjects,
    trainer_config: TrainerConfig,
    command_config: TrainCommandConfig,
    model,
    task,
    patcher,
    patcher_parameters: dict,
    dataset_class,
    augmentation_factor: int,
):
    from ..core.engines.trainers import Trainer

    _write_stage_manifest(runtime.manifest_path, runtime.stage, "running")
    try:
        train_loader, validation_loader = _build_loaders(
            train_subjects,
            validation_subjects,
            runtime,
            trainer_config,
            command_config,
            patcher,
            patcher_parameters,
            dataset_class,
            augmentation_factor,
        )
        trainer = Trainer(
            model,
            task,
            device=runtime.device,
            optimizer_parameters=_optimizer_parameters(trainer_config),
            scheduler_parameters=_scheduler_parameters(trainer_config),
            checkpoint_dir=runtime.checkpoint_dir,
            stage=runtime.stage,
            patience=trainer_config.patience,
        )
        trainer.fit(
            train_loader, validation_loader, n_epochs=trainer_config.max_epochs
        )
        _write_stage_manifest(
            runtime.manifest_path,
            runtime.stage,
            "complete",
            checkpoint_dir=str(runtime.checkpoint_dir.resolve()),
            train_patch_dir=str(runtime.train_patch_dir.resolve()),
            validation_patch_dir=str(runtime.validation_patch_dir.resolve()),
            records=trainer.epoch_records,
        )
    except Exception as error:
        _write_stage_manifest(
            runtime.manifest_path, runtime.stage, "failed", error=str(error)
        )
        raise


def train_detector(
    train_subjects,
    validation_subjects,
    detector_config: DetectorConfig,
    trainer_config: TrainerConfig,
    command_config: TrainCommandConfig,
    device: torch.device,
    experiment_dir: Path,
) -> Path:
    runtime = _stage_runtime(experiment_dir, device, "detector")
    model = CandidateDetector(**_model_kwargs(detector_config))
    task = SegmentationTask()
    _run_stage(
        runtime,
        train_subjects,
        validation_subjects,
        trainer_config,
        command_config,
        model,
        task,
        utils.patch_subject_non_overlapping,
        {"patch_size": detector_config.patch_size},
        SegmentationPatchDataset,
        detector_config.augmentation_factor,
    )
    return runtime.best_checkpoint


def train_teacher(
    train_subjects,
    validation_subjects,
    detector_config: DetectorConfig,
    teacher_config: TeacherConfig,
    trainer_config: TrainerConfig,
    command_config: TrainCommandConfig,
    device: torch.device,
    detector_checkpoint: Path,
    experiment_dir: Path,
) -> Path:
    runtime = _stage_runtime(experiment_dir, device, "teacher")

    detector = CandidateDetector(**_model_kwargs(detector_config))
    core_utils.load_model_weights(detector, device, detector_checkpoint)

    teacher = CandidateDiscriminatorTeacher(
        **_model_kwargs(teacher_config), dropout_rate=teacher_config.dropout_rate
    )
    core_utils.initialize_teacher_from_detector(detector, teacher)

    patcher_parameters = {
        "patch_size": _CANDIDATE_PATCH_SIZE,
        "model": detector,
        "device": device,
        "threshold": detector_config.probability_threshold,
    }
    _run_stage(
        runtime,
        train_subjects,
        validation_subjects,
        trainer_config,
        command_config,
        teacher,
        SegmentationClassificationTask(),
        utils.patch_subject_target_centered,
        patcher_parameters,
        SegmentationClassificationPatchDataset,
        teacher_config.augmentation_factor,
    )
    utils.delete_model(detector)
    return runtime.best_checkpoint


def train_student(
    train_subjects,
    validation_subjects,
    detector_config: DetectorConfig,
    teacher_config: TeacherConfig,
    student_config: StudentConfig,
    trainer_config: TrainerConfig,
    command_config: TrainCommandConfig,
    device: torch.device,
    detector_checkpoint: Path,
    teacher_checkpoint: Path,
    experiment_dir: Path,
) -> Path:
    runtime = _stage_runtime(experiment_dir, device, "student")

    detector = CandidateDetector(**_model_kwargs(detector_config))
    teacher = CandidateDiscriminatorTeacher(
        **_model_kwargs(teacher_config), dropout_rate=teacher_config.dropout_rate
    )
    core_utils.load_model_weights(detector, device, detector_checkpoint)
    core_utils.load_model_weights(teacher, device, teacher_checkpoint)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    student = CandidateDiscriminatorStudent(
        **_model_kwargs(student_config), dropout_rate=student_config.dropout_rate
    )
    patcher_parameters = {
        "patch_size": _CANDIDATE_PATCH_SIZE,
        "model": detector,
        "device": device,
        "threshold": detector_config.probability_threshold,
    }
    _run_stage(
        runtime,
        train_subjects,
        validation_subjects,
        trainer_config,
        command_config,
        student,
        KnowledgeDistillationClassificationTask(
            teacher,
            alpha=student_config.alpha,
            beta=student_config.beta,
            temperature=student_config.temperature,
        ),
        utils.patch_subject_target_centered,
        patcher_parameters,
        ClassificationPatchDataset,
        student_config.augmentation_factor,
    )
    utils.delete_model(detector)
    utils.delete_model(teacher)
    return runtime.best_checkpoint


def execute(config: TrainCommandConfig) -> None:
    preprocessed_manifest_path = config.dataset_dir / constants.manifests.preprocessed
    with open(preprocessed_manifest_path, "r") as preprocessed_manifest_file:
        preprocessed_manifest_content = json.load(preprocessed_manifest_file)
        preprocessed_subjects = preprocessed_manifest_content.get("subjects", [])

    _validate_subjects(preprocessed_subjects)
    train_subjects, validation_subjects = train_test_split(
        preprocessed_subjects,
        test_size=config.datasplit.test_size,
        random_state=config.datasplit.random_state,
        shuffle=config.datasplit.shuffle,
    )

    device = torch.device(config.device)
    config.experiment_dir.mkdir(parents=True, exist_ok=True)

    # Seed and record provenance before any artifact is written, so a run that
    # fails mid-training is still traceable to its config, seed, and revision.
    provenance.seed_everything(config.seed)
    provenance.write_provenance(
        config.experiment_dir, config, seed=config.seed, device=device
    )

    detector_checkpoint = train_detector(
        train_subjects,
        validation_subjects,
        config.detector,
        config.trainer,
        config,
        device,
        config.experiment_dir,
    )
    teacher_checkpoint = train_teacher(
        train_subjects,
        validation_subjects,
        config.detector,
        config.teacher,
        config.trainer,
        config,
        device,
        detector_checkpoint,
        config.experiment_dir,
    )
    train_student(
        train_subjects,
        validation_subjects,
        config.detector,
        config.teacher,
        config.student,
        config.trainer,
        config,
        device,
        detector_checkpoint,
        teacher_checkpoint,
        config.experiment_dir,
    )
