from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import BatchSampler, DataLoader, SequentialSampler

from ...core import io as core_io
from ...core import utils as core_utils
from ...core.common.models import (
    CandidateDetector,
    CandidateDiscriminatorStudent,
    CandidateDiscriminatorTeacher,
)
from ...core.common.tasks import (
    KnowledgeDistillationClassificationTask,
    SegmentationClassificationTask,
    SegmentationTask,
)
from ...core.dataloading.datasets import (
    ClassificationPatchDataset,
    SegmentationClassificationPatchDataset,
    SegmentationPatchDataset,
)
from ...core.dataloading.samplers import EqualBatchSampler
from ...core.engines.trainers import CheckpointConfig, Trainer
from .. import manifests, patching, provenance, utils
from ..configs import TrainConfig
from ..layouts import DatasetLayout, ExperimentLayout
from ..manifests import (
    ManifestStatus,
    PreprocessedDatasetManifest,
    PreprocessedSubject,
    SplitManifest,
    TrainingStageManifest,
)


@dataclass(frozen=True)
class StageRuntime:
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


def stage_runtime(
    experiment_dir: Path, device: torch.device, stage: str
) -> StageRuntime:
    layout = ExperimentLayout()
    return StageRuntime(
        stage=stage,
        device=device,
        train_patch_dir=experiment_dir / layout.train_patch_dir(stage),
        validation_patch_dir=experiment_dir / layout.validation_patch_dir(stage),
        checkpoint_dir=experiment_dir / layout.checkpoint_dir(stage),
        manifest_path=experiment_dir / layout.stage_manifest(stage),
        best_checkpoint=experiment_dir / layout.best_checkpoint(stage),
    )


def write_stage_manifest(
    path: Path,
    stage: str,
    status: ManifestStatus,
    created_at: str,
    *,
    error: str | None = None,
    checkpoint_dir: str | None = None,
    train_patch_dir: str | None = None,
    validation_patch_dir: str | None = None,
    records: list[dict[str, float]] | None = None,
) -> None:
    manifest = TrainingStageManifest(
        status=status,
        created_at=created_at,
        updated_at=manifests.timestamp(),
        error=error,
        stage=stage,
        checkpoint_dir=checkpoint_dir,
        train_patch_dir=train_patch_dir,
        validation_patch_dir=validation_patch_dir,
        records=records or [],
    )
    manifests.write_manifest(path, manifest)


def persist_split(
    path: Path,
    train_subjects: list[PreprocessedSubject],
    validation_subjects: list[PreprocessedSubject],
    datasplit,
) -> None:
    """Write the train/validation split once; refuse a conflicting rewrite.

    The split is durable so a resumed run reads the same partition instead of
    regenerating it. A byte-identical manifest already on disk is left alone;
    any divergence is a hard error rather than a silent overwrite.
    """
    now = manifests.timestamp()
    manifest = SplitManifest(
        status=ManifestStatus.COMPLETE,
        created_at=now,
        updated_at=now,
        random_state=datasplit.random_state,
        test_size=datasplit.test_size,
        shuffle=datasplit.shuffle,
        train=[subject.subject_id for subject in train_subjects],
        validation=[subject.subject_id for subject in validation_subjects],
    )
    if path.exists():
        existing = manifests.read_manifest(path, SplitManifest)
        diverged = (
            existing.train != manifest.train
            or existing.validation != manifest.validation
        )
        if diverged:
            raise ValueError(
                f"split manifest already exists with a different split: {path}"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    manifests.write_manifest(path, manifest)


def build_loaders(
    train_subjects,
    validation_subjects,
    runtime: StageRuntime,
    config: TrainConfig,
    patcher: patching.Patcher,
    dataset_class,
    augmentation_factor: int,
):
    train_patches = patcher.collect(
        train_subjects, runtime.train_patch_dir, augmentation_factor
    )
    # Validation patches are never augmented, so the factor is fixed at 1.
    validation_patches = patcher.collect(
        validation_subjects, runtime.validation_patch_dir, 1
    )

    train_dataset = dataset_class(
        train_patches,
        perform_augmentation=True,
        augmentation=config.augmentation,
    )
    validation_dataset = dataset_class(validation_patches, perform_augmentation=False)

    train_sampler = EqualBatchSampler(
        train_patches, batch_size=config.trainer.batch_size
    )
    validation_sampler = BatchSampler(
        SequentialSampler(validation_dataset),
        batch_size=config.trainer.batch_size,
        drop_last=False,
    )

    loader_kwargs = {
        "num_workers": config.num_workers,
        "pin_memory": config.pin_memory,
        # Seed each worker's RNG from the (already-seeded) base seed so patch
        # augmentation is reproducible across workers.
        "worker_init_fn": provenance.seed_worker,
    }
    train_loader = DataLoader(
        train_dataset, batch_sampler=train_sampler, **loader_kwargs
    )
    validation_loader = DataLoader(
        validation_dataset, batch_sampler=validation_sampler, **loader_kwargs
    )
    return train_loader, validation_loader


def run_stage(
    runtime: StageRuntime,
    train_subjects,
    validation_subjects,
    config: TrainConfig,
    model,
    task,
    patcher: patching.Patcher,
    dataset_class,
    augmentation_factor: int,
):
    created_at = manifests.timestamp()
    write_stage_manifest(
        runtime.manifest_path,
        runtime.stage,
        ManifestStatus.RUNNING,
        created_at,
    )
    try:
        # The target-centered patcher loads its detector only while collecting
        # patches. Once loaders read patches from disk, nothing in fit() needs
        # it, so the patcher is dropped and its GPU memory reclaimed before the
        # trainer is built.
        train_loader, validation_loader = build_loaders(
            train_subjects,
            validation_subjects,
            runtime,
            config,
            patcher,
            dataset_class,
            augmentation_factor,
        )
        del patcher
        utils.release_gpu_memory()

        layout = ExperimentLayout()
        trainer = Trainer(
            model,
            task,
            device=runtime.device,
            config=config.trainer,
            checkpoints=CheckpointConfig(
                directory=runtime.checkpoint_dir,
                stage=runtime.stage,
                latest_name=layout.latest_checkpoint_name,
                best_name=layout.best_checkpoint_name,
            ),
        )
        trainer.fit(train_loader, validation_loader)
        write_stage_manifest(
            runtime.manifest_path,
            runtime.stage,
            ManifestStatus.COMPLETE,
            created_at,
            checkpoint_dir=str(runtime.checkpoint_dir.resolve()),
            train_patch_dir=str(runtime.train_patch_dir.resolve()),
            validation_patch_dir=str(runtime.validation_patch_dir.resolve()),
            records=trainer.epoch_records,
        )
    except Exception as error:
        write_stage_manifest(
            runtime.manifest_path,
            runtime.stage,
            ManifestStatus.FAILED,
            created_at,
            error=str(error),
        )
        raise


def train_detector(
    train_subjects,
    validation_subjects,
    config: TrainConfig,
    device: torch.device,
) -> Path:
    runtime = stage_runtime(config.experiment_dir, device, "detector")

    detector = CandidateDetector(config.detector.architecture)

    run_stage(
        runtime,
        train_subjects,
        validation_subjects,
        config,
        detector,
        SegmentationTask(),
        patching.NonOverlappingPatcher(config.detector.patch_size),
        SegmentationPatchDataset,
        config.detector.augmentation_factor,
    )
    return runtime.best_checkpoint


def train_teacher(
    train_subjects,
    validation_subjects,
    config: TrainConfig,
    device: torch.device,
    detector_checkpoint: Path,
) -> Path:
    runtime = stage_runtime(config.experiment_dir, device, "teacher")

    teacher = CandidateDiscriminatorTeacher(config.teacher.architecture)

    # Copy the detector trunk into the teacher, then release this detector: the
    # patcher below loads its own copy, so the initializer does not linger on the
    # device through training.
    initializer = CandidateDetector(config.detector.architecture)
    core_io.load_model_weights(initializer, device, detector_checkpoint)
    core_utils.initialize_teacher_from_detector(initializer, teacher)
    utils.delete_model(initializer)

    run_stage(
        runtime,
        train_subjects,
        validation_subjects,
        config,
        teacher,
        SegmentationClassificationTask(),
        patching.NonOverlappingPatcher(config.teacher.patch_size),
        SegmentationClassificationPatchDataset,
        config.teacher.augmentation_factor,
    )
    return runtime.best_checkpoint


def train_student(
    train_subjects,
    validation_subjects,
    config: TrainConfig,
    device: torch.device,
    detector_checkpoint: Path,
    teacher_checkpoint: Path,
) -> Path:
    runtime = stage_runtime(config.experiment_dir, device, "student")

    teacher = CandidateDiscriminatorTeacher(config.teacher.architecture)
    core_io.load_model_weights(teacher, device, teacher_checkpoint)

    student = CandidateDiscriminatorStudent(config.student.architecture)

    run_stage(
        runtime,
        train_subjects,
        validation_subjects,
        config,
        student,
        KnowledgeDistillationClassificationTask(
            teacher,
            alpha=config.student.alpha,
            beta=config.student.beta,
            temperature=config.student.temperature,
        ),
        patching.TargetCenteredPatcher(
            config.student.patch_size,
            architecture=config.detector.architecture,
            checkpoint_path=detector_checkpoint,
            device=device,
            threshold=config.detector.probability_threshold,
        ),
        ClassificationPatchDataset,
        config.student.augmentation_factor,
    )
    utils.delete_model(teacher)
    return runtime.best_checkpoint


def execute(config: TrainConfig) -> None:
    layout = DatasetLayout()
    preprocessed_manifest = manifests.read_manifest(
        config.dataset_dir / layout.preprocessed_manifest,
        PreprocessedDatasetManifest,
    )
    preprocessed_subjects = preprocessed_manifest.subjects

    # train_test_split is untyped; it returns the same element type it is given,
    # so cast the two halves back to the subject type it erased.
    train_subjects, validation_subjects = cast(
        tuple[list[PreprocessedSubject], list[PreprocessedSubject]],
        train_test_split(
            preprocessed_subjects,
            test_size=config.datasplit.test_size,
            random_state=config.datasplit.random_state,
            shuffle=config.datasplit.shuffle,
        ),
    )

    device = torch.device(config.device)
    config.experiment_dir.mkdir(parents=True, exist_ok=True)

    # Seed and record provenance before any artifact is written, so a run that
    # fails mid-training is still traceable to its config, seed, and revision.
    provenance.seed_everything(config.seed)
    provenance.write_provenance(
        config.experiment_dir, config, seed=config.seed, device=device
    )

    # Persist the split write-once so a resumed run reuses the same partition.
    persist_split(
        config.experiment_dir / ExperimentLayout().split_manifest(),
        train_subjects,
        validation_subjects,
        config.datasplit,
    )

    detector_checkpoint = train_detector(
        train_subjects,
        validation_subjects,
        config,
        device,
    )
    teacher_checkpoint = train_teacher(
        train_subjects,
        validation_subjects,
        config,
        device,
        detector_checkpoint,
    )
    train_student(
        train_subjects,
        validation_subjects,
        config,
        device,
        detector_checkpoint,
        teacher_checkpoint,
    )
