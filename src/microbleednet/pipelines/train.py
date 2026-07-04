from pathlib import Path

import gc
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

    train_subjects, test_subjects = train_test_split(preprocessed_subjects, **datasplit_parameters)

    # set up experiment directory
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Train detector
    detector_parameters["patcher_parameters"]["patch_dir"] = experiment_dir / constants.train.detector_patches_dir
    detector_parameters["trainer_parameters"]["checkpoint_dir"] = experiment_dir / constants.train.detector_checkpoints_dir

    pipeline(train_subjects, test_subjects, detector_parameters, CandidateDetector, SegmentationPatchDataset, SegmentationTask, utils.patch_subject_non_overlapping)

    # Train discriminator teacher
    discriminator_teacher_parameters["patcher_parameters"]["patch_dir"] = experiment_dir / constants.train.discriminator_teacher_patches_dir
    discriminator_teacher_parameters["trainer_parameters"]["checkpoint_dir"] = experiment_dir / constants.train.discriminator_teacher_checkpoints_dir
    discriminator_teacher_parameters["fit_parameters"]["checkpoint_path"] = detector_parameters["trainer_parameters"]["checkpoint_dir"] / core_constants.engines.trainers.best_checkpoint_path
    discriminator_teacher_parameters["fit_parameters"]["weights_only"] = True

    pipeline(train_subjects, test_subjects, discriminator_teacher_parameters, CandidateDiscriminatorTeacher, SegmentationClassificationPatchDataset, SegmentationClassificationTask, utils.patch_subject_non_overlapping)

    # Train discriminator student
    detector_model = CandidateDetector(**detector_parameters["model_parameters"])
    detector_model = detector_model.to(discriminator_student_parameters["trainer_parameters"]["device"])
    core_utils.load_model_weights(
        detector_model,
        discriminator_student_parameters["trainer_parameters"]["device"],
        detector_parameters["trainer_parameters"]["checkpoint_dir"] / core_constants.engines.trainers.best_checkpoint_path
    )

    discriminator_student_parameters["patcher_parameters"]["patch_dir"] = experiment_dir / constants.train.discriminator_student_patches_dir
    discriminator_student_parameters["patcher_parameters"]["model"] = detector_model
    discriminator_student_parameters["patcher_parameters"]["device"] = discriminator_student_parameters["trainer_parameters"]["device"]

    discriminator_student_parameters["trainer_parameters"]["checkpoint_dir"] = experiment_dir / constants.train.discriminator_student_checkpoints_dir

    teacher_model = CandidateDiscriminatorTeacher(**discriminator_teacher_parameters["model_parameters"])
    teacher_model = teacher_model.to(discriminator_student_parameters["trainer_parameters"]["device"])
    core_utils.load_model_weights(
        teacher_model,
        discriminator_student_parameters["trainer_parameters"]["device"],
        discriminator_teacher_parameters["trainer_parameters"]["checkpoint_dir"] / core_constants.engines.trainers.best_checkpoint_path
    )
    discriminator_student_parameters["task_parameters"] = {
        "teacher_model": teacher_model
    }

    pipeline(train_subjects, test_subjects, discriminator_student_parameters, CandidateDiscriminatorStudent, ClassificationPatchDataset, KnowledgeDistillationClassificationTask, utils.patch_subject_target_centered)


def pipeline(
    train_subjects: list,
    test_subjects: list,
    parameters: dict,
    ModelClass,
    DatasetClass,
    TaskClass,
    patcher_function
) -> None:
    train_patches = utils.collect_patches(train_subjects, patcher_function, parameters["patcher_parameters"])
    train_set = DatasetClass(train_patches, **parameters["dataset_parameters"])
    train_sampler = EqualBatchSampler(train_patches, **parameters["sampler_parameters"])
    train_loader = DataLoader(
        dataset=train_set,
        batch_sampler=train_sampler,
        **parameters["dataloader_parameters"]
    )

    test_patcher_parameters = parameters["patcher_parameters"].copy()
    test_patcher_parameters["augmentation_factor"] = 1 # test set doesn't need augmentation

    test_patches = utils.collect_patches(test_subjects, patcher_function, test_patcher_parameters)
    test_set = DatasetClass(test_patches)
    test_sampler = BatchSampler(SequentialSampler(test_set), **parameters["sampler_parameters"])
    test_loader = DataLoader(
        dataset=test_set,
        batch_sampler=test_sampler,
        **parameters["dataloader_parameters"]
    )

    # Cleanup model from patcher params if it exists (specific to student pipeline)
    if "model" in parameters["patcher_parameters"]:
        cached_model = parameters["patcher_parameters"].pop("model")
        del cached_model
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    model = ModelClass(**parameters["model_parameters"])
    task = TaskClass(**parameters.get("task_parameters", {}))
    trainer = Trainer(model, task, **parameters["trainer_parameters"])

    trainer.fit(train_loader, test_loader, **parameters["fit_parameters"])

    utils.delete_model(model)