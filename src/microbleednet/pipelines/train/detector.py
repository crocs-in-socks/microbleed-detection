from pathlib import Path

import json
from torch.utils.data import DataLoader
from torch.utils.data import BatchSampler
from torch.utils.data import SequentialSampler
from sklearn.model_selection import train_test_split

from ...core.engines.trainers import Trainer
from ...core.common.tasks import SegmentationTask
from ...core.common.models import CandidateDetector
from ...core.dataloading.samplers import EqualBatchSampler
from ...core.dataloading.datasets import SegmentationPatchDataset

from .. import constants
from .. import utils

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
    # These keys satisfy Trainer(...) constructor and Trainer.fit(...)
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
    datasplit_parameters: dict,
    patcher_parameters: dict,
    dataset_parameters: dict,
    sampler_parameters: dict,
    dataloader_parameters: dict,
    model_parameters: dict,
    trainer_parameters: dict,
    fit_parameters: dict
) -> None:
    preprocessed_manifest_path = dataset_dir / constants.manifests.preprocessed
    with open(preprocessed_manifest_path, "r") as preprocessed_manifest_file:
        preprocessed_manifest_content = json.load(preprocessed_manifest_file)
        preprocessed_subjects = preprocessed_manifest_content.get("subjects", [])

    train_subjects, test_subjects = train_test_split(preprocessed_subjects, **datasplit_parameters)

    train_patches = utils.collect_patches(train_subjects, utils.patch_subject_non_overlapping, patcher_parameters)

    train_set = SegmentationPatchDataset(train_patches, **dataset_parameters)
    train_sampler = EqualBatchSampler(train_patches, **sampler_parameters)
    train_loader = DataLoader(
        dataset=train_set,
        batch_sampler=train_sampler,
        **dataloader_parameters
    )

    patcher_parameters["augmentation_factor"] = 1 # test set doesn't need augmentation

    test_patches = utils.collect_patches(test_subjects, utils.patch_subject_non_overlapping, patcher_parameters)

    test_set = SegmentationPatchDataset(test_patches)
    test_sampler = BatchSampler(SequentialSampler(test_set), **sampler_parameters)
    test_loader = DataLoader(
        dataset=test_set,
        batch_sampler=test_sampler,
        **dataloader_parameters
    )

    model = CandidateDetector(**model_parameters)
    task = SegmentationTask()
    trainer = Trainer(model, task, **trainer_parameters)

    trainer.fit(train_loader, test_loader, **fit_parameters)