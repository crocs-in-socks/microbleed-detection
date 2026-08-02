from pathlib import Path
from typing import cast

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from microbleednet.core import utils
from microbleednet.core.common.models import (
    CandidateDetector,
    CandidateDiscriminatorTeacher,
)
from microbleednet.core.common.tasks import BaseTask
from microbleednet.core.engines.trainers import Trainer


class EmptyTask(BaseTask):
    def training_step(self, model, device, batch):
        return torch.tensor(0.0)

    def validation_step(self, model, device, batch):
        return torch.tensor(0.0)


def test_checkpoint_round_trip_and_best_latest_match(tmp_path: Path) -> None:
    model = nn.Linear(2, 2)
    trainer = Trainer(
        model,
        EmptyTask(),
        torch.device("cpu"),
        {"lr": 1e-3, "clip_norm": 1.0},
        {"gamma": 0.1, "step_size": 2, "minimum_learning_rate": 1e-6},
        tmp_path,
        compile_model=False,
    )
    trainer.save_checkpoint(3, True)
    latest = torch.load(tmp_path / "latest_model.pth", weights_only=False)
    best = torch.load(tmp_path / "best_model.pth", weights_only=False)
    assert latest["format_version"] == utils.CHECKPOINT_FORMAT_VERSION
    assert latest["model_state_dict"].keys() == best["model_state_dict"].keys()

    restored = nn.Linear(2, 2)
    utils.load_model_weights(restored, torch.device("cpu"), tmp_path / "best_model.pth")
    assert torch.allclose(model.weight, restored.weight)


def test_missing_checkpoint_is_fatal(tmp_path: Path) -> None:
    model = nn.Linear(2, 2)
    checkpoint_path = tmp_path / "missing.pth"
    with pytest.raises(FileNotFoundError):
        utils.load_model_weights(model, torch.device("cpu"), checkpoint_path)


def _make_trainer(
    tmp_path: Path, optimizer_parameters: dict, scheduler_parameters: dict
) -> Trainer:
    return Trainer(
        nn.Linear(2, 2),
        EmptyTask(),
        torch.device("cpu"),
        optimizer_parameters,
        scheduler_parameters,
        tmp_path,
        compile_model=False,
    )


def test_trainer_honors_configured_learning_rate(tmp_path: Path) -> None:
    # The Trainer must use the learning rate it is handed, not a hardcoded one.
    trainer = _make_trainer(
        tmp_path,
        {"lr": 5e-4, "eps": 1e-4},
        {"gamma": 0.1, "step_size": 2, "minimum_learning_rate": 1e-6},
    )
    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(5e-4)


def test_trainer_scheduler_reproduces_paper_step_decay(tmp_path: Path) -> None:
    # With paper values the multiplier is gamma**(epoch // step_size), floored
    # at minimum_learning_rate / initial_learning_rate.
    trainer = _make_trainer(
        tmp_path,
        {"lr": 1e-3, "eps": 1e-4},
        {"gamma": 0.1, "step_size": 2, "minimum_learning_rate": 1e-6},
    )
    lambda_fn = trainer.scheduler.lr_lambdas[0]
    assert lambda_fn(0) == pytest.approx(1.0)
    assert lambda_fn(1) == pytest.approx(1.0)
    assert lambda_fn(2) == pytest.approx(0.1)
    assert lambda_fn(3) == pytest.approx(0.1)
    assert lambda_fn(4) == pytest.approx(0.01)
    # Floor at 1e-6 / 1e-3 = 1e-3; deeper epochs never drop below it.
    assert lambda_fn(100) == pytest.approx(1e-3)


def test_trainer_scheduler_honors_configured_decay(tmp_path: Path) -> None:
    # Non-paper values drive the schedule too, proving nothing is hardcoded.
    trainer = _make_trainer(
        tmp_path,
        {"lr": 1e-2, "eps": 1e-4},
        {"gamma": 0.5, "step_size": 3, "minimum_learning_rate": 1e-3},
    )
    lambda_fn = trainer.scheduler.lr_lambdas[0]
    assert lambda_fn(2) == pytest.approx(1.0)
    assert lambda_fn(3) == pytest.approx(0.5)
    assert lambda_fn(6) == pytest.approx(0.25)
    # Floor at 1e-3 / 1e-2 = 0.1.
    assert lambda_fn(60) == pytest.approx(0.1)


class _TrainableTask(BaseTask):
    """Minimal task whose loss is differentiable w.r.t. the model."""

    def training_step(self, model, device, batch):
        return model(batch["input"]).pow(2).mean()

    def validation_step(self, model, device, batch):
        return model(batch["input"]).pow(2).mean()


def test_trainer_fit_honors_max_epochs_beyond_100(tmp_path: Path) -> None:
    # The old code capped epochs at min(n_epochs, 100); max_epochs now wins.
    trainer = Trainer(
        nn.Linear(2, 2),
        _TrainableTask(),
        torch.device("cpu"),
        {"lr": 1e-3, "eps": 1e-4},
        {"gamma": 0.1, "step_size": 2, "minimum_learning_rate": 1e-6},
        tmp_path,
        compile_model=False,
    )
    trainer.patience = 10_000  # disable early stopping for this check

    class _OneSampleLoader:
        def __iter__(self):
            yield {"input": torch.zeros(1, 2), "target": torch.zeros(1)}

    trainer.fit(
        cast(DataLoader, _OneSampleLoader()),
        cast(DataLoader, _OneSampleLoader()),
        n_epochs=105,
    )
    assert len(trainer.epoch_records) == 105
    assert trainer.epoch_records[-1]["epoch"] == 104


def test_detector_initializes_teacher_without_overwriting_classifier() -> None:
    detector = CandidateDetector(2, 2, 2)
    teacher = CandidateDiscriminatorTeacher(2, 2, 2, 0.1)
    classifier_before = {
        key: value.clone() for key, value in teacher.classifier.state_dict().items()
    }
    utils.initialize_teacher_from_detector(detector, teacher)
    for key, value in detector.feature_extractor.state_dict().items():
        assert torch.equal(value, teacher.feature_extractor.state_dict()[key])
    for key, value in detector.segmentor.state_dict().items():
        assert torch.equal(value, teacher.segmentor.state_dict()[key])
    for key, value in classifier_before.items():
        assert torch.equal(value, teacher.classifier.state_dict()[key])
    assert all(parameter.requires_grad for parameter in teacher.parameters())
