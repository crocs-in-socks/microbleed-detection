from pathlib import Path

import pytest
import torch
from torch import nn

from microbleednet.core import utils
from microbleednet.core.common.models import CandidateDiscriminatorTeacher, CandidateDetector
from microbleednet.core.engines.trainers import Trainer


class EmptyTask:
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
        {"milestones": [1], "gamma": 0.1},
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


def test_detector_initializes_teacher_without_overwriting_classifier() -> None:
    detector = CandidateDetector(2, 2, 2)
    teacher = CandidateDiscriminatorTeacher(2, 2, 2, 0.1)
    classifier_before = {key: value.clone() for key, value in teacher.classifier.state_dict().items()}
    utils.initialize_teacher_from_detector(detector, teacher)
    for key, value in detector.feature_extractor.state_dict().items():
        assert torch.equal(value, teacher.feature_extractor.state_dict()[key])
    for key, value in detector.segmentor.state_dict().items():
        assert torch.equal(value, teacher.segmentor.state_dict()[key])
    for key, value in classifier_before.items():
        assert torch.equal(value, teacher.classifier.state_dict()[key])
    assert all(parameter.requires_grad for parameter in teacher.parameters())
