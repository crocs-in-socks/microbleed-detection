import pytest
import torch

from microbleednet.core.common.losses import (
    DetectorLoss,
    DiscriminatorStudentLoss,
    DiscriminatorTeacherLoss,
)
from microbleednet.core.common.models import (
    CandidateDiscriminatorStudent,
    CandidateDiscriminatorTeacher,
    CandidateDetector,
)


def test_detector_loss_matches_weighted_cross_entropy_and_dice() -> None:
    logits = torch.tensor([[[[2.0, 0.0]], [[0.0, 2.0]]]])
    target = torch.tensor([[[0, 1]]], dtype=torch.long)
    loss = DetectorLoss()(logits, target)

    probabilities = torch.softmax(logits, dim=1)
    expected_cross_entropy = -(torch.log(probabilities[0, 0, 0, 0]) + 10 * torch.log(probabilities[0, 1, 0, 1])) / 11
    foreground = probabilities[:, 1]
    expected_dice = 1 - (2 * foreground[0, 0, 1] + 1) / (foreground.sum() + 1 + 1)
    assert torch.allclose(loss, expected_cross_entropy + expected_dice)


def test_detector_loss_rejects_invalid_targets() -> None:
    logits = torch.zeros((1, 2, 2, 2, 2))
    wrong_shape = torch.zeros((1, 1, 2, 2, 2), dtype=torch.long)
    invalid_values = torch.full((1, 2, 2, 2), 2, dtype=torch.long)
    with pytest.raises(ValueError, match="shape"):
        DetectorLoss()(logits, wrong_shape)
    with pytest.raises(ValueError, match="0 and 1"):
        DetectorLoss()(logits, invalid_values)


def test_detector_loss_is_finite_for_empty_foreground() -> None:
    loss = DetectorLoss()(torch.zeros((2, 2, 2, 2, 2)), torch.zeros((2, 2, 2, 2), dtype=torch.long))
    assert torch.isfinite(loss)


def test_teacher_loss_is_segmentation_plus_classification() -> None:
    segmentation_logits = torch.zeros((2, 2, 2, 2, 2))
    segmentation_target = torch.zeros((2, 2, 2, 2), dtype=torch.long)
    classification_logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    classification_target = torch.tensor([0, 1], dtype=torch.long)
    criterion = DiscriminatorTeacherLoss()
    expected = criterion.segmentation_loss(segmentation_logits, segmentation_target)
    expected = expected + criterion.classification_loss(classification_logits, classification_target)
    assert torch.allclose(criterion(classification_logits, classification_target, segmentation_logits, segmentation_target), expected)


def test_distillation_loss_is_weighted_and_identical_logits_are_zero() -> None:
    teacher = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    student = teacher.clone().requires_grad_()
    target = torch.tensor([0, 1], dtype=torch.long)
    criterion = DiscriminatorStudentLoss()
    assert criterion.knowledge_distillation_loss(teacher, student).abs() < 1e-6
    loss = criterion(teacher, student, target)
    expected = criterion.alpha * criterion.cross_entropy_loss(student, target)
    assert torch.allclose(loss, expected)
    loss.backward()
    assert torch.isfinite(student.grad).all()


def test_models_support_paper_patch_geometry() -> None:
    detector = CandidateDetector(2, 2, 16)
    detector_output = detector(torch.randn(2, 2, 48, 48, 48))
    assert detector_output.shape == (2, 2, 48, 48, 48)

    teacher = CandidateDiscriminatorTeacher(2, 2, 16, 0.1)
    segmentation, classification = teacher(torch.randn(2, 2, 24, 24, 24))
    assert segmentation.shape == (2, 2, 24, 24, 24)
    assert classification.shape == (2, 2)

    student = CandidateDiscriminatorStudent(2, 2, 16, 0.1)
    student_output = student(torch.randn(2, 2, 24, 24, 24))
    assert student_output.shape == (2, 2)
