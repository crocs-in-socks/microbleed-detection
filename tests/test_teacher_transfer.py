import torch

from microbleednet.core import utils
from microbleednet.core.common.models import CandidateDiscriminatorTeacher, CandidateDetector


def test_teacher_transfer_updates_all_parameter_groups() -> None:
    torch.manual_seed(7)
    detector = CandidateDetector(1, 2, 1)
    teacher = CandidateDiscriminatorTeacher(1, 2, 1, 0.0)
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

    before = {
        name: parameter.detach().clone() for name, parameter in teacher.named_parameters()
    }
    optimizer = torch.optim.Adam(teacher.parameters(), lr=1e-3)
    inputs = torch.randn(2, 1, 24, 24, 24)
    segmentation_target = torch.zeros(2, 24, 24, 24, dtype=torch.long)
    classification_target = torch.zeros(2, dtype=torch.long)
    segmentation_logits, classification_logits = teacher(inputs)
    loss = (
        torch.nn.functional.cross_entropy(segmentation_logits, segmentation_target)
        + torch.nn.functional.cross_entropy(classification_logits, classification_target)
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    changed = {
        name: not torch.equal(value, before[name])
        for name, value in teacher.named_parameters()
    }
    assert any(name.startswith("feature_extractor.") and value for name, value in changed.items())
    assert any(name.startswith("segmentor.") and value for name, value in changed.items())
    assert any(name.startswith("classifier.") and value for name, value in changed.items())
