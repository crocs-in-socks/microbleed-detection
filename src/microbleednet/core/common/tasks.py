from typing import Generic, TypeVar

import torch
import torch.nn as nn

from microbleednet.core.common import losses
from microbleednet.core.dataloading.datasets import (
    ClassificationBatch,
    SegmentationBatch,
    SegmentationClassificationBatch,
)
from microbleednet.core.transforms import frst

# Each task consumes exactly one batch shape, so BaseTask is generic over it:
# the type parameter lets a subclass narrow ``batch`` to its own batch type
# without violating the override contract.
BatchT = TypeVar("BatchT", bound=tuple)


class BaseTask(Generic[BatchT]):
    def training_step(
        self, model: nn.Module, device: torch.device, batch: BatchT
    ) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement the training_step method.")

    def validation_step(
        self, model: nn.Module, device: torch.device, batch: BatchT
    ) -> torch.Tensor:
        raise NotImplementedError(
            "Subclasses must implement the validation_step method."
        )


class SegmentationTask(BaseTask[SegmentationBatch]):
    def __init__(self):
        self.criterion = losses.DetectorLoss()

    def training_step(
        self, model: nn.Module, device: torch.device, batch: SegmentationBatch
    ) -> torch.Tensor:
        volume = batch.volume.to(device, dtype=torch.float)
        mask = batch.mask.to(device, dtype=torch.long)

        logits = model(frst.prepend_frst_channel(volume))
        loss = self.criterion(logits, mask)

        return loss

    def validation_step(
        self, model: nn.Module, device: torch.device, batch: SegmentationBatch
    ) -> torch.Tensor:
        return self.training_step(model, device, batch)


class SegmentationClassificationTask(BaseTask[SegmentationClassificationBatch]):
    def __init__(self):
        self.criterion = losses.DiscriminatorTeacherLoss()

    def training_step(
        self,
        model: nn.Module,
        device: torch.device,
        batch: SegmentationClassificationBatch,
    ) -> torch.Tensor:
        volume = batch.volume.to(device, dtype=torch.float)
        mask = batch.mask.to(device, dtype=torch.long)
        label = batch.label.to(device, dtype=torch.long)

        volume = frst.prepend_frst_channel(volume)
        segmentation_logits, classification_logits = model(volume)
        loss = self.criterion(classification_logits, label, segmentation_logits, mask)

        return loss

    def validation_step(
        self,
        model: nn.Module,
        device: torch.device,
        batch: SegmentationClassificationBatch,
    ) -> torch.Tensor:
        return self.training_step(model, device, batch)


class KnowledgeDistillationClassificationTask(BaseTask[ClassificationBatch]):
    def __init__(self, teacher_model, alpha: float, beta: float, temperature: float):
        self.teacher_model = teacher_model
        self.criterion = losses.DiscriminatorStudentLoss(
            alpha=alpha, beta=beta, temperature=temperature
        )

        # The teacher only emits soft targets; it is never trained. Put it in
        # eval mode (stable, dropout-free logits) and freeze its parameters so it
        # cannot accumulate gradients even if a caller runs it outside no_grad.
        self.teacher_model.eval()
        for parameter in self.teacher_model.parameters():
            parameter.requires_grad_(False)

    def training_step(
        self, model: nn.Module, device: torch.device, batch: ClassificationBatch
    ) -> torch.Tensor:
        volume = batch.volume.to(device, dtype=torch.float)
        label = batch.label.to(device, dtype=torch.long)

        volume = frst.prepend_frst_channel(volume)

        with torch.no_grad():
            _, teacher_logits = self.teacher_model(volume)

        student_logits = model(volume)

        loss = self.criterion(teacher_logits, student_logits, label)

        return loss

    def validation_step(
        self, model: nn.Module, device: torch.device, batch: ClassificationBatch
    ) -> torch.Tensor:
        return self.training_step(model, device, batch)
