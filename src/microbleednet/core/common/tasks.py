from collections.abc import Mapping

import torch
import torch.nn as nn

from microbleednet.core.common import losses
from microbleednet.core.transforms import frst

# A collated training batch as produced by the patch datasets: a mapping from
# a fixed set of tensor names to tensors. Every dataset supplies ``volume`` (the
# input image patch); segmentation datasets add ``mask`` and classification
# datasets add ``label``. A task reads only the keys its dataset supplies.
TrainingBatch = Mapping[str, torch.Tensor]


class BaseTask:
    def training_step(
        self, model: nn.Module, device: torch.device, batch: TrainingBatch
    ) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement the training_step method.")

    def validation_step(
        self, model: nn.Module, device: torch.device, batch: TrainingBatch
    ) -> torch.Tensor:
        raise NotImplementedError(
            "Subclasses must implement the validation_step method."
        )


class SegmentationTask(BaseTask):
    def __init__(self):
        self.criterion = losses.DetectorLoss()

    def training_step(
        self, model: nn.Module, device: torch.device, batch: TrainingBatch
    ) -> torch.Tensor:
        volume = batch["volume"].to(device, dtype=torch.float)
        mask = batch["mask"].to(device, dtype=torch.long)

        logits = model(frst.prepend_frst_channel(volume))
        loss = self.criterion(logits, mask)

        return loss

    def validation_step(
        self, model: nn.Module, device: torch.device, batch: TrainingBatch
    ) -> torch.Tensor:
        return self.training_step(model, device, batch)


class SegmentationClassificationTask(BaseTask):
    def __init__(self):
        self.criterion = losses.DiscriminatorTeacherLoss()

    def training_step(
        self, model: nn.Module, device: torch.device, batch: TrainingBatch
    ) -> torch.Tensor:
        volume = batch["volume"].to(device, dtype=torch.float)
        mask = batch["mask"].to(device, dtype=torch.long)
        label = batch["label"].to(device, dtype=torch.long)

        volume = frst.prepend_frst_channel(volume)
        segmentation_logits, classification_logits = model(volume)
        loss = self.criterion(classification_logits, label, segmentation_logits, mask)

        return loss

    def validation_step(
        self, model: nn.Module, device: torch.device, batch: TrainingBatch
    ) -> torch.Tensor:
        return self.training_step(model, device, batch)


class KnowledgeDistillationClassificationTask(BaseTask):
    def __init__(self, teacher_model, alpha: float, beta: float, temperature: float):
        self.teacher_model = teacher_model
        self.criterion = losses.DiscriminatorStudentLoss(
            alpha=alpha, beta=beta, temperature=temperature
        )

        self.teacher_model.eval()

    def training_step(
        self, model: nn.Module, device: torch.device, batch: TrainingBatch
    ) -> torch.Tensor:
        volume = batch["volume"].to(device, dtype=torch.float)
        label = batch["label"].to(device, dtype=torch.long)

        volume = frst.prepend_frst_channel(volume)

        with torch.no_grad():
            _, teacher_logits = self.teacher_model(volume)

        student_logits = model(volume)

        loss = self.criterion(teacher_logits, student_logits, label)

        return loss

    def validation_step(
        self, model: nn.Module, device: torch.device, batch: TrainingBatch
    ) -> torch.Tensor:
        return self.training_step(model, device, batch)
