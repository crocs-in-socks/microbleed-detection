import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import constants


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = constants.common.losses.dice.default.smooth):
        super().__init__()
        self.smooth = smooth

    def forward(self, prediction, target):
        prediction = prediction.reshape(prediction.size(0), -1)
        target = target.reshape(target.size(0), -1)

        intersection = (prediction * target).sum(dim=1)
        union = prediction.sum(dim=1) + target.sum(dim=1)

        dice_coefficient = (2.0 * intersection + self.smooth) / (union + self.smooth)

        return 1.0 - dice_coefficient.mean()


class KnowledgeDistillationLoss(nn.Module):
    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, teacher_logits, student_logits):
        teacher_predictions = F.softmax(teacher_logits / self.temperature, dim=1)
        student_predictions = F.log_softmax(student_logits / self.temperature, dim=1)

        return F.kl_div(student_predictions, teacher_predictions, reduction="batchmean") # batchmean is for standard KL divergence
    

class DetectorLoss(nn.Module):
    def __init__(self, dice_smooth=constants.common.losses.dice.default.smooth):
        super().__init__()
        self.dice_loss = DiceLoss(smooth=dice_smooth)
        self.register_buffer("class_weights", torch.tensor([1.0, 10.0]))

    def forward(self, logits, target):
        expected_shape = (logits.size(0), *logits.shape[2:])
        if target.shape != expected_shape:
            raise ValueError(f"target must have shape {expected_shape}")
        if target.dtype != torch.long:
            raise ValueError("class-index targets must use torch.int64")
        if torch.any((target < 0) | (target > 1)):
            raise ValueError("class-index targets must contain only 0 and 1")
        prediction = F.softmax(logits, dim=1)
        dice_loss = self.dice_loss(prediction[:, 1], target == 1)
        cross_entropy_loss = F.cross_entropy(
            logits, target.to(logits.device),
            weight=self.class_weights.to(logits.device, logits.dtype),
            reduction="mean",
        )
        return dice_loss + cross_entropy_loss

class DiscriminatorTeacherLoss(nn.Module):
    """
    dice loss + weighted voxel-wise cross entropy loss + binary cross entropy
    """
    def __init__(self, dice_smooth=constants.common.losses.dice.default.smooth):
        super().__init__()
        self.segmentation_loss = DetectorLoss(dice_smooth)
        self.classification_loss = nn.CrossEntropyLoss()

    def forward(self, classification_logits, classification_target, segmentation_logits, segmentation_target):
        if classification_target.shape != (classification_logits.size(0),):
            raise ValueError("classification targets must have shape (batch,)")
        if classification_target.dtype != torch.long:
            raise ValueError("classification targets must use torch.int64")
        segmentation_loss = self.segmentation_loss(segmentation_logits, segmentation_target)
        classification_loss = self.classification_loss(
            classification_logits, classification_target.to(classification_logits.device)
        )

        return segmentation_loss + classification_loss

class DiscriminatorStudentLoss(nn.Module):
    """
    weight_alpha * cross entropy loss + weight_beta * knowledge distillation loss
    """
    def __init__(
        self,
        alpha: float = constants.common.losses.discriminator.student.default.alpha,
        beta: float = constants.common.losses.discriminator.student.default.beta,
        temperature: float = constants.common.losses.discriminator.student.default.temperature
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.knowledge_distillation_loss = KnowledgeDistillationLoss(temperature)
    
    def forward(self, teacher_logits, student_logits, target):
        if target.shape != (student_logits.size(0),) or target.dtype != torch.long:
            raise ValueError("classification targets must be int64 with shape (batch,)")
        cross_entropy_loss = self.cross_entropy_loss(student_logits, target.to(student_logits.device))
        knowledge_distillation_loss = self.knowledge_distillation_loss(teacher_logits, student_logits)

        return self.alpha * cross_entropy_loss + self.beta * knowledge_distillation_loss
