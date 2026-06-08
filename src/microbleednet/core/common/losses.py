import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, prediction, target):
        prediction = prediction.reshape(prediction.size(0), -1)
        target = target.reshape(target.size(0), -1)

        intersection = (prediction * target).sum(dim=1)
        union = prediction.sum(dim=1) + target.sum(dim=1)

        dice_coefficient = (2.0 * intersection + self.smooth) / (union + self.smooth)

        return 1.0 - dice_coefficient.mean()


class DetectorLoss(nn.Module):
    """
    dice loss + weighted voxel-wise cross entropy loss
    """
    def __init__(self, dice_smooth=1.0):
        super().__init__()
        self.dice_loss = DiceLoss(smooth=dice_smooth)
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction="none") # no reduction so that we can apply weights

    def forward(self, logits, target, voxel_weights=None):
        prediction = F.softmax(logits, dim=1)
        dice_loss = self.dice_loss(prediction[:, 1], target[:, 1])

        cross_entropy_loss = self.cross_entropy_loss(logits, target)

        if voxel_weights is not None:
            voxel_weights = voxel_weights.to(logits.device)
            cross_entropy_loss = cross_entropy_loss * voxel_weights

        cross_entropy_loss = cross_entropy_loss.mean()

        return dice_loss + cross_entropy_loss

class DiscriminatorTeacherLoss(nn.Module):
    """
    dice loss + weighted voxel-wise cross entropy loss + binary cross entropy
    """
    def __init__(self, dice_smooth=1.0):
        super().__init__()
        self.segmentation_loss = DetectorLoss(dice_smooth)
        self.classification_loss = nn.BCEWithLogitsLoss()

    def forward(self, classification_logits, classification_target, segmentation_logits, segmentation_target, voxel_weights=None):
        segmentation_loss = self.segmentation_loss(segmentation_logits, segmentation_target, voxel_weights)
        classification_loss = self.classification_loss(classification_logits, classification_target)

        return segmentation_loss + classification_loss
