import torch

from microbleednet.core.common import losses
from microbleednet.core.transforms import frst


class BaseTask:
    def training_step(self, batch):
        raise NotImplementedError("Subclasses must implement the training_step method.")

    def validation_step(self, batch):
        raise NotImplementedError("Subclasses must implement the validation_step method.")

class SegmentationTask(BaseTask):
    def __init__(self):
        self.criterion = losses.DetectorLoss()

    def training_step(self, model, device, batch):
        x = batch.get("x").to(device, dtype=torch.float)
        y = batch.get("y").to(device, dtype=torch.float)
        weights = batch.get("weights").to(device, dtype=torch.float)

        x_frst = frst.apply(x)
        x = torch.cat((x, x_frst), dim=1) # Shape: (Batch, 2, H, W, D)

        predictions = model(x)
        loss = self.criterion(predictions, y, weights)

        return loss

    def validation_step(self, model, device, batch):
        return self.training_step(model, device, batch)


class SegmentationClassificationTask(BaseTask):
    def __init__(self):
        self.criterion = losses.DiscriminatorTeacherLoss()
    
    def training_step(self, model, device, batch):
        volume = batch.get("volume").to(device, dtype=torch.float)
        mask = batch.get("mask").to(device, dtype=torch.float)
        weights = batch.get("weights").to(device, dtype=torch.float)
        label = batch.get("label").to(device, dtype=torch.float)

        volume_frst = frst.apply(volume)
        volume = torch.cat((volume, volume_frst), dim=1) # Shape: (Batch, 2, H, W, D)

        segmentation_predictions, classification_predictions = model(volume)
        loss = self.criterion(classification_predictions, label, segmentation_predictions, mask, weights)

        return loss

    def validation_step(self, model, device, batch):
        return self.training_step(model, device, batch)