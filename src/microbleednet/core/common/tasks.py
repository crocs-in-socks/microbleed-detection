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
        x = batch["x"].to(device, dtype=torch.float)
        y = batch["y"].to(device, dtype=torch.long)

        x_frst = frst.apply(x)
        x = torch.cat((x, x_frst), dim=1) # Shape: (Batch, 2, H, W, D)

        logits = model(x)
        loss = self.criterion(logits, y)

        return loss

    def validation_step(self, model, device, batch):
        return self.training_step(model, device, batch)


class SegmentationClassificationTask(BaseTask):
    def __init__(self):
        self.criterion = losses.DiscriminatorTeacherLoss()
    
    def training_step(self, model, device, batch):
        volume = batch["volume"].to(device, dtype=torch.float)
        mask = batch["mask"].to(device, dtype=torch.long)
        label = batch["label"].to(device, dtype=torch.long)

        volume_frst = frst.apply(volume)
        volume = torch.cat((volume, volume_frst), dim=1) # Shape: (Batch, 2, H, W, D)

        segmentation_logits, classification_logits = model(volume)
        loss = self.criterion(classification_logits, label, segmentation_logits, mask)

        return loss

    def validation_step(self, model, device, batch):
        return self.training_step(model, device, batch)

class KnowledgeDistillationClassificationTask(BaseTask):
    def __init__(self, teacher_model):
        self.teacher_model = teacher_model
        self.criterion = losses.DiscriminatorStudentLoss()

        self.teacher_model.eval()
    
    def training_step(self, student_model, device, batch):
        x = batch["x"].to(device, dtype=torch.float)
        y = batch["y"].to(device, dtype=torch.long)

        x_frst = frst.apply(x)
        x = torch.cat((x, x_frst), dim=1)

        with torch.no_grad():
            _, teacher_logits = self.teacher_model(x)
        
        student_logits = student_model(x)

        loss = self.criterion(teacher_logits, student_logits, y)

        return loss

    def validation_step(self, student_model, device, batch):
        return self.training_step(student_model, device, batch)