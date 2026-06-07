import torch
import torch.nn as nn
from torch.amp import autocast
from torch.utils.data import DataLoader

from microbleednet.core.engines.tasks import BaseTask


class Evaluator:
    def __init__(
        self,
        model = nn.Module,
        device = torch.device,
        task = BaseTask
    ):
        self.model = model
        self.device = device
        self.task = task

        use_amp = (device.type == "cuda")
        self.amp_dtype = torch.float16 if use_amp else torch.bfloat16

    def evaluate(self, dataloader: DataLoader) -> float:
        self.model.eval()
        running_loss = 0.0

        with torch.no_grad():
            for batch in dataloader:

                with autocast(device_type=self.device.type, dtype=self.amp_dtype):
                    loss = self.task.validation_step(self.model, self.device, batch)

                running_loss += loss.item()
        
        average_loss = running_loss  /len(dataloader)
        
        return average_loss