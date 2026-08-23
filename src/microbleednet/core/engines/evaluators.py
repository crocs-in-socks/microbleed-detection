import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from microbleednet.core.common.tasks import BaseTask


class Evaluator:
    def __init__(self, model: nn.Module, device: torch.device, task: BaseTask):
        self.model = model.to(device)
        self.device = device
        self.task = task

        self.use_amp = False
        self.amp_dtype = torch.float16

    @staticmethod
    def _batch_size(batch) -> int:
        for value in batch:
            if isinstance(value, torch.Tensor):
                return value.shape[0]
        raise ValueError("validation batch contains no tensor with a sample dimension")

    def evaluate(self, dataloader: DataLoader) -> float:
        self.model.eval()
        running_loss = 0.0
        sample_count = 0

        with torch.no_grad():
            for batch in dataloader:
                loss = self.task.validation_step(self.model, self.device, batch)
                batch_size = self._batch_size(batch)
                running_loss += loss.item() * batch_size
                sample_count += batch_size

        if sample_count == 0:
            raise ValueError("cannot evaluate an empty DataLoader")
        return running_loss / sample_count
