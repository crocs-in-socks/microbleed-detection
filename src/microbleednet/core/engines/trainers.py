from pathlib import Path

import torch
import torch.nn as nn
from torch import optim
from torch.amp import autocast
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from microbleednet.core.engines.tasks import BaseTask
from microbleednet.core.engines.evaluators import Evaluator


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        optimizer_parameters: dict,
        scheduler_parameters: dict,
        task: BaseTask,
        checkpoint_dir: Path
    ):
        self.model = model
        self.device = device
        self.task = task
        
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = checkpoint_dir

        self.clip_norm = optimizer_parameters.pop("clip_norm", 1.0)

        self.optimizer = optim.Adam(self.model.parameters(), **optimizer_parameters)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, **scheduler_parameters)

        use_amp = (device.type == "cuda")
        self.amp_dtype = torch.float16 if use_amp else torch.bfloat16
        self.scaler = GradScaler(device.type, enabled=use_amp)

        self.best_val_loss = float('inf')

        self.evaluator = Evaluator(self.model, self.device, self.task)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, num_epochs: int):
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.evaluator.evaluate(val_loader)

            print(f"Epoch {epoch+1:03d}/{num_epochs:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            is_best = val_loss < self.best_val_loss 
            if is_best:
                self.best_val_loss = val_loss
                print("--> Checkpoint saved!")

            self.save_checkpoint(epoch, is_best)

    def train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        running_loss = 0.0

        for batch in dataloader:
            self.optimizer.zero_grad()
            with autocast(device_type=self.device.type, dtype=self.amp_dtype):
                loss = self.task.training_step(self.model, self.device, batch)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            clip_grad_norm_(self.model.parameters(), max_norm=self.clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item()

        average_loss = running_loss / len(dataloader)
        self.scheduler.step()

        return average_loss

    def save_checkpoint(self, epoch: int, is_best: bool) -> None:
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_val_loss": self.best_val_loss
        }

        latest_path = self.checkpoint_dir / "latest_model.pth"
        torch.save(state, latest_path)

        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(self.model.state_dict(), best_path)

