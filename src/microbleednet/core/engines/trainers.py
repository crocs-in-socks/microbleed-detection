import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch import optim
from torch.amp import autocast
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from .. import constants
from microbleednet.core import utils
from microbleednet.core.common.tasks import BaseTask
from microbleednet.core.engines.evaluators import Evaluator


logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        task: BaseTask,
        device: torch.device,
        optimizer_parameters: dict,
        scheduler_parameters: dict,
        checkpoint_dir: Path,
        compile_model: bool = constants.engines.trainers.default.compile_model,
        stage: str = "training",
        model_config: dict | None = None,
        provenance: dict | None = None,
        use_amp: bool = False,
        min_delta: float = 0.0,
        patience: int = 20,
    ):
        self.model = model
        self.device = device
        self.task = task
        self.stage = stage
        self.model_config = model_config or {}
        self.provenance = provenance or {}
        self.use_amp = bool(use_amp and device.type == "cuda")
        self.min_delta = min_delta
        self.patience = patience
        self.epochs_without_improvement = 0
        self.epoch_records = []
        
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if compile_model and hasattr(torch, "compile"):
            logger.info("Compiling model for faster training...")
            self.model = torch.compile(model)
        else:
            self.model = model

        optimizer_parameters = dict(optimizer_parameters)
        optimizer_parameters["lr"] = 1e-3
        self.clip_norm = optimizer_parameters.pop("clip_norm", constants.engines.trainers.default.clip_norm)
        self.optimizer = optim.Adam(self.model.parameters(), **optimizer_parameters)
        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lambda completed_epochs: max(
                1e-6 / optimizer_parameters["lr"],
                0.1 ** (completed_epochs // 2),
            ),
        )

        self.amp_dtype = torch.float16
        self.scaler = GradScaler(device.type, enabled=self.use_amp)

        self.best_val_loss = float('inf')
        self.model = self.model.to(self.device)

        self.evaluator = Evaluator(self.model, self.device, self.task)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        n_epochs: int,
        checkpoint_path: Path = constants.engines.trainers.default.checkpoint_path,
        weights_only: bool = constants.engines.trainers.default.weights_only,
    ):
        start_epoch = 0
        if checkpoint_path:
            start_epoch = self.load_checkpoint(checkpoint_path, weights_only)

        for epoch in range(start_epoch, min(n_epochs, 100)):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.evaluator.evaluate(val_loader)
            is_best = val_loss < self.best_val_loss - self.min_delta
            if is_best:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1

            self.epoch_records.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "best": is_best})
            self.save_checkpoint(epoch, is_best)
            if self.epochs_without_improvement >= self.patience:
                break

    def train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        running_loss = 0.0
        sample_count = 0

        for batch in dataloader:
            self.optimizer.zero_grad(set_to_none=True)
            if self.use_amp:
                with autocast(device_type=self.device.type, dtype=self.amp_dtype):
                    loss = self.task.training_step(self.model, self.device, batch)
            else:
                loss = self.task.training_step(self.model, self.device, batch)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            clip_grad_norm_(self.model.parameters(), max_norm=self.clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item()
            sample_count += self._batch_size(batch)

        if sample_count == 0:
            raise ValueError("cannot train on an empty DataLoader")
        average_loss = running_loss / sample_count
        self.scheduler.step()

        return average_loss

    @staticmethod
    def _batch_size(batch) -> int:
        for value in batch.values():
            if isinstance(value, torch.Tensor):
                return value.shape[0]
        raise ValueError("training batch contains no tensor with a sample dimension")

    def save_checkpoint(self, epoch: int, is_best: bool) -> None:
        model_state = utils.unwrap_model(self.model).state_dict()

        state = {
            "format_version": utils.CHECKPOINT_FORMAT_VERSION,
            "stage": self.stage,
            "model_config": self.model_config,
            "provenance": self.provenance,
            "epoch": epoch,
            "model_state_dict": model_state,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "epochs_without_improvement": self.epochs_without_improvement,
            "epoch_records": self.epoch_records,
        }

        latest_path = self.checkpoint_dir / constants.engines.trainers.latest_checkpoint_path
        torch.save(state, latest_path)

        if is_best:
            best_path = self.checkpoint_dir / constants.engines.trainers.best_checkpoint_path
            torch.save(state, best_path)

    def load_checkpoint(self, checkpoint_path: Path, weights_only: bool) -> int:

        checkpoint = utils.load_model_weights(self.model, self.device, checkpoint_path)

        if weights_only:
            logger.info("Loaded model weights only. Starting from epoch 0.")
            return 0

        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self.best_val_loss = checkpoint["best_val_loss"]
        self.epochs_without_improvement = checkpoint.get("epochs_without_improvement", 0)
        self.epoch_records = checkpoint.get("epoch_records", [])
        
        start_epoch = checkpoint.get("epoch", -1) + 1
        logger.info(
            "Successfully restored full state. Resuming from epoch %s.", start_epoch
        )
        
        return start_epoch
