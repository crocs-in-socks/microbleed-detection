import logging
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
import torch.nn as nn
from torch import optim
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from microbleednet.core import io, utils
from microbleednet.core.common.tasks import BaseTask
from microbleednet.core.datamodels import TrainerConfig
from microbleednet.core.engines.evaluators import Evaluator

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CheckpointConfig:
    """Where a ``Trainer`` writes checkpoints, plus the stage stamped into them.

    Bundles the persistence concern into one value the pipeline builds from its
    experiment layout, so ``Trainer`` takes a single ``checkpoints`` argument
    rather than a directory and two loose filenames. The filename defaults mirror
    ``orchestration.layouts.ExperimentLayout`` (the layer that owns the on-disk names,
    which ``core`` cannot import); the pipeline overrides them from the layout.
    """

    directory: Path
    stage: str = "training"
    latest_name: Path = Path("latest_model.pth")
    best_name: Path = Path("best_model.pth")

    @property
    def latest_path(self) -> Path:
        return self.directory / self.latest_name

    @property
    def best_path(self) -> Path:
        return self.directory / self.best_name


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        task: BaseTask,
        device: torch.device,
        config: TrainerConfig,
        checkpoints: CheckpointConfig,
    ):
        self.model = model
        self.device = device
        self.task = task
        self.config = config
        self.checkpoints = checkpoints
        self.use_amp = bool(config.use_amp and device.type == "cuda")
        self.epochs_without_improvement = 0
        self.epoch_records = []

        self.checkpoints.directory.mkdir(parents=True, exist_ok=True)

        if config.compile_model and hasattr(torch, "compile"):
            logger.info("Compiling model for faster training...")
            self.model = cast(nn.Module, torch.compile(model))
        else:
            self.model = model

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            eps=config.adam_epsilon,
            weight_decay=config.weight_decay,
        )

        # Step decay: every ``learning_rate_period`` epochs the learning rate is
        # multiplied by ``learning_rate_factor``, floored at
        # ``minimum_learning_rate``. With the paper's values (lr=1e-3, factor=0.1,
        # period=2, floor=1e-6) this reproduces the original schedule exactly.
        # LambdaLR scales the *initial* LR by the returned factor, so the floor is
        # expressed as its ratio to the initial LR.
        floor_ratio = config.minimum_learning_rate / config.learning_rate
        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lambda epoch: max(
                floor_ratio,
                config.learning_rate_factor ** (epoch // config.learning_rate_period),
            ),
        )

        self.amp_dtype = torch.float16
        self.scaler = GradScaler(device.type, enabled=self.use_amp)

        self.best_val_loss = float("inf")
        self.model = self.model.to(self.device)

        self.evaluator = Evaluator(self.model, self.device, self.task)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        n_epochs: int | None = None,
        checkpoint_path: Path | None = None,
        weights_only: bool = False,
    ) -> None:
        if n_epochs is None:
            n_epochs = self.config.max_epochs

        start_epoch = 0
        if checkpoint_path:
            start_epoch = self.load_checkpoint(checkpoint_path, weights_only)

        for epoch in range(start_epoch, n_epochs):
            if self._run_epoch(train_loader, val_loader, epoch):
                break

    def _run_epoch(
        self, train_loader: DataLoader, val_loader: DataLoader, epoch: int
    ) -> bool:
        train_loss = self.train_epoch(train_loader)
        val_loss = self.evaluator.evaluate(val_loader)
        is_best = val_loss < self.best_val_loss - self.config.minimum_improvement
        if is_best:
            self.best_val_loss = val_loss
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

        self.epoch_records.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "best": is_best,
            }
        )
        self.save_checkpoint(epoch, is_best)
        return self.epochs_without_improvement >= self.config.patience

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
            clip_grad_norm_(
                self.model.parameters(), max_norm=self.config.gradient_clip_norm
            )
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
        for value in batch:
            if isinstance(value, torch.Tensor):
                return value.shape[0]
        raise ValueError("training batch contains no tensor with a sample dimension")

    def save_checkpoint(self, epoch: int, is_best: bool) -> None:
        model_state = utils.unwrap_model(self.model).state_dict()

        state = {
            "format_version": io.CHECKPOINT_FORMAT_VERSION,
            "stage": self.checkpoints.stage,
            "epoch": epoch,
            "model_state_dict": model_state,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "epochs_without_improvement": self.epochs_without_improvement,
            "epoch_records": self.epoch_records,
        }

        io.save_checkpoint_atomic(state, self.checkpoints.latest_path)

        if is_best:
            io.save_checkpoint_atomic(state, self.checkpoints.best_path)

    def load_checkpoint(self, checkpoint_path: Path, weights_only: bool) -> int:

        checkpoint = io.load_model_weights(self.model, self.device, checkpoint_path)

        if weights_only:
            logger.info("Loaded model weights only. Starting from epoch 0.")
            return 0

        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self.best_val_loss = checkpoint["best_val_loss"]
        self.epochs_without_improvement = checkpoint.get(
            "epochs_without_improvement", 0
        )
        self.epoch_records = checkpoint.get("epoch_records", [])

        start_epoch = checkpoint.get("epoch", -1) + 1
        logger.info(
            "Successfully restored full state. Resuming from epoch %s.", start_epoch
        )

        return start_epoch
