"""On-disk directory layouts the orchestration layer reads and writes within.

These are filesystem contracts, not user-tunable config: they define where each
pipe places and finds artifacts inside a dataset or experiment directory, so the
pipes use the default instances. Keeping the path knowledge here means the layout
is defined in one place rather than scattered as string literals across the
pipes. Every path is relative to its root directory; callers join the results
onto ``dataset_dir`` / ``experiment_dir``.
"""

from pathlib import Path

from pydantic import Field

from ..core.datamodels import FrozenModel


class DatasetLayout(FrozenModel):
    """Relative paths and filename suffixes of an indexed dataset directory."""

    raw_manifest: Path = Field(
        default=Path("manifests/raw.json"),
        description="Raw dataset manifest written by index-data, relative to the "
        "dataset directory.",
    )
    preprocessed_manifest: Path = Field(
        default=Path("manifests/preprocessed.json"),
        description="Preprocessed dataset manifest written by preprocess, relative "
        "to the dataset directory.",
    )
    preprocessed_volumes_dir: Path = Field(
        default=Path("preprocessed/volumes"),
        description="Directory for preprocessed volumes, relative to the dataset "
        "directory.",
    )
    preprocessed_masks_dir: Path = Field(
        default=Path("preprocessed/masks"),
        description="Directory for preprocessed masks, relative to the dataset "
        "directory.",
    )
    volume_suffix: str = Field(
        default="_volume.nii.gz",
        description="Filename suffix for a preprocessed volume.",
    )
    mask_suffix: str = Field(
        default="_mask.nii.gz",
        description="Filename suffix for a preprocessed mask.",
    )


class ExperimentLayout(FrozenModel):
    """Relative paths within a training experiment directory.

    Everything is relative to the experiment directory; callers join the results
    onto ``experiment_dir``. The train/patch/checkpoint paths are parameterized
    by stage (detector/teacher/student), so they are methods rather than fixed
    fields.
    """

    train_dir: Path = Field(
        default=Path("train"),
        description="Root for per-stage training artifacts, relative to the "
        "experiment directory.",
    )
    manifests_dir: Path = Field(
        default=Path("manifests"),
        description="Directory for run manifests, relative to the experiment "
        "directory.",
    )
    latest_checkpoint_name: Path = Field(
        default=Path("latest_model.pth"),
        description="Filename of the most-recent checkpoint within a stage's "
        "checkpoint directory.",
    )
    best_checkpoint_name: Path = Field(
        default=Path("best_model.pth"),
        description="Filename of the best-validation checkpoint within a stage's "
        "checkpoint directory.",
    )

    def stage_dir(self, stage: str) -> Path:
        """Root holding one training stage's artifacts."""
        return self.train_dir / stage

    def checkpoint_dir(self, stage: str) -> Path:
        """Directory holding a stage's checkpoints."""
        return self.stage_dir(stage) / "checkpoints"

    def latest_checkpoint(self, stage: str) -> Path:
        """A stage's most-recent checkpoint file."""
        return self.checkpoint_dir(stage) / self.latest_checkpoint_name

    def best_checkpoint(self, stage: str) -> Path:
        """A stage's best-validation checkpoint file."""
        return self.checkpoint_dir(stage) / self.best_checkpoint_name

    def train_patch_dir(self, stage: str) -> Path:
        """Directory of a stage's materialized training patches."""
        return self.stage_dir(stage) / "patches" / "train"

    def validation_patch_dir(self, stage: str) -> Path:
        """Directory of a stage's materialized validation patches."""
        return self.stage_dir(stage) / "patches" / "validation"

    def stage_manifest(self, stage: str) -> Path:
        """Per-stage training manifest, e.g. manifests/detector.json."""
        return self.manifests_dir / f"{stage}.json"

    def split_manifest(self) -> Path:
        """Train/validation split manifest."""
        return self.manifests_dir / "split.json"
