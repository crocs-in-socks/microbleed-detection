"""On-disk directory layouts the orchestration layer reads and writes within.

These are filesystem contracts, not user-tunable config: they define where each
pipe places and finds artifacts inside a dataset directory, so the pipes use
the default instance. Keeping the path knowledge here means the layout is
defined in one place rather than scattered as string literals across the
pipes.
"""

from pathlib import Path

from pydantic import Field

from ..core.datamodels import FrozenModel


class DatasetLayout(FrozenModel):
    """Paths within an indexed dataset directory."""

    dataset_dir: Path = Field(
        description="Root directory for this dataset's artifacts."
    )

    raw_manifest: Path = Field(
        default=Path("manifests/raw.json"),
        description="Raw dataset manifest written by index-data.",
    )
    preprocessed_manifest: Path = Field(
        default=Path("manifests/preprocessed.json"),
        description=(
            "Preprocessed dataset manifest written by preprocess."
        ),
    )
    preprocessed_volumes_dir: Path = Field(
        default=Path("preprocessed/volumes"),
        description=(
            "Directory for preprocessed volumes."
        ),
    )
    preprocessed_masks_dir: Path = Field(
        default=Path("preprocessed/masks"),
        description=(
            "Directory for preprocessed masks."
        ),
    )
    volume_suffix: str = Field(
        default=".nii.gz",
        description="Filename suffix for a preprocessed volume.",
    )
    mask_suffix: str = Field(
        default=".nii.gz",
        description="Filename suffix for a preprocessed mask.",
    )

    def raw_manifest_path(self) -> Path:
        return self.dataset_dir / self.raw_manifest

    def preprocessed_manifest_path(self) -> Path:
        return self.dataset_dir / self.preprocessed_manifest

    def preprocessed_volumes_path(self) -> Path:
        return self.dataset_dir / self.preprocessed_volumes_dir

    def preprocessed_masks_path(self) -> Path:
        return self.dataset_dir / self.preprocessed_masks_dir
