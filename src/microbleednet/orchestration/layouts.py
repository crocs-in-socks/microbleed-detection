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
    """Relative paths of an indexed dataset directory."""

    raw_manifest: Path = Field(
        default=Path("manifests/raw.json"),
        description="Raw dataset manifest written by index-data, relative to the "
        "dataset directory.",
    )
