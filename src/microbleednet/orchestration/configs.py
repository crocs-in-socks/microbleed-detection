"""Top-level command configs the orchestration layer consumes.

Each pipe's ``execute()`` takes one of these config objects whole. They live
here because the orchestration layer owns them; the CLI imports them to parse
TOML into already-validated objects, which keeps the dependency direction
``cli -> orchestration`` intact. Loading a config from a TOML file is a CLI
concern (``cli/utils.py``).
"""

import re
from pathlib import Path

from pydantic import Field, model_validator

from ..core.datamodels import FrozenModel, Modality

# Token a volume/mask filename pattern must contain exactly once; the text it
# matches becomes the subject ID. Shared by the index-data pipeline (which
# splits filenames on it) and IndexDataConfig (which validates its presence).
SUBJECT_ID_PLACEHOLDER = "{subject_id}"

# A source_id namespaces subject IDs and becomes part of on-disk paths, so it is
# restricted to filesystem-safe characters.
SOURCE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


class IndexDataConfig(FrozenModel):
    dataset_dir: Path = Field(
        description="Directory to which the indexed dataset manifests are written.",
    )
    input_dir: Path = Field(
        description="Input directory containing the volumes to index.",
    )
    volume_pattern: str = Field(
        description=(
            "Naming pattern of volumes in the input directory. "
            f"Must contain the '{SUBJECT_ID_PLACEHOLDER}' placeholder."
        ),
    )
    label_dir: Path = Field(
        description="Label directory containing the masks to index.",
    )
    mask_pattern: str = Field(
        description=(
            "Naming pattern of masks in the label directory. "
            f"Must contain the '{SUBJECT_ID_PLACEHOLDER}' placeholder."
        ),
    )
    source_id: str = Field(
        description=(
            "Namespace prepended to each subject ID as "
            "'{source_id}_{subject_id}'. Use it to keep subjects unique when "
            "indexing several sources into one dataset. "
            "Allowed characters: letters, digits, '-', '_'."
        ),
    )
    modality: Modality = Field(
        default="T2*-GRE",
        description="Imaging modality of this source's volumes.",
    )

    @model_validator(mode="after")
    def validate_source_id(self) -> "IndexDataConfig":
        if not SOURCE_ID_PATTERN.match(self.source_id):
            raise ValueError(
                "source_id may contain only letters, digits, '-', and '_'"
            )
        return self

    @model_validator(mode="after")
    def validate_patterns(self) -> "IndexDataConfig":
        for name, pattern in (
            ("volume_pattern", self.volume_pattern),
            ("mask_pattern", self.mask_pattern),
        ):
            if pattern is not None and pattern.count(SUBJECT_ID_PLACEHOLDER) < 1:
                raise ValueError(
                    f"{name} must contain the '{SUBJECT_ID_PLACEHOLDER}' "
                    "placeholder at least once"
                )
        return self

    @model_validator(mode="after")
    def validate_directories(self) -> "IndexDataConfig":
        if not self.input_dir.is_dir():
            raise ValueError(f"input_dir does not exist: {self.input_dir}")
        if not self.label_dir.is_dir():
            raise ValueError(f"label_dir does not exist: {self.label_dir}")
        return self


class PreprocessConfig(FrozenModel):
    dataset_dir: Path = Field(
        description="Indexed dataset directory containing manifests/raw.json."
    )

    @model_validator(mode="after")
    def validate_dataset_dir(self) -> "PreprocessConfig":
        if not self.dataset_dir.is_dir():
            raise ValueError(f"dataset_dir does not exist: {self.dataset_dir}")
        return self
