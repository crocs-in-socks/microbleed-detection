"""Top-level command configs the orchestration layer consumes.

Each pipe's ``execute()`` takes one of these config objects whole. They live
here because the orchestration layer owns them; the CLI imports them back so its
command table can validate configs against them, which keeps the dependency
direction ``cli -> orchestration`` intact. Loading a config from a TOML file is a
CLI concern (``cli/utils.py``); the pipes only ever receive an already-validated
object.

The per-stage model configs (the ``DetectorConfig`` family) also live here: each
wraps a core ``ModelArchitecture`` under an ``architecture`` block and adds the
orchestration-only knobs (patching, thresholds, distillation weights) that
``core`` never sees. The reusable building blocks they compose come from
``core/datamodels.py`` (architectures, preprocessing, augmentation, trainer,
and ``FrozenModel``).
"""

import math
import re
from pathlib import Path

from pydantic import Field, model_validator

from ..core.datamodels import (
    AugmentationConfig,
    ClassifierArchitecture,
    FrozenModel,
    ModelArchitecture,
    Modality,
    TrainerConfig,
)

# Token a volume/mask filename pattern must contain exactly once; the text it
# matches becomes the subject ID. Shared by the index-data pipeline (which
# splits filenames on it) and IndexDataConfig (which validates its presence).
SUBJECT_ID_PLACEHOLDER = "{subject_id}"

# A source_id namespaces subject IDs and becomes part of on-disk paths, so it is
# restricted to filesystem-safe characters.
SOURCE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


class PostprocessingConfig(FrozenModel):
    minimum_volume_mm3: float = Field(
        gt=0.0,
        description=(
            "Reject candidate components smaller than this physical volume in mm^3. "
            "Paper: 2.5."
        ),
    )
    maximum_eccentricity: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Reject candidates more elongated than this eccentricity, in [0, 1]. "
            "Paper: 0.2."
        ),
    )
    minimum_boundary_distance_mm: float = Field(
        ge=0.0,
        description=(
            "Reject candidates nearer than this physical distance in mm to the "
            "brain boundary. Paper: 5."
        ),
    )


class DataSplitConfig(FrozenModel):
    test_size: float = Field(
        gt=0.0,
        lt=1.0,
        description=(
            "Fraction of subjects held out for validation/early-stopping, "
            "in (0, 1). Paper: 0.2."
        ),
    )
    random_state: int = Field(
        default=42,
        description="Seed for the train/validation split. Paper: 42.",
    )
    shuffle: bool = Field(
        default=True,
        description="Shuffle subjects before splitting.",
    )


class DetectorConfig(FrozenModel):
    """Candidate-detector stage config: a segmentation architecture plus the
    pipeline knobs (patching, candidate threshold) that ``core`` never sees."""

    architecture: ModelArchitecture = Field(
        description="3D U-Net construction contract passed straight to the model.",
    )
    patch_size: int = Field(
        gt=0,
        description=(
            "Cubic training patch edge length in voxels for the detector. Paper: 48."
        ),
    )
    augmentation_factor: int = Field(
        ge=1,
        description=(
            "Multiplier for detector training data via augmentation. Paper: 10."
        ),
    )
    probability_threshold: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Detector foreground probability threshold for candidates, in [0, 1]."
        ),
    )


class TeacherConfig(FrozenModel):
    """Discriminator-teacher stage config: a classifier architecture plus the
    teacher's patching knobs."""

    architecture: ClassifierArchitecture = Field(
        description="3D U-Net + classifier construction contract for the teacher.",
    )
    patch_size: int = Field(
        gt=0,
        description=(
            "Cubic training patch edge length in voxels for the teacher. Paper: 24."
        ),
    )
    augmentation_factor: int = Field(
        ge=1,
        description="Multiplier for teacher training data via augmentation. Paper: 5.",
    )


class StudentConfig(FrozenModel):
    """Discriminator-student stage config: a classifier architecture plus the
    student's patching, acceptance-threshold, and distillation knobs."""

    architecture: ClassifierArchitecture = Field(
        description="3D U-Net + classifier construction contract for the student.",
    )
    patch_size: int = Field(
        gt=0,
        description=(
            "Cubic training patch edge length in voxels for the student. Paper: 24."
        ),
    )
    augmentation_factor: int = Field(
        ge=1,
        description="Multiplier for student training data via augmentation. Paper: 5.",
    )
    probability_threshold: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Student (discriminator) acceptance probability threshold, in [0, 1]."
        ),
    )
    temperature: float = Field(
        gt=0.0,
        description="Distillation softmax temperature T. Paper: 4.",
    )
    alpha: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Weight of the supervised loss term. Paper: 0.4. "
            "Must satisfy alpha + beta = 1."
        ),
    )
    beta: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Weight of the distillation loss term. Paper: 0.6. "
            "Must satisfy alpha + beta = 1."
        ),
    )

    @model_validator(mode="after")
    def validate_loss_weights(self) -> "StudentConfig":
        if not math.isclose(self.alpha + self.beta, 1.0):
            raise ValueError("alpha and beta must sum to 1.0")
        return self


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
    label_dir: Path | None = Field(
        default=None,
        description="Label directory containing masks to index.",
    )
    mask_pattern: str | None = Field(
        default=None,
        description=(
            "Naming pattern of masks in the label directory. "
            f"Must contain the '{SUBJECT_ID_PLACEHOLDER}' placeholder. "
            "Required with label_dir."
        ),
    )
    require_masks: bool = Field(
        default=True,
        description="Require every volume to have a matching mask.",
    )
    source_id: str | None = Field(
        default=None,
        description=(
            "Optional namespace prepended to each subject ID as "
            "'{source_id}_{subject_id}'. Use it to keep subjects unique when "
            "indexing several sources into one dataset. "
            "Allowed characters: letters, digits, '-', '_'."
        ),
    )

    @model_validator(mode="after")
    def validate_source_id(self) -> "IndexDataConfig":
        if self.source_id is not None and not SOURCE_ID_PATTERN.match(self.source_id):
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
            if pattern is not None and pattern.count(SUBJECT_ID_PLACEHOLDER) != 1:
                raise ValueError(
                    f"{name} must contain the '{SUBJECT_ID_PLACEHOLDER}' "
                    "placeholder exactly once"
                )
        if self.label_dir is not None and self.mask_pattern is None:
            raise ValueError("mask_pattern is required when label_dir is provided")
        if self.label_dir is None and self.require_masks:
            raise ValueError(
                "label_dir is required when require_masks is true; set "
                "require_masks = false to index volumes without masks"
            )
        return self


class PreprocessConfig(FrozenModel):
    dataset_dir: Path = Field(
        description=(
            "Indexed dataset directory containing manifests/raw.json; "
            "preprocessed artifacts are written under it."
        ),
    )
    preprocessing: Modality = Field(
        default="T2*-GRE",
        description=(
            "Fixed paper preprocessing pipeline, with modality controlling "
            "bias correction and intensity inversion."
        ),
    )


class TrainConfig(FrozenModel):
    dataset_dir: Path = Field(
        description=(
            "Indexed dataset directory containing manifests/preprocessed.json."
        ),
    )
    experiment_dir: Path = Field(
        description=(
            "Directory to write per-stage patches, checkpoints, and manifests."
        ),
    )
    detector: DetectorConfig = Field(
        description="Candidate-detector model configuration (its own block).",
    )
    teacher: TeacherConfig = Field(
        description="Discriminator-teacher model configuration (its own block).",
    )
    student: StudentConfig = Field(
        description="Discriminator-student model configuration (its own block).",
    )
    trainer: TrainerConfig = Field(
        description="Optimizer/epoch/early-stopping hyperparameters shared by stages.",
    )
    augmentation: AugmentationConfig = Field(
        default_factory=AugmentationConfig,
        description="Augmentation ranges applied to training patches.",
    )
    datasplit: DataSplitConfig = Field(
        default_factory=lambda: DataSplitConfig(test_size=0.2),
        description="Train/validation split settings.",
    )
    device: str = Field(
        default="cpu",
        description="Torch device string, e.g. 'cpu' or 'cuda'.",
    )
    seed: int = Field(
        default=1,
        description=(
            "Experiment seed. Seeds Python/NumPy/PyTorch and derives worker "
            "seeds, and is recorded in the run provenance."
        ),
    )
    num_workers: int = Field(
        default=0,
        ge=0,
        description="DataLoader worker processes. 0 keeps loading in-process.",
    )
    pin_memory: bool = Field(
        default=False,
        description="Pin DataLoader host memory (only helps with CUDA).",
    )


class InferConfig(FrozenModel):
    volume_path: Path = Field(
        description="Input source-space volume to run inference on."
    )
    output_dir: Path = Field(
        description=(
            "Directory to write the predicted mask, probability map, and records."
        ),
    )
    detector_checkpoint: Path = Field(
        description="Trained candidate-detector checkpoint (.pth).",
    )
    student_checkpoint: Path = Field(
        description="Trained candidate-discriminator student checkpoint (.pth).",
    )
    detector: DetectorConfig = Field(
        description="Detector model configuration (its own parameter block).",
    )
    student: StudentConfig = Field(
        description="Student model configuration (its own parameter block).",
    )
    preprocessing: Modality = Field(
        default="T2*-GRE",
        description="Fixed preprocessing pipeline applied before inference.",
    )
    device: str = Field(
        default="cpu",
        description="Torch device string, e.g. 'cpu' or 'cuda'.",
    )
    seed: int = Field(
        default=1,
        description=(
            "Experiment seed. Seeds Python/NumPy/PyTorch and is recorded in the "
            "run provenance for traceability."
        ),
    )
    patch_batch_size: int = Field(
        default=8,
        gt=0,
        description="Number of candidate patches scored per student forward pass.",
    )
    subject_id: str | None = Field(
        default=None,
        description="Identifier for outputs; defaults to the volume filename stem.",
    )
    postprocessing: PostprocessingConfig | None = Field(
        default=None,
        description=(
            "Morphological candidate filtering applied to the student-accepted "
            "mask before it is written. Omit to write the raw accepted mask."
        ),
    )


class EvaluateConfig(FrozenModel):
    dataset_dir: Path = Field(
        description=(
            "Indexed dataset directory containing manifests/raw.json; every "
            "subject's volume is inferred and scored against its reference mask."
        ),
    )
    output_dir: Path = Field(
        description="Directory to write predictions, evaluation.json, and the CSVs.",
    )
    detector: DetectorConfig = Field(
        description="Detector model configuration (its own parameter block).",
    )
    student: StudentConfig = Field(
        description="Student model configuration (its own parameter block).",
    )
    experiment_dir: Path | None = Field(
        default=None,
        description=(
            "Training experiment directory to resolve the best detector and "
            "student checkpoints from. Provide this, or both explicit checkpoint "
            "paths below."
        ),
    )
    detector_checkpoint: Path | None = Field(
        default=None,
        description=(
            "Explicit detector checkpoint (.pth). Overrides experiment_dir; use "
            "it for pretrained weights not produced by this pipeline."
        ),
    )
    student_checkpoint: Path | None = Field(
        default=None,
        description=(
            "Explicit student checkpoint (.pth). Overrides experiment_dir; use "
            "it for pretrained weights not produced by this pipeline."
        ),
    )
    preprocessing: Modality = Field(
        default="T2*-GRE",
        description="Fixed preprocessing pipeline applied before inference.",
    )
    postprocessing: PostprocessingConfig | None = Field(
        default=None,
        description=(
            "Morphological candidate filtering applied to each student-accepted "
            "mask before scoring. Omit to score the raw accepted mask."
        ),
    )
    device: str = Field(
        default="cpu",
        description="Torch device string, e.g. 'cpu' or 'cuda'.",
    )
    seed: int = Field(
        default=1,
        description=(
            "Experiment seed. Seeds Python/NumPy/PyTorch and is recorded in the "
            "run provenance for traceability."
        ),
    )
    patch_batch_size: int = Field(
        default=8,
        gt=0,
        description="Number of candidate patches scored per student forward pass.",
    )
    metadata: dict[str, str] | None = Field(
        default=None,
        description="Optional free-form metadata recorded in the report header.",
    )
    froc_thresholds: tuple[float, ...] | None = Field(
        default=None,
        description=(
            "Probability thresholds to sweep for FROC, each in [0, 1]. When set, "
            "the sweep is written to froc.json and froc.csv."
        ),
    )

    @model_validator(mode="after")
    def validate_checkpoints(self) -> "EvaluateConfig":
        explicit = (self.detector_checkpoint, self.student_checkpoint)
        if any(path is None for path in explicit) and self.experiment_dir is None:
            raise ValueError(
                "provide experiment_dir, or both detector_checkpoint and "
                "student_checkpoint"
            )
        return self

    @model_validator(mode="after")
    def validate_froc(self) -> "EvaluateConfig":
        if self.froc_thresholds is None:
            return self
        if not self.froc_thresholds or any(
            value < 0.0 or value > 1.0 for value in self.froc_thresholds
        ):
            raise ValueError("froc_thresholds must be nonempty and each in [0, 1]")
        return self
