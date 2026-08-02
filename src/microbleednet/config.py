import math
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FrozenConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PreprocessingConfig(FrozenConfig):
    canonical_orientation: bool = Field(
        default=True,
        description="Reorient to canonical (RAS-like) axis order before processing.",
    )
    extract_brain: bool = Field(
        default=True,
        description="Skull-strip with FSL BET (requires 'bet' on PATH).",
    )
    bias_field_correction: bool = Field(
        default=True,
        description=(
            "Apply SimpleITK N4 bias-field correction (paper deviation from FSL FAST)."
        ),
    )
    invert_volume: bool = Field(
        default=True,
        description="Invert normalized intensities so microbleeds appear bright.",
    )
    inpaint_vessels: bool = Field(
        default=True,
        description=(
            "Remove and inpaint vessel-like structures before candidate detection."
        ),
    )


class AugmentationConfig(FrozenConfig):
    translation_offset_range: tuple[int, int] = Field(
        default=(-15, 15),
        description=(
            "Random translation range in voxels, applied to both axes. "
            "Paper: (-15, 15)."
        ),
    )
    noise_variance_range: tuple[float, float] = Field(
        default=(0.01, 0.04),
        description=(
            "Gaussian noise variance range (intensity units). Paper: (0.01, 0.04)."
        ),
    )
    blur_sigma_range: tuple[float, float] = Field(
        default=(0.1, 0.2),
        description="Gaussian blur sigma range in voxels. Paper: (0.1, 0.2).",
    )


class ModelConfig(FrozenConfig):
    initial_channels: int = Field(
        gt=0,
        description="Initial convolution channel depth of the 3D U-Net. Paper: 64.",
    )
    input_channels: int = Field(
        gt=0,
        description="Number of input channels (image plus FRST). Paper: 2.",
    )
    output_classes: int = Field(
        ge=2,
        description="Number of segmentation/classification output classes. Paper: 2.",
    )
    dropout_rate: float = Field(
        ge=0.0,
        lt=1.0,
        description="Dropout probability in the classifier head, in [0, 1).",
    )


class DetectorConfig(ModelConfig):
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


class TeacherConfig(ModelConfig):
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


class StudentConfig(ModelConfig):
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


class TrainerConfig(FrozenConfig):
    learning_rate: float = Field(
        gt=0.0,
        description="Initial Adam learning rate. Paper: 1e-3.",
    )
    adam_epsilon: float = Field(
        gt=0.0,
        description="Adam epsilon for numerical stability. Paper: 1e-4.",
    )
    batch_size: int = Field(
        ge=2,
        description=(
            "Training batch size (must be at least 2 for balanced sampling). Paper: 8."
        ),
    )
    max_epochs: int = Field(
        gt=0,
        description="Maximum number of training epochs. Paper: 100.",
    )
    patience: int = Field(
        ge=0,
        description=(
            "Early-stopping patience in epochs without validation improvement. "
            "Paper: 20."
        ),
    )
    learning_rate_factor: float = Field(
        gt=0.0,
        lt=1.0,
        description=(
            "Multiplicative learning-rate decay factor, in (0, 1). Paper: 0.1."
        ),
    )
    learning_rate_period: int = Field(
        gt=0,
        description="Number of epochs between learning-rate decays. Paper: 2.",
    )
    minimum_learning_rate: float = Field(
        gt=0.0,
        description="Learning-rate floor below which decay stops. Paper: 1e-6.",
    )


class PostprocessingConfig(FrozenConfig):
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
    minimum_boundary_distance_voxels: float = Field(
        ge=0.0,
        description=(
            "Reject candidates nearer than this distance (in voxels) to the "
            "brain boundary."
        ),
    )


class PreprocessCommandConfig(FrozenConfig):
    dataset_dir: Path = Field(
        description=(
            "Indexed dataset directory containing manifests/raw.json; "
            "preprocessed artifacts are written under it."
        ),
    )
    preprocessing: PreprocessingConfig = Field(
        default_factory=PreprocessingConfig,
        description=(
            "Preprocessing steps to apply. Defaults follow the paper "
            "(all steps enabled)."
        ),
    )


class EvaluationSubject(FrozenConfig):
    subject_id: str = Field(description="Identifier used in the evaluation report.")
    prediction_path: Path = Field(
        description="Source-space predicted lesion mask for this subject.",
    )
    reference_path: Path = Field(
        description="Source-space reference (ground-truth) lesion mask.",
    )
    probability_path: Path | None = Field(
        default=None,
        description=(
            "Source-space predicted probability map for this subject. Required "
            "only for the FROC sweep; omit it for binary-mask metrics alone."
        ),
    )


class EvaluateCommandConfig(FrozenConfig):
    subjects: tuple[EvaluationSubject, ...] = Field(
        min_length=1,
        description="Subjects to evaluate; each pairs a prediction with a reference.",
    )
    output_dir: Path = Field(
        description="Directory to write evaluation.json and the metric CSVs.",
    )
    metadata: dict[str, str] | None = Field(
        default=None,
        description="Optional free-form metadata recorded in the report header.",
    )
    froc_thresholds: tuple[float, ...] | None = Field(
        default=None,
        description=(
            "Probability thresholds to sweep for FROC, each in [0, 1]. When set, "
            "every subject must also supply probability_path; the sweep is "
            "written to froc.json and froc.csv."
        ),
    )

    @model_validator(mode="after")
    def validate_froc(self) -> "EvaluateCommandConfig":
        if self.froc_thresholds is None:
            return self
        if not self.froc_thresholds or any(
            value < 0.0 or value > 1.0 for value in self.froc_thresholds
        ):
            raise ValueError("froc_thresholds must be nonempty and each in [0, 1]")
        missing = [
            subject.subject_id
            for subject in self.subjects
            if subject.probability_path is None
        ]
        if missing:
            raise ValueError(
                "froc_thresholds requires probability_path for every subject; "
                f"missing for: {missing}"
            )
        return self


class DataSplitConfig(FrozenConfig):
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


class TrainCommandConfig(FrozenConfig):
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


class InferCommandConfig(FrozenConfig):
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
    preprocessing: PreprocessingConfig = Field(
        default_factory=PreprocessingConfig,
        description="Preprocessing steps applied to the input volume before inference.",
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
