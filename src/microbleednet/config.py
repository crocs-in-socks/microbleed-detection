import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FrozenConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DataConfig(FrozenConfig):
    modality: Literal["T2*-GRE", "SWI"]
    require_masks: bool = True


class PreprocessingConfig(FrozenConfig):
    canonical_orientation: bool = True
    extract_brain: bool = True
    bias_field_correction: bool = True
    invert_volume: bool = True
    inpaint_vessels: bool = True


class AugmentationConfig(FrozenConfig):
    translation_offset_range: tuple[int, int] = (-15, 15)
    noise_variance_range: tuple[float, float] = (0.01, 0.04)
    blur_sigma_range: tuple[float, float] = (0.1, 0.2)


class ModelConfig(FrozenConfig):
    initial_channels: int = Field(gt=0)
    input_channels: int = Field(gt=0)
    output_classes: int = Field(ge=2)
    dropout_rate: float = Field(ge=0.0, lt=1.0)


class DetectorConfig(ModelConfig):
    patch_size: int = Field(gt=0)
    augmentation_factor: int = Field(ge=1)
    probability_threshold: float = Field(ge=0.0, le=1.0)


class TeacherConfig(ModelConfig):
    patch_size: int = Field(gt=0)
    augmentation_factor: int = Field(ge=1)


class StudentConfig(ModelConfig):
    patch_size: int = Field(gt=0)
    augmentation_factor: int = Field(ge=1)
    probability_threshold: float = Field(ge=0.0, le=1.0)
    temperature: float = Field(gt=0.0)
    alpha: float = Field(ge=0.0, le=1.0)
    beta: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_loss_weights(self) -> "StudentConfig":
        if not math.isclose(self.alpha + self.beta, 1.0):
            raise ValueError("alpha and beta must sum to 1.0")
        return self


class TrainerConfig(FrozenConfig):
    learning_rate: float = Field(gt=0.0)
    adam_epsilon: float = Field(gt=0.0)
    batch_size: int = Field(ge=2)
    max_epochs: int = Field(gt=0)
    patience: int = Field(ge=0)
    learning_rate_factor: float = Field(gt=0.0, lt=1.0)
    learning_rate_period: int = Field(gt=0)
    minimum_learning_rate: float = Field(gt=0.0)


class PostprocessingConfig(FrozenConfig):
    minimum_volume_mm3: float = Field(gt=0.0)
    maximum_eccentricity: float = Field(ge=0.0, le=1.0)
    minimum_boundary_distance_voxels: float = Field(ge=0.0)


class EvaluationConfig(FrozenConfig):
    detector_thresholds: tuple[float, ...]
    discriminator_thresholds: tuple[float, ...]

    @model_validator(mode="after")
    def validate_thresholds(self) -> "EvaluationConfig":
        thresholds = self.detector_thresholds + self.discriminator_thresholds
        if not thresholds or any(value < 0.0 or value > 1.0 for value in thresholds):
            raise ValueError("evaluation thresholds must be in the range [0, 1]")
        return self


class DetectorRunConfig(FrozenConfig):
    data: DataConfig
    preprocessing: PreprocessingConfig
    augmentation: AugmentationConfig
    detector: DetectorConfig
    trainer: TrainerConfig
    postprocessing: PostprocessingConfig
    evaluation: EvaluationConfig


class TeacherRunConfig(DetectorRunConfig):
    teacher: TeacherConfig


class StudentRunConfig(DetectorRunConfig):
    teacher: TeacherConfig
    student: StudentConfig


class InferenceConfig(FrozenConfig):
    data: DataConfig
    preprocessing: PreprocessingConfig
    detector: DetectorConfig
    student: StudentConfig
    postprocessing: PostprocessingConfig
    evaluation: EvaluationConfig