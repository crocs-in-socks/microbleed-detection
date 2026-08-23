"""Datamodels owned by the ``core`` layer.

Two kinds of thing live here, both intrinsic to ``core``'s domain logic:

- the configs that describe core operations: the model-architecture contracts
  the ``nn.Module`` constructors consume directly (``ModelArchitecture``,
    ``ClassifierArchitecture``), plus ``AugmentationConfig`` and
    ``TrainerConfig``, and
- the records core produces from image data (``PreprocessResult`` and the
  geometry/transform types it carries), plus the array aliases they use.

They are colocated so ``core`` is self-sufficient: nothing here reaches up into
``orchestration`` or ``cli``. ``FrozenModel`` is the app-wide Pydantic base used
by core and orchestration models. The orchestration configs that compose these
(``DetectorConfig``/``TeacherConfig``/``StudentConfig``) live in
``orchestration/configs.py``, since only ``orchestration`` consumes them.
"""

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

FloatArray = NDArray[np.floating]
IntArray = NDArray[np.integer]


class FrozenModel(BaseModel):
    """Immutable, unknown-field-rejecting base for application data models."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelArchitecture(FrozenModel):
    """Construction contract for a segmentation network (e.g. the detector).

    Maps one-to-one onto the ``nn.Module`` constructor. Input channels are fixed
    by the architecture (image + FRST) and defaulted on the model, so they are
    intentionally not a field here.
    """

    initial_channels: int = Field(
        gt=0,
        description="Initial convolution channel depth of the 3D U-Net. Paper: 64.",
    )
    output_classes: int = Field(
        ge=2,
        description="Number of segmentation/classification output classes. Paper: 2.",
    )


class ClassifierArchitecture(ModelArchitecture):
    """Architecture for a model with a classifier head (teacher and student).

    Extends ``ModelArchitecture`` with the head's dropout. The detector has no
    classifier, so it uses the base and never carries a dropout knob.
    """

    dropout_rate: float = Field(
        ge=0.0,
        lt=1.0,
        description="Dropout probability in the classifier head, in [0, 1).",
    )


Modality = Literal["T2*-GRE", "SWI", "QSM"]


class AugmentationConfig(FrozenModel):
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


class TrainerConfig(FrozenModel):
    """Complete construction contract for ``core.engines.trainers.Trainer``.

    Groups the paper hyperparameters (learning rate, schedule, epochs, patience)
    with the execution knobs the trainer needs to run (AMP, compilation, gradient
    clipping, early-stopping delta). Lives in ``core`` because ``Trainer``
    consumes it directly; ``orchestration`` composes it into ``TrainConfig``.
    """

    learning_rate: float = Field(
        gt=0.0,
        description="Initial Adam learning rate. Paper: 1e-3.",
    )
    adam_epsilon: float = Field(
        gt=0.0,
        description="Adam epsilon for numerical stability. Paper: 1e-4.",
    )
    weight_decay: float = Field(
        default=0.0,
        ge=0.0,
        description="Adam L2 weight-decay coefficient. Default: 0.",
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
    use_amp: bool = Field(
        default=False,
        description=(
            "Enable mixed-precision autocast (only takes effect on CUDA devices)."
        ),
    )
    compile_model: bool = Field(
        default=True,
        description="Compile the model with torch.compile before training.",
    )
    minimum_improvement: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Minimum validation-loss decrease that counts as an improvement for "
            "early stopping."
        ),
    )
    gradient_clip_norm: float = Field(
        default=1.0,
        gt=0.0,
        description="Maximum gradient norm applied before each optimizer step.",
    )


def _validate_shape(name: str, value: tuple[int, int, int]) -> None:
    if len(value) != 3 or any(dimension <= 0 for dimension in value):
        raise ValueError(f"{name} must contain three positive dimensions")


def _validate_affine(name: str, value: FloatArray) -> None:
    if value.shape != (4, 4) or not np.isfinite(value).all():
        raise ValueError(f"{name} must be a finite 4 x 4 affine matrix")


@dataclass(frozen=True)
class ImageGeometry:
    shape: tuple[int, int, int]
    affine: FloatArray
    spacing: tuple[float, float, float]

    def __post_init__(self) -> None:
        _validate_shape("shape", self.shape)
        _validate_affine("affine", self.affine)
        if len(self.spacing) != 3 or any(
            not np.isfinite(value) or value <= 0 for value in self.spacing
        ):
            raise ValueError("spacing must contain three positive finite values")


@dataclass(frozen=True)
class CropTransform:
    source_shape: tuple[int, int, int]
    canonical_shape: tuple[int, int, int]
    crop_start: tuple[int, int, int]
    crop_stop: tuple[int, int, int]
    source_affine: FloatArray
    canonical_affine: FloatArray
    orientation_transform: IntArray

    def __post_init__(self) -> None:
        _validate_shape("source_shape", self.source_shape)
        _validate_shape("canonical_shape", self.canonical_shape)
        if len(self.crop_start) != 3 or len(self.crop_stop) != 3:
            raise ValueError("crop bounds must contain three axes")
        for axis, (start, stop, size) in enumerate(
            zip(self.crop_start, self.crop_stop, self.canonical_shape)
        ):
            if start < 0 or stop <= start or stop > size:
                raise ValueError(f"invalid crop bounds on axis {axis}")
        _validate_affine("source_affine", self.source_affine)
        _validate_affine("canonical_affine", self.canonical_affine)
        if self.orientation_transform.shape != (3, 2):
            raise ValueError("orientation_transform must be a 3 x 2 orientation array")


@dataclass(frozen=True)
class PreprocessResult:
    image: FloatArray
    mask: Optional[IntArray]
    geometry: ImageGeometry
    transform: CropTransform

    def __post_init__(self) -> None:
        if self.image.ndim != 3 or self.image.shape != self.geometry.shape:
            raise ValueError("image must be a 3D array matching geometry.shape")
        if not np.isfinite(self.image).all():
            raise ValueError("image must contain only finite values")
        if self.mask is not None and self.mask.shape != self.image.shape:
            raise ValueError("mask must match image shape")


@dataclass(frozen=True)
class ExtractedPatch:
    """A patch extracted from a subject volume, before it is written to disk.

    Produced by the patcher functions in ``core.dataloading.patchers`` and
    consumed by ``materialize_patches``, which stacks a subject's patches into
    per-subject ``.npy`` arrays and records each patch's location as a
    ``PatchRecord``.
    """

    volume: FloatArray
    mask: IntArray


@dataclass(frozen=True)
class LoadedPatch:
    """A patch reloaded from disk, ready for a dataset to tensorize.

    Produced by ``BasePatchDataset.load_patch``, which slices one patch out of
    the memory-mapped per-subject arrays and pairs it with the label from the
    patch's ``PatchRecord``. The read-side counterpart to ``ExtractedPatch``:
    same ``volume``/``mask`` arrays, plus the resolved ``has_microbleed`` label
    the classification datasets need.
    """

    volume: FloatArray
    mask: IntArray
    has_microbleed: bool


@dataclass(frozen=True)
class PatchRecord:
    """Location of one materialized patch within its subject's stacked arrays.

    Produced by ``core.dataloading.patchers.materialize_patches``, which stacks a
    subject's patches into ``volume_path``/``mask_path`` (each a ``(N, P, P, P)``
    ``.npy`` array) and emits one record per patch. The patch datasets slice
    ``patch_index`` out of the memory-mapped arrays; the balanced sampler reads
    ``has_microbleed``. Store-once inflation reuses a record's file+index, so
    several records may share the same ``patch_index``.
    """

    volume_path: str
    mask_path: str
    patch_index: int
    has_microbleed: bool
