from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.floating]
IntArray = NDArray[np.integer]


def _validate_shape(name: str, value: tuple[int, int, int]) -> None:
    if len(value) != 3 or any(dimension <= 0 for dimension in value):
        raise ValueError(f"{name} must contain three positive dimensions")


def _validate_affine(name: str, value: FloatArray) -> None:
    if value.shape != (4, 4) or not np.isfinite(value).all():
        raise ValueError(f"{name} must be a finite 4 x 4 affine matrix")


@dataclass(frozen=True)
class SubjectRecord:
    subject_id: str
    volume_path: Path
    mask_path: Optional[Path] = None

    def __post_init__(self) -> None:
        if not self.subject_id.strip():
            raise ValueError("subject_id must be nonempty")
        if not self.volume_path:
            raise ValueError("volume_path must be provided")


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
class PatchRecord:
    source_subject: str
    bounds: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
    label: int
    image: Optional[FloatArray] = None
    mask: Optional[IntArray] = None

    def __post_init__(self) -> None:
        if not self.source_subject.strip():
            raise ValueError("source_subject must be nonempty")
        if len(self.bounds) != 3 or any(start >= stop for start, stop in self.bounds):
            raise ValueError("bounds must contain three increasing intervals")
        if self.label not in (0, 1):
            raise ValueError("label must be 0 or 1")
        if self.image is not None and self.image.ndim != 3:
            raise ValueError("image patch must be 3D")
        if self.mask is not None and self.mask.ndim != 3:
            raise ValueError("mask patch must be 3D")
        if self.image is not None and self.mask is not None:
            if self.image.shape != self.mask.shape:
                raise ValueError("image and mask patches must have matching shapes")


@dataclass(frozen=True)
class PredictionSummary:
    """Paths and counts an inference run returns to its caller.

    Carries no arrays — just the locations of the artifacts written to disk and
    how many candidate components were found — so a frozen dataclass rather than
    a Pydantic model is the right fit for this in-memory return value.
    """

    subject_id: str
    mask_path: Path
    probability_path: Path
    components_path: Path
    component_count: int

    def __post_init__(self) -> None:
        if not self.subject_id.strip():
            raise ValueError("subject_id must be nonempty")
        if self.component_count < 0:
            raise ValueError("component_count must be non-negative")


@dataclass(frozen=True)
class CandidateComponent:
    label: int
    voxel_count: int
    centroid: tuple[float, float, float]
    bounding_box: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
    source_subject: str

    def __post_init__(self) -> None:
        if self.label <= 0 or self.voxel_count <= 0:
            raise ValueError("candidate label and voxel_count must be positive")
        if len(self.centroid) != 3 or not all(np.isfinite(self.centroid)):
            raise ValueError("centroid must contain three finite values")
        if len(self.bounding_box) != 3 or any(
            start >= stop for start, stop in self.bounding_box
        ):
            raise ValueError("bounding_box must contain three increasing intervals")
        if not self.source_subject.strip():
            raise ValueError("source_subject must be nonempty")