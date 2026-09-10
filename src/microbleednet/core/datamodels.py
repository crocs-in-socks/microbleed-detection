from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

FloatArray = NDArray[np.floating]
IntArray = NDArray[np.integer]
Shape3D = tuple[int, int, int]
BoundingBox = tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
Modality = Literal["T2*-GRE", "SWI", "QSM"]


class FrozenModel(BaseModel):
    """Immutable, unknown-field-rejecting base for application data models."""

    model_config = ConfigDict(extra="forbid", frozen=True)


@dataclass(frozen=True)
class PreprocessResult:
    image: FloatArray
    mask: IntArray
    affine: FloatArray

    def __post_init__(self) -> None:
        if self.image.ndim != 3:
            raise ValueError("image must be a 3D array")
        if not np.isfinite(self.image).all():
            raise ValueError("image must contain only finite values")
        if self.mask.shape != self.image.shape:
            raise ValueError("mask must match image shape")
