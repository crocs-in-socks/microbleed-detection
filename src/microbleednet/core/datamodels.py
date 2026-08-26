"""Datamodels owned by the ``core`` layer.

``FrozenModel`` is the app-wide Pydantic base used by core and orchestration
models. It is colocated with ``core`` so ``core`` is self-sufficient: nothing
here reaches up into ``orchestration`` or ``cli``.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict

Modality = Literal["T2*-GRE", "SWI", "QSM"]


class FrozenModel(BaseModel):
    """Immutable, unknown-field-rejecting base for application data models."""

    model_config = ConfigDict(extra="forbid", frozen=True)
