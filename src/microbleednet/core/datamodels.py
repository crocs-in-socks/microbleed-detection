"""Datamodels owned by the ``core`` layer.

``FrozenModel`` is the app-wide Pydantic base used by core and orchestration
models. It is colocated with ``core`` so ``core`` is self-sufficient: nothing
here reaches up into ``orchestration`` or ``cli``.
"""

from pydantic import BaseModel, ConfigDict


class FrozenModel(BaseModel):
    """Immutable, unknown-field-rejecting base for application data models."""

    model_config = ConfigDict(extra="forbid", frozen=True)
