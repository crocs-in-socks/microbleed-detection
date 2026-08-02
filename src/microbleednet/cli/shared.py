"""Helpers shared across the per-command CLI modules.

Each command lives in its own ``cli/<command>.py`` module (see ARCHITECTURE.md's
command table). The parsing, help-rendering, and output helpers they all use live
here so no command module has to import another command module's internals.
"""

from pathlib import Path
from typing import TypeVar

import typer
from pydantic import BaseModel, ValidationError

from .config import load_config

_ConfigModel = TypeVar("_ConfigModel", bound=BaseModel)


def parse_config(path: Path, model: type[_ConfigModel]) -> _ConfigModel:
    """Load a config file and validate it into a typed model.

    Pydantic validation errors are surfaced as clean CLI errors rather than
    tracebacks.
    """
    try:
        return model.model_validate(load_config(path))
    except ValidationError as error:
        raise typer.BadParameter(f"invalid configuration {path}:\n{error}") from error


def _config_help(model: type[BaseModel], prefix: str = "") -> list[str]:
    """Render a model's fields (recursing into nested models) as help lines.

    The command help is derived from the model so descriptions, defaults, and
    required-ness never drift from the single source of truth in config.py.
    """
    lines: list[str] = []
    for name, field in model.model_fields.items():
        key = f"{prefix}{name}"
        annotation = field.annotation
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            lines.extend(_config_help(annotation, prefix=f"{key}."))
            continue
        if field.is_required():
            requirement = "required"
        else:
            requirement = f"default {field.default!r}"
        description = field.description or ""
        lines.append(f"  {key} ({requirement}): {description}".rstrip())
    return lines


def config_epilog(model: type[BaseModel]) -> str:
    return "Configuration keys (JSON/TOML):\n" + "\n".join(_config_help(model))


def finish(message: str) -> None:
    typer.echo(message)
