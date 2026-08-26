"""Helpers shared by the CLI commands.

The commands are generated from a declarative table in ``cli/commands.py`` and
wired up by ``entrypoint.py``. Everything they need but do not own — config
loading and parsing, precondition checks, config-key introspection, and output —
lives here.
"""

import tomllib
from pathlib import Path
from typing import Any, NamedTuple

import typer
from pydantic import BaseModel, ValidationError


def load_config(path: Path) -> dict[str, Any]:
    """Read a TOML config file into a plain dict."""
    if not path.is_file():
        raise ValueError(f"configuration file does not exist: {path}")
    if path.suffix.lower() != ".toml":
        raise ValueError(f"configuration file must be a .toml file: {path}")
    try:
        with path.open("rb") as config_file:
            config = tomllib.load(config_file)
    except (OSError, ValueError) as error:
        raise ValueError(f"could not read configuration {path}: {error}") from error
    if not isinstance(config, dict):
        raise ValueError("configuration root must be an object")
    return config


def parse_config[ConfigModel: BaseModel](
    path: Path, model: type[ConfigModel]
) -> ConfigModel:
    """Load a config file and validate it into a typed model.

    Pydantic validation errors are surfaced as clean CLI errors rather than
    tracebacks.
    """
    try:
        return model.model_validate(load_config(path))
    except ValidationError as error:
        raise typer.BadParameter(f"invalid configuration {path}:\n{error}") from error


class ConfigField(NamedTuple):
    """A single leaf configuration key derived from a config model."""

    section: str  # nested-model prefix, e.g. "detector"; "" for top-level keys
    key: str  # dotted key relative to its section, e.g. "patch_size"
    default: str  # "required" or a repr of the default value
    description: str


def config_fields(model: type[BaseModel], prefix: str = "") -> list[ConfigField]:
    """Flatten a config model (recursing into nested models) into leaf fields.

    Derived from the model so descriptions, defaults, and required-ness never
    drift from the Pydantic configuration models.
    """
    fields: list[ConfigField] = []
    for name, field in model.model_fields.items():
        key = f"{prefix}{name}"
        annotation = field.annotation
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            fields.extend(config_fields(annotation, prefix=f"{key}."))
            continue
        section, _, leaf = key.rpartition(".")
        default = "required" if field.is_required() else repr(field.default)
        fields.append(
            ConfigField(
                section=section,
                key=leaf,
                default=default,
                description=field.description or "",
            )
        )
    return fields


def require_dir(path: Path, label: str) -> None:
    """Raise a clean CLI error unless ``path`` is an existing directory."""
    if not path.is_dir():
        raise typer.BadParameter(f"{label} does not exist: {path}")


def describe_hint(command: str) -> str:
    """One-line epilog pointing users at the full config reference."""
    return f"Run 'microbleednet describe {command}' for the configuration keys."


def report(message: str) -> None:
    typer.echo(message)
