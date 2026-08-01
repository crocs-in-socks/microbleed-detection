import logging
from pathlib import Path
from typing import TypeVar

import typer
from pydantic import BaseModel, ValidationError
from rich.logging import RichHandler
from typing_extensions import Annotated

from ..config import (
    EvaluateCommandConfig,
    InferCommandConfig,
    PreprocessCommandConfig,
)
from ..pipelines import evaluate, infer, preprocess, train
from . import index_data
from .config import load_config, path_value, require_keys

_ConfigModel = TypeVar("_ConfigModel", bound=BaseModel)

app = typer.Typer(
    name="microbleednet",
    help="Research workflow for indexing, preprocessing, training, inference, and evaluation.",
    no_args_is_help=True,
)

app.add_typer(index_data.app)


@app.callback()
def _configure_logging(
    verbose: Annotated[bool, typer.Option(help="Emit debug-level logs.")] = False,
    quiet: Annotated[bool, typer.Option(help="Only emit warnings and errors.")] = False,
) -> None:
    """Configure Rich logging once for the whole CLI."""
    if verbose and quiet:
        raise typer.BadParameter("--verbose and --quiet are mutually exclusive")
    level = logging.DEBUG if verbose else logging.WARNING if quiet else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
        force=True,
    )


def _config(path: Path, required: tuple[str, ...]) -> dict:
    config = load_config(path)
    require_keys(config, required)
    return config


def _parse_config(path: Path, model: type[_ConfigModel]) -> _ConfigModel:
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


def _config_epilog(model: type[BaseModel]) -> str:
    return "Configuration keys (JSON/TOML):\n" + "\n".join(_config_help(model))


def _finish(message: str) -> None:
    typer.echo(message)


@app.command("validate-data", help="Validate an indexed dataset manifest.")
def validate_data(
    dataset_dir: Annotated[Path, typer.Option(..., exists=True, file_okay=False)],
    require_masks: Annotated[bool, typer.Option(help="Require a mask for every subject.")] = True,
    dry_run: Annotated[bool, typer.Option(help="Validate without writing artifacts.")] = False,
) -> None:
    manifest_path = dataset_dir / "manifests" / "raw.json"
    if not manifest_path.is_file():
        raise typer.BadParameter(f"raw manifest does not exist: {manifest_path}")
    import json
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    subjects = manifest.get("subjects", [])
    missing = [subject.get("subject_id", "unknown") for subject in subjects if not Path(subject["volume_path"]).is_file()]
    if require_masks:
        missing.extend(subject.get("subject_id", "unknown") for subject in subjects if not subject.get("mask_path") or not Path(subject["mask_path"]).is_file())
    if missing:
        raise typer.BadParameter(f"invalid dataset subjects: {sorted(set(missing))}")
    _finish(f"Validated {len(subjects)} subjects" + (" (dry run)" if dry_run else ""))


@app.command(
    "preprocess",
    help="Preprocess the indexed dataset.",
    epilog=_config_epilog(PreprocessCommandConfig),
)
def preprocess_command(
    config: Annotated[Path, typer.Option(..., exists=True, dir_okay=False)],
    dry_run: Annotated[bool, typer.Option(help="Validate configuration without writing outputs.")] = False,
) -> None:
    settings = _parse_config(config, PreprocessCommandConfig)
    dataset_dir = settings.dataset_dir
    if not (dataset_dir / "manifests" / "raw.json").is_file():
        raise typer.BadParameter("dataset_dir has no raw manifest")
    if dry_run:
        _finish(f"Preprocess configuration valid for {dataset_dir} (dry run)")
        return
    preprocess.execute(dataset_dir, settings.preprocessing)
    _finish(f"Preprocessed artifacts written under {dataset_dir}")


@app.command("train", help="Train detector, teacher, and student models.")
def train_command(
    config: Annotated[Path, typer.Option(..., exists=True, dir_okay=False)],
    dry_run: Annotated[bool, typer.Option(help="Validate configuration without writing outputs.")] = False,
) -> None:
    values = _config(config, ("dataset_dir", "experiment_dir", "datasplit_parameters", "detector_parameters", "discriminator_teacher_parameters", "discriminator_student_parameters"))
    dataset_dir = path_value(values, "dataset_dir")
    experiment_dir = Path(values["experiment_dir"])
    if not (dataset_dir / "manifests" / "preprocessed.json").is_file():
        raise typer.BadParameter("dataset_dir has no preprocessed manifest")
    if dry_run:
        _finish(f"Training configuration valid; outputs would use {experiment_dir} (dry run)")
        return
    train.execute(dataset_dir, experiment_dir, values["datasplit_parameters"], values["detector_parameters"], values["discriminator_teacher_parameters"], values["discriminator_student_parameters"])
    _finish(f"Training artifacts written under {experiment_dir}")


@app.command(
    "infer",
    help="Run detector and student inference for a volume.",
    epilog=_config_epilog(InferCommandConfig),
)
def infer_command(
    config: Annotated[Path, typer.Option(..., exists=True, dir_okay=False)],
    dry_run: Annotated[bool, typer.Option(help="Validate configuration without writing outputs.")] = False,
) -> None:
    settings = _parse_config(config, InferCommandConfig)
    for label, path in (
        ("volume_path", settings.volume_path),
        ("detector_checkpoint", settings.detector_checkpoint),
        ("student_checkpoint", settings.student_checkpoint),
    ):
        if not path.exists():
            raise typer.BadParameter(f"configured path does not exist ({label}): {path}")
    if dry_run:
        _finish(f"Inference configuration valid for {settings.volume_path} (dry run)")
        return
    result = infer.execute(settings)
    _finish(f"Wrote prediction artifacts under {result['mask_path']}")


@app.command(
    "evaluate",
    help="Evaluate source-space predictions against references.",
    epilog=_config_epilog(EvaluateCommandConfig),
)
def evaluate_command(
    config: Annotated[Path, typer.Option(..., exists=True, dir_okay=False)],
    dry_run: Annotated[bool, typer.Option(help="Validate configuration without writing outputs.")] = False,
) -> None:
    settings = _parse_config(config, EvaluateCommandConfig)
    for subject in settings.subjects:
        for path in (subject.prediction_path, subject.reference_path):
            if not path.is_file():
                raise typer.BadParameter(f"configured path does not exist: {path}")
    if dry_run:
        _finish(f"Evaluation configuration valid for {len(settings.subjects)} subjects (dry run)")
        return
    result = evaluate.execute(settings.subjects, settings.output_dir, settings.metadata)
    _finish(f"Evaluated {len(result['subjects'])} subjects; report written under {settings.output_dir}")


def main() -> None:
    app()
