"""Table-driven construction of the CLI commands.

Every command is the same skeleton — parse a config, check preconditions, honor
``--dry-run``, defer-import one pipe, run it, report — so each is described
declaratively by a :class:`CommandSpec` and built by :func:`build_command`
rather than hand-written in its own module. Only the per-command specifics
(config model, help text, preconditions, messages, and which pipe to run)
vary, and those are exactly the spec's fields.

The pipe is imported lazily inside the command body, by module name, so
``--help`` and ``describe`` never pay to import torch.
"""

from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any, Callable

import typer
from pydantic import BaseModel
from rich.console import Console
from rich.table import Table
from typing_extensions import Annotated

from ..orchestration.configs import (
    EvaluateConfig,
    IndexDataConfig,
    InferConfig,
    PreprocessConfig,
    TrainConfig,
)
from ..orchestration.layouts import DatasetLayout
from ..orchestration.manifests import PreprocessedDatasetManifest, read_manifest
from .utils import (
    config_fields,
    describe_hint,
    finish,
    parse_config,
    require_dir,
    require_file,
)


@dataclass(frozen=True)
class CommandSpec:
    """Everything that distinguishes one CLI command from the shared skeleton."""

    name: str
    help: str
    config: type[BaseModel]
    pipe: str  # module under orchestration.pipes; imported lazily to defer torch
    dry_run_message: Callable[[Any], str]
    success_message: Callable[[Any, Any], str]
    preconditions: Callable[[Any], None] = field(default=lambda settings: None)


def validate_index_data(settings: IndexDataConfig) -> None:
    require_dir(settings.input_dir, "input_dir")
    if settings.label_dir is not None:
        require_dir(settings.label_dir, "label_dir")


def validate_preprocess(settings: PreprocessConfig) -> None:
    require_file(settings.dataset_dir / "manifests" / "raw.json", "raw manifest")


def validate_train(settings: TrainConfig) -> None:
    manifest_path = settings.dataset_dir / DatasetLayout().preprocessed_manifest
    require_file(manifest_path, "preprocessed manifest")
    # Training is fully supervised, so every preprocessed subject must carry a
    # mask. Validate the dataset's contents here in the CLI layer, before the
    # orchestration pipe runs, so a dataset that cannot train fails fast.
    try:
        manifest = read_manifest(manifest_path, PreprocessedDatasetManifest)
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    missing = [
        subject.subject_id for subject in manifest.subjects if not subject.mask_path
    ]
    if missing:
        raise typer.BadParameter(f"training requires masks for subjects: {missing}")


def validate_infer(settings: InferConfig) -> None:
    require_file(settings.volume_path, "volume_path")
    require_file(settings.detector_checkpoint, "detector_checkpoint")
    require_file(settings.student_checkpoint, "student_checkpoint")


def validate_evaluate(settings: EvaluateConfig) -> None:
    require_file(settings.dataset_dir / "manifests" / "raw.json", "raw manifest")
    if settings.detector_checkpoint is not None:
        require_file(settings.detector_checkpoint, "detector_checkpoint")
    if settings.student_checkpoint is not None:
        require_file(settings.student_checkpoint, "student_checkpoint")
    if settings.experiment_dir is not None:
        require_dir(settings.experiment_dir, "experiment_dir")


SPECS: list[CommandSpec] = [
    CommandSpec(
        name="index-data",
        help=(
            "Generate a dataset directory containing manifests that list the "
            "subjects included in the dataset. If masks are provided, they are "
            "matched to the volumes using subject ids extracted from the file "
            "names based on the patterns provided."
        ),
        config=IndexDataConfig,
        pipe="index_data",
        preconditions=validate_index_data,
        dry_run_message=lambda s: (
            f"Index configuration valid; manifests would be written under "
            f"{s.dataset_dir} (dry run)"
        ),
        success_message=lambda s, _: (
            f"Indexed dataset manifests written under {s.dataset_dir}"
        ),
    ),
    CommandSpec(
        name="preprocess",
        help="Preprocess the indexed dataset.",
        config=PreprocessConfig,
        pipe="preprocess",
        preconditions=validate_preprocess,
        dry_run_message=lambda s: (
            f"Preprocess configuration valid for {s.dataset_dir} (dry run)"
        ),
        success_message=lambda s, _: (
            f"Preprocessed artifacts written under {s.dataset_dir}"
        ),
    ),
    CommandSpec(
        name="train",
        help="Train detector, teacher, and student models.",
        config=TrainConfig,
        pipe="train",
        preconditions=validate_train,
        dry_run_message=lambda s: (
            "Training configuration valid; outputs would use "
            f"{s.experiment_dir} (dry run)"
        ),
        success_message=lambda s, _: (
            f"Training artifacts written under {s.experiment_dir}"
        ),
    ),
    CommandSpec(
        name="infer",
        help="Run detector and student inference for a volume.",
        config=InferConfig,
        pipe="infer",
        preconditions=validate_infer,
        dry_run_message=lambda s: (
            f"Inference configuration valid for {s.volume_path} (dry run)"
        ),
        success_message=lambda s, result: (
            f"Prediction artifacts written under {result.mask_path}"
        ),
    ),
    CommandSpec(
        name="evaluate",
        help="Infer over an indexed dataset and score predictions against references.",
        config=EvaluateConfig,
        pipe="evaluate",
        preconditions=validate_evaluate,
        dry_run_message=lambda s: (
            f"Evaluation configuration valid for {s.dataset_dir} (dry run)"
        ),
        success_message=lambda s, result: (
            f"Evaluated {len(result['subjects'])} subjects; "
            f"report written under {s.output_dir}"
        ),
    ),
]


def build_command(app: typer.Typer, spec: CommandSpec) -> None:
    """Register ``spec`` as a command on ``app``.

    The generated command runs the shared skeleton; the pipe is imported by
    module name inside the body so torch stays out of ``--help``/``describe``.
    """

    @app.command(spec.name, help=spec.help, epilog=describe_hint(spec.name))
    def command(
        config: Annotated[Path, typer.Option(..., exists=True, dir_okay=False)],
        dry_run: Annotated[
            bool, typer.Option(help="Validate configuration without writing outputs.")
        ] = False,
    ) -> None:
        settings = parse_config(config, spec.config)
        spec.preconditions(settings)
        if dry_run:
            finish(spec.dry_run_message(settings))
            return
        # Imported lazily, by module name, so torch stays out of --help/describe.
        pipe = import_module(
            f"..orchestration.pipes.{spec.pipe}", package=__package__
        )
        result = pipe.execute(settings)
        finish(spec.success_message(settings, result))


def build_describe_command(app: typer.Typer, specs: list[CommandSpec]) -> None:
    """Register the meta-command that documents each command's config keys.

    ``describe`` reads only the config models (via ``config_fields``), so it
    never imports a pipe and stays torch-free.
    """

    configs = {spec.name: spec.config for spec in specs}

    @app.command(
        "describe",
        help="Print the configuration keys, descriptions, and defaults for a command.",
        no_args_is_help=True,
    )
    def describe_command(
        command: Annotated[
            str,
            typer.Argument(help=f"Command to describe: {', '.join(configs)}."),
        ],
    ) -> None:
        model = configs.get(command)
        if model is None:
            raise typer.BadParameter(
                f"unknown command '{command}'; choose one of: {', '.join(configs)}"
            )

        table = Table(
            title=f"microbleednet {command} - configuration keys (TOML)",
            title_style="bold",
            header_style="bold",
            expand=True,
            row_styles=["", "dim"],
        )
        table.add_column("Key", style="cyan", no_wrap=True)
        table.add_column("Default", style="magenta", no_wrap=True)
        table.add_column("Description", ratio=1)

        current_section = ""
        for config_field in config_fields(model):
            if config_field.section != current_section:
                table.add_section()
                if config_field.section:
                    table.add_row(f"[bold]\\[{config_field.section}][/bold]", "", "")
                current_section = config_field.section
            required = config_field.default == "required"
            table.add_row(
                config_field.key,
                "[yellow]required[/yellow]" if required else config_field.default,
                config_field.description,
            )

        Console().print(table)


def register_commands(app: typer.Typer) -> None:
    """Register every pipe command and the ``describe`` meta-command."""
    for spec in SPECS:
        build_command(app, spec)
    build_describe_command(app, SPECS)
