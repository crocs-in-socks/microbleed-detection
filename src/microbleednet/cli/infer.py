from pathlib import Path

import typer
from typing_extensions import Annotated

from ..config import InferCommandConfig
from ..pipelines import infer
from .errors import domain_errors
from .shared import config_epilog, finish, parse_config

app = typer.Typer()


@app.command(
    "infer",
    help="Run detector and student inference for a volume.",
    epilog=config_epilog(InferCommandConfig),
)
def infer_command(
    config: Annotated[Path, typer.Option(..., exists=True, dir_okay=False)],
    dry_run: Annotated[bool, typer.Option(help="Validate configuration without writing outputs.")] = False,
) -> None:
    settings = parse_config(config, InferCommandConfig)
    for label, path in (
        ("volume_path", settings.volume_path),
        ("detector_checkpoint", settings.detector_checkpoint),
        ("student_checkpoint", settings.student_checkpoint),
    ):
        if not path.exists():
            raise typer.BadParameter(f"configured path does not exist ({label}): {path}")
    if dry_run:
        finish(f"Inference configuration valid for {settings.volume_path} (dry run)")
        return
    with domain_errors():
        result = infer.execute(settings)
    finish(f"Wrote prediction artifacts under {result.mask_path}")
