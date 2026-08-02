from pathlib import Path

import typer
from typing_extensions import Annotated

from ..config import TrainCommandConfig
from ..pipelines import train
from .errors import domain_errors
from .shared import config_epilog, finish, parse_config

app = typer.Typer()


@app.command(
    "train",
    help="Train detector, teacher, and student models.",
    epilog=config_epilog(TrainCommandConfig),
)
def train_command(
    config: Annotated[Path, typer.Option(..., exists=True, dir_okay=False)],
    dry_run: Annotated[bool, typer.Option(help="Validate configuration without writing outputs.")] = False,
) -> None:
    settings = parse_config(config, TrainCommandConfig)
    if not (settings.dataset_dir / "manifests" / "preprocessed.json").is_file():
        raise typer.BadParameter("dataset_dir has no preprocessed manifest")
    if dry_run:
        finish(
            "Training configuration valid; outputs would use "
            f"{settings.experiment_dir} (dry run)"
        )
        return
    with domain_errors():
        train.execute(settings)
    finish(f"Training artifacts written under {settings.experiment_dir}")
