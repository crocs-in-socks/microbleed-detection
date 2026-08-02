from pathlib import Path

import typer
from typing_extensions import Annotated

from ..config import PreprocessCommandConfig
from ..pipelines import preprocess
from .errors import domain_errors
from .shared import config_epilog, finish, parse_config

app = typer.Typer()


@app.command(
    "preprocess",
    help="Preprocess the indexed dataset.",
    epilog=config_epilog(PreprocessCommandConfig),
)
def preprocess_command(
    config: Annotated[Path, typer.Option(..., exists=True, dir_okay=False)],
    dry_run: Annotated[bool, typer.Option(help="Validate configuration without writing outputs.")] = False,
) -> None:
    settings = parse_config(config, PreprocessCommandConfig)
    dataset_dir = settings.dataset_dir
    if not (dataset_dir / "manifests" / "raw.json").is_file():
        raise typer.BadParameter("dataset_dir has no raw manifest")
    if dry_run:
        finish(f"Preprocess configuration valid for {dataset_dir} (dry run)")
        return
    with domain_errors():
        preprocess.execute(dataset_dir, settings.preprocessing)
    finish(f"Preprocessed artifacts written under {dataset_dir}")
