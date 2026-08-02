import logging

import typer
from rich.logging import RichHandler
from typing_extensions import Annotated

from . import evaluate, index_data, infer, preprocess, train

app = typer.Typer(
    name="microbleednet",
    help="Research workflow for indexing, preprocessing, training, inference, and evaluation.",
    no_args_is_help=True,
)

app.add_typer(index_data.app)
app.add_typer(preprocess.app)
app.add_typer(train.app)
app.add_typer(infer.app)
app.add_typer(evaluate.app)


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


def main() -> None:
    app()
