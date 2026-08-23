import logging

import typer
from rich.logging import RichHandler
from typing_extensions import Annotated

from .commands import register_commands

app = typer.Typer(
    name="microbleednet",
    help=(
        "Research workflow for indexing, preprocessing, training, inference, "
        "and evaluation."
    ),
    no_args_is_help=True,
)

register_commands(app)


@app.callback()
def configure_logging(
    verbose: Annotated[bool, typer.Option(help="Emit debug-level logs.")] = False,
    quiet: Annotated[bool, typer.Option(help="Only emit warnings and errors.")] = False,
) -> None:
    """Configure Rich logging once for the whole CLI."""
    if verbose and quiet:
        raise typer.BadParameter("--verbose and --quiet are mutually exclusive")

    if verbose:
        level = logging.DEBUG
    elif quiet:
        level = logging.WARNING
    else:
        level = logging.INFO

    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
        force=True,
    )


def main() -> None:
    app()
