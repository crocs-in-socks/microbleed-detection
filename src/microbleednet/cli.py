import typer
import importlib.metadata
from pathlib import Path
from typing import Optional
from rich.console import Console

from microbleednet.commands.preprocess import preprocess_pipeline

app = typer.Typer(
    name="microbleednet",
    help="""
        microbleednet: Triplanar ensemble U-Net model
        
        For detailed help regarding the options for each command type:
        microbleednet <command> --help (e.g. microbleednet train --help)
    """,
    no_args_is_help=True,
)

console = Console()

@app.callback(invoke_without_command=True)
def callback(
        version: Optional[bool] = typer.Option(None, "--version", is_eager=True, help="Show the version of microbleednet and exit."),
    ) -> None:
    """
    A callback function that handles the --version option
    """
    if version:
        try:
            __version__ = importlib.metadata.version("microbleednet")
        except importlib.metadata.PackageNotFoundError:
            __version__ = "unknown"
        
        console.print(f"microbleednet version: [bold green]{__version__}[/bold green]")
        raise typer.Exit()

def main() -> None:
    app()