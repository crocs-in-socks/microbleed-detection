import typer

from . import index_data

# TODO: improve help message
app = typer.Typer(
    name="microbleednet",
    help="Allan please add more detail.",
    no_args_is_help=True,
)

app.add_typer(index_data.app)


def main() -> None:
    app()
