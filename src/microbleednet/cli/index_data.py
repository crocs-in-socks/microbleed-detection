from pathlib import Path
from typing import Optional
from typing_extensions import Annotated

from ..pipelines import index_data

import typer

app = typer.Typer()

def validate_pattern(value: str) -> str:
    if value is None:
        return value
    
    if "{subject_id}" not in value:
        raise typer.BadParameter("The input must contain the '{subject_id}' placeholder.")
    return value

# TODO: improve the help message
@app.command(
    name="index-data",
    help="""
        Generates a dataset directory containing manifests that list the subjects included in the dataset. If masks are provided, they are matched to the volumes using subject ids extracted from the file names based on the patterns provided.
    """,
    no_args_is_help=True
)
def validate_parameters(
    dataset_dir: Annotated[Path, typer.Option(
        help="Directory to which the indexed dataset will be saved."
    )],
    input_dir: Annotated[Path, typer.Option(
        help="Input directory containing volumes to be indexed.",
        exists=True,
    )],
    volume_pattern: Annotated[str, typer.Option(
        help="Naming pattern of volumes in the input directory. Must contain '{subject_id}'",
        callback=validate_pattern
    )],
    label_dir: Annotated[Optional[Path], typer.Option(
        help="Label directory containing masks to be indexed.",
        exists=True
    )] = None,
    mask_pattern: Annotated[str, typer.Option(
        help="Naming pattern of masks in the label directory. Must contain '{subject_id}'",
        callback=validate_pattern
    )] = None,
) -> None:
    if label_dir is not None and mask_pattern is None:
        raise typer.BadParameter("If you provider --label-dir, you MUST also provide --mask-pattern")

    # pass arguments to pipeline

    index_data.execute(
        dataset_dir=dataset_dir,
        input_dir=input_dir,
        volume_pattern=volume_pattern,
        label_dir=label_dir,
        mask_pattern=mask_pattern
    )