from pathlib import Path

import typer
from typing_extensions import Annotated

from ..config import EvaluateCommandConfig
from ..pipelines import evaluate
from .errors import domain_errors
from .shared import config_epilog, finish, parse_config

app = typer.Typer()


@app.command(
    "evaluate",
    help="Evaluate source-space predictions against references.",
    epilog=config_epilog(EvaluateCommandConfig),
)
def evaluate_command(
    config: Annotated[Path, typer.Option(..., exists=True, dir_okay=False)],
    dry_run: Annotated[bool, typer.Option(help="Validate configuration without writing outputs.")] = False,
) -> None:
    settings = parse_config(config, EvaluateCommandConfig)
    for subject in settings.subjects:
        paths = [subject.prediction_path, subject.reference_path]
        if subject.probability_path is not None:
            paths.append(subject.probability_path)
        for path in paths:
            if not path.is_file():
                raise typer.BadParameter(f"configured path does not exist: {path}")
    if dry_run:
        finish(f"Evaluation configuration valid for {len(settings.subjects)} subjects (dry run)")
        return
    with domain_errors():
        result = evaluate.execute(
            settings.subjects,
            settings.output_dir,
            settings.metadata,
            settings.froc_thresholds,
        )
    finish(f"Evaluated {len(result['subjects'])} subjects; report written under {settings.output_dir}")
