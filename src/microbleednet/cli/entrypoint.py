from pathlib import Path
from typing_extensions import Annotated

import torch
import typer

from . import index_data
from .config import load_config, path_value, require_keys
from ..pipelines import evaluate, predict, preprocess, train

app = typer.Typer(
    name="microbleednet",
    help="Research workflow for indexing, preprocessing, training, prediction, and evaluation.",
    no_args_is_help=True,
)

app.add_typer(index_data.app)


def _config(path: Path, required: tuple[str, ...]) -> dict:
    config = load_config(path)
    require_keys(config, required)
    return config


def _finish(message: str) -> None:
    typer.echo(message)


@app.command("validate-data", help="Validate an indexed dataset manifest.")
def validate_data(
    dataset_dir: Annotated[Path, typer.Option(..., exists=True, file_okay=False)],
    require_masks: Annotated[bool, typer.Option(help="Require a mask for every subject.")] = True,
    dry_run: Annotated[bool, typer.Option(help="Validate without writing artifacts.")] = False,
) -> None:
    manifest_path = dataset_dir / "manifests" / "raw.json"
    if not manifest_path.is_file():
        raise typer.BadParameter(f"raw manifest does not exist: {manifest_path}")
    import json
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    subjects = manifest.get("subjects", [])
    missing = [subject.get("subject_id", "unknown") for subject in subjects if not Path(subject["volume_path"]).is_file()]
    if require_masks:
        missing.extend(subject.get("subject_id", "unknown") for subject in subjects if not subject.get("mask_path") or not Path(subject["mask_path"]).is_file())
    if missing:
        raise typer.BadParameter(f"invalid dataset subjects: {sorted(set(missing))}")
    _finish(f"Validated {len(subjects)} subjects" + (" (dry run)" if dry_run else ""))


@app.command("preprocess", help="Preprocess the indexed dataset.")
def preprocess_command(
    config: Annotated[Path, typer.Option(..., exists=True, dir_okay=False)],
    dry_run: Annotated[bool, typer.Option(help="Validate configuration without writing outputs.")] = False,
) -> None:
    values = _config(config, ("dataset_dir", "preprocessor_parameters"))
    dataset_dir = path_value(values, "dataset_dir")
    if not (dataset_dir / "manifests" / "raw.json").is_file():
        raise typer.BadParameter("dataset_dir has no raw manifest")
    if dry_run:
        _finish(f"Preprocess configuration valid for {dataset_dir} (dry run)")
        return
    preprocess.execute(dataset_dir, values["preprocessor_parameters"])
    _finish(f"Preprocessed artifacts written under {dataset_dir}")


@app.command("train", help="Train detector, teacher, and student models.")
def train_command(
    config: Annotated[Path, typer.Option(..., exists=True, dir_okay=False)],
    dry_run: Annotated[bool, typer.Option(help="Validate configuration without writing outputs.")] = False,
) -> None:
    values = _config(config, ("dataset_dir", "experiment_dir", "datasplit_parameters", "detector_parameters", "discriminator_teacher_parameters", "discriminator_student_parameters"))
    dataset_dir = path_value(values, "dataset_dir")
    experiment_dir = Path(values["experiment_dir"])
    if not (dataset_dir / "manifests" / "preprocessed.json").is_file():
        raise typer.BadParameter("dataset_dir has no preprocessed manifest")
    if dry_run:
        _finish(f"Training configuration valid; outputs would use {experiment_dir} (dry run)")
        return
    train.execute(dataset_dir, experiment_dir, values["datasplit_parameters"], values["detector_parameters"], values["discriminator_teacher_parameters"], values["discriminator_student_parameters"])
    _finish(f"Training artifacts written under {experiment_dir}")


@app.command("predict", help="Run detector and student prediction for a volume.")
def predict_command(
    config: Annotated[Path, typer.Option(..., exists=True, dir_okay=False)],
    dry_run: Annotated[bool, typer.Option(help="Validate configuration without writing outputs.")] = False,
) -> None:
    values = _config(config, ("volume_path", "output_dir", "model_parameters", "detector_checkpoint", "student_checkpoint", "preprocess_parameters"))
    volume_path = path_value(values, "volume_path")
    detector_checkpoint = path_value(values, "detector_checkpoint")
    student_checkpoint = path_value(values, "student_checkpoint")
    if dry_run:
        _finish(f"Prediction configuration valid for {volume_path} (dry run)")
        return
    result = predict.execute(
        volume_path=volume_path,
        output_dir=Path(values["output_dir"]),
        model_parameters=values["model_parameters"],
        detector_checkpoint=detector_checkpoint,
        student_checkpoint=student_checkpoint,
        preprocess_parameters=values["preprocess_parameters"],
        device=torch.device(values.get("device", "cpu")),
        detector_threshold=values.get("detector_threshold", 0.5),
        student_threshold=values.get("student_threshold", 0.5),
        patch_batch_size=values.get("patch_batch_size", 8),
    )
    _finish(f"Wrote prediction artifacts under {result['mask_path']}")


@app.command("evaluate", help="Evaluate source-space predictions against references.")
def evaluate_command(
    config: Annotated[Path, typer.Option(..., exists=True, dir_okay=False)],
    dry_run: Annotated[bool, typer.Option(help="Validate configuration without writing outputs.")] = False,
) -> None:
    values = _config(config, ("subjects", "output_dir"))
    if not isinstance(values["subjects"], list) or not values["subjects"]:
        raise typer.BadParameter("configuration key 'subjects' must be a non-empty list")
    for subject in values["subjects"]:
        for key in ("prediction_path", "reference_path"):
            path = Path(subject.get(key, ""))
            if not path.is_file():
                raise typer.BadParameter(f"configured path does not exist: {path}")
    if dry_run:
        _finish(f"Evaluation configuration valid for {len(values['subjects'])} subjects (dry run)")
        return
    result = evaluate.execute(values["subjects"], Path(values["output_dir"]), values.get("metadata"))
    _finish(f"Evaluated {len(result['subjects'])} subjects; report written under {values['output_dir']}")


def main() -> None:
    app()
