import json
from pathlib import Path

import nibabel as nib
import numpy as np
from typer.testing import CliRunner

from microbleednet.cli.entrypoint import app


runner = CliRunner()


def test_root_and_command_help() -> None:
    root = runner.invoke(app, ["--help"])
    assert root.exit_code == 0
    for command in ("validate-data", "preprocess", "train", "infer", "evaluate", "index-data"):
        result = runner.invoke(app, [command, "--help"])
        assert result.exit_code == 0, result.output


def test_preprocess_dry_run_writes_no_outputs(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    manifest_dir = dataset_dir / "manifests"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "raw.json").write_text(json.dumps({"subjects": []}), encoding="utf-8")
    config_path = tmp_path / "preprocess.json"
    config_path.write_text(json.dumps({"dataset_dir": str(dataset_dir)}), encoding="utf-8")

    result = runner.invoke(app, ["preprocess", "--config", str(config_path), "--dry-run"])
    assert result.exit_code == 0, result.output
    assert not (dataset_dir / "preprocessed").exists()


def test_synthetic_evaluate_workflow_completes(tmp_path: Path) -> None:
    affine = np.eye(4)
    prediction = np.zeros((8, 8, 8), dtype=np.uint8)
    reference = np.zeros_like(prediction)
    prediction[2:4, 2:4, 2:4] = 1
    reference[2:4, 2:4, 2:4] = 1
    prediction_path = tmp_path / "prediction.nii.gz"
    reference_path = tmp_path / "reference.nii.gz"
    nib.save(nib.Nifti1Image(prediction, affine), prediction_path)
    nib.save(nib.Nifti1Image(reference, affine), reference_path)
    config_path = tmp_path / "evaluate.json"
    output_dir = tmp_path / "report"
    config_path.write_text(json.dumps({
        "subjects": [{
            "subject_id": "synthetic",
            "prediction_path": str(prediction_path),
            "reference_path": str(reference_path),
        }],
        "output_dir": str(output_dir),
    }), encoding="utf-8")

    result = runner.invoke(app, ["evaluate", "--config", str(config_path)])
    assert result.exit_code == 0, result.output
    report = json.loads((output_dir / "evaluation.json").read_text(encoding="utf-8"))
    assert report["aggregate"]["true_positives"] == 1
