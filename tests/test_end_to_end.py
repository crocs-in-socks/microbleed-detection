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


def test_typed_command_help_documents_config_keys() -> None:
    result = runner.invoke(app, ["infer", "--help"])
    assert result.exit_code == 0, result.output
    output = " ".join(result.output.split())
    assert "Configuration keys" in output
    assert "detector.initial_channels" in output
    assert "patch_batch_size" in output


def test_train_help_documents_config_keys() -> None:
    result = runner.invoke(app, ["train", "--help"])
    assert result.exit_code == 0, result.output
    output = " ".join(result.output.split())
    assert "Configuration keys" in output
    assert "detector.initial_channels" in output
    assert "datasplit.test_size" in output


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


def _train_config_dict(dataset_dir: Path, experiment_dir: Path) -> dict:
    model_block = {
        "initial_channels": 64,
        "input_channels": 2,
        "output_classes": 2,
        "dropout_rate": 0.5,
    }
    return {
        "dataset_dir": str(dataset_dir),
        "experiment_dir": str(experiment_dir),
        "detector": {**model_block, "patch_size": 48, "augmentation_factor": 10,
                     "probability_threshold": 0.0},
        "teacher": {**model_block, "patch_size": 24, "augmentation_factor": 5},
        "student": {**model_block, "patch_size": 24, "augmentation_factor": 5,
                    "probability_threshold": 0.0, "temperature": 4.0,
                    "alpha": 0.4, "beta": 0.6},
        "trainer": {"learning_rate": 0.001, "adam_epsilon": 0.0001, "batch_size": 8,
                    "max_epochs": 100, "patience": 20, "learning_rate_factor": 0.1,
                    "learning_rate_period": 2, "minimum_learning_rate": 0.000001},
    }


def _write_preprocessed_manifest(manifest_dir: Path, subjects: list[dict]) -> None:
    """Write a schema-valid preprocessed manifest for the given subjects."""
    from microbleednet import manifests
    from microbleednet.manifests import (
        PreprocessedDatasetManifest,
        PreprocessedSubject,
    )

    now = manifests.timestamp()
    manifest = PreprocessedDatasetManifest(
        status=manifests.ManifestStatus.COMPLETE,
        created_at=now,
        updated_at=now,
        preprocess_parameters={},
        subjects=[
            PreprocessedSubject(
                subject_id=subject["subject_id"],
                volume_path=subject.get(
                    "volume_path", f"{subject['subject_id']}.nii.gz"
                ),
                mask_path=subject.get("mask_path"),
                bounding_box=[[0, 1], [0, 1], [0, 1]],
            )
            for subject in subjects
        ],
    )
    manifests.write_manifest(manifest_dir / "preprocessed.json", manifest)


def test_train_dry_run_writes_no_outputs(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    manifest_dir = dataset_dir / "manifests"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "preprocessed.json").write_text(
        json.dumps({"subjects": []}), encoding="utf-8"
    )
    experiment_dir = tmp_path / "experiment"
    config_path = tmp_path / "train.json"
    config_path.write_text(
        json.dumps(_train_config_dict(dataset_dir, experiment_dir)), encoding="utf-8"
    )

    result = runner.invoke(app, ["train", "--config", str(config_path), "--dry-run"])
    assert result.exit_code == 0, result.output
    assert not experiment_dir.exists()


def test_run_stage_writes_failed_manifest_and_reraises(tmp_path: Path) -> None:
    import pytest
    import torch

    from microbleednet.config import TrainCommandConfig
    from microbleednet.pipelines import train

    experiment_dir = tmp_path / "experiment"
    settings = TrainCommandConfig.model_validate(
        _train_config_dict(experiment_dir, experiment_dir)
    )
    runtime = train._stage_runtime(experiment_dir, torch.device("cpu"), "detector")

    def _failing_patcher(subject, **_kwargs):
        raise ValueError("synthetic patcher failure")

    with pytest.raises(ValueError, match="synthetic patcher failure"):
        train._run_stage(
            runtime,
            [{"subject_id": "s1"}],
            [{"subject_id": "s2"}],
            settings.trainer,
            settings,
            model=None,
            task=None,
            patcher=_failing_patcher,
            patcher_parameters={"patch_size": 48},
            dataset_class=None,
            augmentation_factor=1,
        )

    manifest = json.loads(runtime.manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["error"]
    assert manifest["manifest_type"] == "training_stage"
    assert manifest["schema_version"] == 1
    # A failed stage published no checkpoint.
    assert manifest["checkpoint_dir"] is None


def test_train_emits_provenance_before_training(tmp_path: Path) -> None:
    import pytest

    from microbleednet.config import TrainCommandConfig
    from microbleednet.pipelines import train

    dataset_dir = tmp_path / "dataset"
    manifest_dir = dataset_dir / "manifests"
    manifest_dir.mkdir(parents=True)
    subjects = [
        {"subject_id": f"s{index}", "mask_path": str(tmp_path / f"s{index}.nii.gz")}
        for index in range(5)
    ]
    _write_preprocessed_manifest(manifest_dir, subjects)
    experiment_dir = tmp_path / "experiment"
    settings = TrainCommandConfig.model_validate(
        {**_train_config_dict(dataset_dir, experiment_dir), "seed": 7}
    )

    # Provenance must be recorded before any training runs; stub the first stage
    # so the run fails immediately after the record is written.
    def _boom(*_args, **_kwargs):
        raise RuntimeError("stop after provenance")

    original = train.train_detector
    train.train_detector = _boom  # type: ignore[assignment]
    try:
        with pytest.raises(RuntimeError, match="stop after provenance"):
            train.execute(settings)
    finally:
        train.train_detector = original  # type: ignore[assignment]

    provenance_path = experiment_dir / "provenance.json"
    assert provenance_path.exists()
    record = json.loads(provenance_path.read_text(encoding="utf-8"))
    assert record["seed"] == 7
    assert record["configuration"] == settings.model_dump(mode="json")

    # The train/validation split is persisted before the first stage runs.
    split = json.loads(
        (experiment_dir / "manifests" / "split.json").read_text(encoding="utf-8")
    )
    assert split["manifest_type"] == "split"
    assert split["schema_version"] == 1
    assert sorted(split["train"] + split["validation"]) == [f"s{i}" for i in range(5)]
    assert not set(split["train"]) & set(split["validation"])


def test_persist_split_is_write_once(tmp_path: Path) -> None:
    import pytest

    from microbleednet.config import DataSplitConfig
    from microbleednet.manifests import PreprocessedSubject
    from microbleednet.pipelines import train

    path = tmp_path / "split.json"

    def _subjects(ids: list[str]) -> list[PreprocessedSubject]:
        return [
            PreprocessedSubject(
                subject_id=subject_id,
                volume_path=f"{subject_id}.nii.gz",
                mask_path=f"{subject_id}_mask.nii.gz",
                bounding_box=[[0, 1], [0, 1], [0, 1]],
            )
            for subject_id in ids
        ]

    datasplit = DataSplitConfig(test_size=0.4)
    train._persist_split(path, _subjects(["a", "b"]), _subjects(["c"]), datasplit)
    first = path.read_text(encoding="utf-8")

    # An identical rewrite is a no-op.
    train._persist_split(path, _subjects(["a", "b"]), _subjects(["c"]), datasplit)
    assert path.read_text(encoding="utf-8") == first

    # A conflicting split is a hard error, never a silent overwrite.
    with pytest.raises(ValueError, match="different split"):
        train._persist_split(path, _subjects(["a"]), _subjects(["b", "c"]), datasplit)


def test_infer_emits_provenance_before_prediction(tmp_path: Path) -> None:
    from microbleednet.config import InferCommandConfig
    from microbleednet.pipelines import infer
    from microbleednet.records import PredictionSummary

    model_block = {
        "initial_channels": 64,
        "input_channels": 2,
        "output_classes": 2,
        "dropout_rate": 0.5,
    }
    output_dir = tmp_path / "prediction"
    settings = InferCommandConfig.model_validate(
        {
            "volume_path": str(tmp_path / "volume.nii.gz"),
            "output_dir": str(output_dir),
            "detector_checkpoint": str(tmp_path / "detector.pth"),
            "student_checkpoint": str(tmp_path / "student.pth"),
            "detector": {**model_block, "patch_size": 48, "augmentation_factor": 10,
                         "probability_threshold": 0.0},
            "student": {**model_block, "patch_size": 24, "augmentation_factor": 5,
                        "probability_threshold": 0.0, "temperature": 4.0,
                        "alpha": 0.4, "beta": 0.6},
            "seed": 11,
        }
    )

    calls: dict[str, bool] = {}

    def _fake_predict(**_kwargs):
        # Provenance must already be on disk before any prediction is produced.
        calls["provenance_first"] = (output_dir / "provenance.json").exists()
        return PredictionSummary(
            subject_id="volume",
            mask_path=output_dir / "volume_prediction.nii.gz",
            probability_path=output_dir / "volume_probability.nii.gz",
            components_path=output_dir / "volume_components.json",
            component_count=0,
        )

    original = infer.predict_volume
    infer.predict_volume = _fake_predict  # type: ignore[assignment]
    try:
        infer.execute(settings)
    finally:
        infer.predict_volume = original  # type: ignore[assignment]

    assert calls["provenance_first"] is True
    record = json.loads((output_dir / "provenance.json").read_text(encoding="utf-8"))
    assert record["seed"] == 11
    assert record["configuration"] == settings.model_dump(mode="json")


def test_write_provenance_captures_config_and_seed(tmp_path: Path) -> None:
    import torch

    from microbleednet.config import PreprocessingConfig
    from microbleednet.provenance import ProvenanceRecord, write_provenance

    config = PreprocessingConfig()
    path = write_provenance(tmp_path, config, seed=1234, device=torch.device("cpu"))
    record = ProvenanceRecord.model_validate_json(path.read_text(encoding="utf-8"))
    assert record.seed == 1234
    assert record.configuration == config.model_dump(mode="json")
    assert record.device == "cpu"
