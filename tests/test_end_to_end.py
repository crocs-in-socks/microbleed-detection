import json
from pathlib import Path

import nibabel as nib
import numpy as np
from click.testing import Result
from typer.testing import CliRunner

from microbleednet.cli.entrypoint import app

runner = CliRunner()


def _toml_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    raise TypeError(f"unsupported TOML value: {value!r}")


def _toml_lines(data: dict, prefix: str = "") -> list[str]:
    lines: list[str] = []
    tables = {}
    table_arrays = {}
    for key, value in data.items():
        is_table_array = (
            isinstance(value, list)
            and bool(value)
            and all(isinstance(item, dict) for item in value)
        )
        if isinstance(value, dict):
            tables[key] = value
        elif is_table_array:
            table_arrays[key] = value
        else:
            lines.append(f"{key} = {_toml_value(value)}")
    for key, value in tables.items():
        header = f"{prefix}{key}"
        lines.extend(["", f"[{header}]", *_toml_lines(value, f"{header}.")])
    for key, items in table_arrays.items():
        header = f"{prefix}{key}"
        for item in items:
            lines.extend(["", f"[[{header}]]", *_toml_lines(item, f"{header}.")])
    return lines


def _write_config(path: Path, data: dict) -> None:
    """Serialize a config dict to TOML and write it to ``path``."""
    path.write_text("\n".join(_toml_lines(data)).strip() + "\n", encoding="utf-8")


def test_root_and_command_help() -> None:
    root = runner.invoke(app, ["--help"])
    assert root.exit_code == 0
    for command in ("preprocess", "train", "infer", "evaluate", "index-data"):
        result = runner.invoke(app, [command, "--help"])
        assert result.exit_code == 0, result.output


def test_command_help_points_to_describe() -> None:
    result = runner.invoke(app, ["infer", "--help"])
    assert result.exit_code == 0, result.output
    output = " ".join(result.output.split())
    assert "describe infer" in output


def test_describe_documents_config_keys() -> None:
    result = runner.invoke(app, ["describe", "infer"])
    assert result.exit_code == 0, result.output
    output = " ".join(result.output.split())
    assert "configuration keys" in output.lower()
    assert "[detector]" in output
    assert "initial_channels" in output
    assert "patch_batch_size" in output


def test_describe_train_documents_config_keys() -> None:
    result = runner.invoke(app, ["describe", "train"])
    assert result.exit_code == 0, result.output
    output = " ".join(result.output.split())
    assert "configuration keys" in output.lower()
    assert "[datasplit]" in output
    assert "test_size" in output


def test_describe_rejects_unknown_command() -> None:
    result = runner.invoke(app, ["describe", "nonsense"])
    assert result.exit_code != 0
    assert "unknown command" in result.output


def test_preprocess_dry_run_writes_no_outputs(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    manifest_dir = dataset_dir / "manifests"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "raw.json").write_text(
        json.dumps({"subjects": []}), encoding="utf-8"
    )
    config_path = tmp_path / "preprocess.toml"
    _write_config(config_path, {"dataset_dir": str(dataset_dir)})

    result = runner.invoke(
        app, ["preprocess", "--config", str(config_path), "--dry-run"]
    )
    assert result.exit_code == 0, result.output
    assert not (dataset_dir / "preprocessed").exists()


def _evaluate_config_blocks() -> tuple[dict, dict]:
    detector = {
        "architecture": {"initial_channels": 64, "output_classes": 2},
        "patch_size": 48,
        "augmentation_factor": 10,
        "probability_threshold": 0.0,
    }
    student = {
        "architecture": {
            "initial_channels": 64,
            "output_classes": 2,
            "dropout_rate": 0.5,
        },
        "patch_size": 24,
        "augmentation_factor": 5,
        "probability_threshold": 0.0,
        "temperature": 4.0,
        "alpha": 0.4,
        "beta": 0.6,
    }
    return detector, student


def _write_raw_manifest(dataset_dir: Path, subjects: list[dict]) -> None:
    """Write a schema-valid raw dataset manifest for the given subjects."""
    from microbleednet.orchestration import manifests
    from microbleednet.orchestration.manifests import (
        ManifestStatus,
        RawDatasetManifest,
        RawSource,
        RawSubject,
    )

    now = manifests.timestamp()
    manifest_dir = dataset_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest = RawDatasetManifest(
        status=ManifestStatus.COMPLETE,
        created_at=now,
        updated_at=now,
        sources=[
            RawSource(
                input_dir=str(dataset_dir),
                volume_pattern="{subject_id}_volume.nii.gz",
                added_on=now,
            )
        ],
        subjects=[
            RawSubject(
                subject_id=subject["subject_id"],
                volume_path=subject["volume_path"],
                mask_path=subject.get("mask_path"),
            )
            for subject in subjects
        ],
    )
    manifests.write_manifest(manifest_dir / "raw.json", manifest)


def _stub_inference(monkeypatch, lesion: tuple[slice, slice, slice]) -> None:
    """Stub model loading/prediction so evaluate scores without real checkpoints.

    Each stubbed prediction writes a mask and probability map with a lesion at
    ``lesion`` (probability 0.6 there), so scoring, FROC, and orchestration run
    against real NIfTIs without needing trained weights.
    """
    from microbleednet.orchestration.pipes import evaluate
    from microbleednet.orchestration.records import PredictionSummary

    class _FakeModel:
        def to(self, *_args):
            return self

        def eval(self):
            return self

    monkeypatch.setattr(evaluate, "CandidateDetector", lambda *_args: _FakeModel())
    monkeypatch.setattr(
        evaluate, "CandidateDiscriminatorStudent", lambda *_args: _FakeModel()
    )
    monkeypatch.setattr(evaluate.core_io, "load_model_weights", lambda *_args: None)

    def _fake_predict(detector, student, *, volume_path, output_dir, subject_id, **_):
        output_dir.mkdir(parents=True, exist_ok=True)
        mask = np.zeros((8, 8, 8), dtype=np.uint8)
        mask[lesion] = 1
        probability = np.zeros((8, 8, 8), dtype=np.float32)
        probability[lesion] = 0.6
        mask_path = output_dir / f"{subject_id}_prediction.nii.gz"
        probability_path = output_dir / f"{subject_id}_probability.nii.gz"
        nib.save(nib.Nifti1Image(mask, np.eye(4)), mask_path)
        nib.save(nib.Nifti1Image(probability, np.eye(4)), probability_path)
        return PredictionSummary(
            subject_id=subject_id,
            mask_path=mask_path,
            probability_path=probability_path,
            components_path=output_dir / f"{subject_id}_components.json",
            component_count=1,
        )

    monkeypatch.setattr(evaluate.infer, "predict_and_write", _fake_predict)


def _evaluate_dataset(tmp_path: Path) -> Path:
    """Build a dataset dir with one subject whose reference has a lesion."""
    dataset_dir = tmp_path / "dataset"
    lesion = (slice(2, 4), slice(2, 4), slice(2, 4))
    reference = np.zeros((8, 8, 8), dtype=np.uint8)
    reference[lesion] = 1
    volume_path = tmp_path / "s0_volume.nii.gz"
    mask_path = tmp_path / "s0_mask.nii.gz"
    volume = np.zeros((8, 8, 8), dtype=np.float32)
    nib.save(nib.Nifti1Image(volume, np.eye(4)), volume_path)
    nib.save(nib.Nifti1Image(reference, np.eye(4)), mask_path)
    _write_raw_manifest(
        dataset_dir,
        [
            {
                "subject_id": "s0",
                "volume_path": str(volume_path),
                "mask_path": str(mask_path),
            }
        ],
    )
    return dataset_dir


def _evaluate_config_dict(dataset_dir: Path, output_dir: Path) -> dict:
    detector, student = _evaluate_config_blocks()
    return {
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "detector": detector,
        "student": student,
        "detector_checkpoint": str(dataset_dir / "detector.pth"),
        "student_checkpoint": str(dataset_dir / "student.pth"),
    }


def test_synthetic_evaluate_workflow_completes(tmp_path: Path, monkeypatch) -> None:
    from microbleednet.orchestration.configs import EvaluateConfig
    from microbleednet.orchestration.pipes import evaluate

    dataset_dir = _evaluate_dataset(tmp_path)
    output_dir = tmp_path / "report"
    # The stubbed prediction places its lesion where the reference has one.
    _stub_inference(monkeypatch, (slice(2, 4), slice(2, 4), slice(2, 4)))

    settings = EvaluateConfig.model_validate(
        _evaluate_config_dict(dataset_dir, output_dir)
    )
    report = evaluate.execute(settings)

    assert report["aggregate"]["true_positives"] == 1
    assert (output_dir / "evaluation.json").is_file()
    # Predictions land under a per-subject predictions directory.
    assert (output_dir / "predictions" / "s0" / "s0_prediction.nii.gz").is_file()
    # Without froc_thresholds the sweep is not produced.
    assert not (output_dir / "froc.json").exists()


def test_evaluate_with_froc_thresholds_writes_sweep(
    tmp_path: Path, monkeypatch
) -> None:
    from microbleednet.orchestration.configs import EvaluateConfig
    from microbleednet.orchestration.pipes import evaluate

    dataset_dir = _evaluate_dataset(tmp_path)
    output_dir = tmp_path / "report"
    # Lesion probability is 0.6: recovered at 0.5, dropped at 0.9.
    _stub_inference(monkeypatch, (slice(2, 4), slice(2, 4), slice(2, 4)))

    config = _evaluate_config_dict(dataset_dir, output_dir)
    config["froc_thresholds"] = [0.5, 0.9]
    settings = EvaluateConfig.model_validate(config)
    evaluate.execute(settings)

    points = json.loads((output_dir / "froc.json").read_text(encoding="utf-8"))
    by_threshold = {point["threshold"]: point for point in points}
    assert by_threshold[0.5]["true_positives"] == 1
    assert by_threshold[0.9]["true_positives"] == 0
    assert (output_dir / "froc.csv").exists()


def test_evaluate_requires_checkpoints_or_experiment_dir(tmp_path: Path) -> None:
    dataset_dir = _evaluate_dataset(tmp_path)
    config_path = tmp_path / "evaluate.toml"
    config = _evaluate_config_dict(dataset_dir, tmp_path / "report")
    del config["detector_checkpoint"]
    del config["student_checkpoint"]
    _write_config(config_path, config)

    result = runner.invoke(app, ["evaluate", "--config", str(config_path)])
    assert result.exit_code != 0
    # Rich wraps the error across box borders, so collapse whitespace first.
    assert "experiment_dir" in "".join(result.output.split())


def test_evaluate_rejects_missing_raw_manifest(tmp_path: Path) -> None:
    config_path = tmp_path / "evaluate.toml"
    _write_config(
        config_path, _evaluate_config_dict(tmp_path / "empty_dataset", tmp_path / "out")
    )

    result = runner.invoke(app, ["evaluate", "--config", str(config_path)])
    assert result.exit_code != 0
    # Rich wraps the error across box borders, so collapse whitespace first.
    assert "rawmanifest" in "".join(result.output.split())


def test_index_data_rejects_pattern_without_single_placeholder(tmp_path: Path) -> None:
    config_path = tmp_path / "index.toml"
    _write_config(
        config_path,
        {
            "dataset_dir": str(tmp_path / "dataset"),
            "input_dir": str(tmp_path),
            "volume_pattern": "volume.nii.gz",  # missing {subject_id}
            "require_masks": False,
        },
    )

    result = runner.invoke(app, ["index-data", "--config", str(config_path)])
    assert result.exit_code != 0
    assert "subject_id" in result.output


def test_index_data_requires_label_dir_when_masks_required(tmp_path: Path) -> None:
    config_path = tmp_path / "index.toml"
    _write_config(
        config_path,
        {
            "dataset_dir": str(tmp_path / "dataset"),
            "input_dir": str(tmp_path),
            "volume_pattern": "{subject_id}.nii.gz",
            # require_masks defaults to true, but no label_dir is provided.
        },
    )

    result = runner.invoke(app, ["index-data", "--config", str(config_path)])
    assert result.exit_code != 0
    assert "label_dir" in result.output


def _index_source(tmp_path: Path, name: str, subject_ids: list[str]) -> Path:
    """Create a source directory of volumes and return it."""
    source_dir = tmp_path / name
    source_dir.mkdir()
    for subject_id in subject_ids:
        (source_dir / f"{subject_id}_volume.nii.gz").write_bytes(b"")
    return source_dir


def _run_index_data(
    tmp_path: Path,
    dataset_dir: Path,
    input_dir: Path,
    source_id: str | None = None,
) -> Result:
    config_path = tmp_path / f"index_{input_dir.name}.toml"
    config: dict = {
        "dataset_dir": str(dataset_dir),
        "input_dir": str(input_dir),
        "volume_pattern": "{subject_id}_volume.nii.gz",
        "require_masks": False,
    }
    if source_id is not None:
        config["source_id"] = source_id
    _write_config(config_path, config)
    return runner.invoke(app, ["index-data", "--config", str(config_path)])


def _read_raw_manifest(dataset_dir: Path) -> dict:
    manifest_path = dataset_dir / "manifests" / "raw.json"
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def test_index_data_accumulates_sources(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    first = _index_source(tmp_path, "first", ["subject_1", "subject_2"])
    second = _index_source(tmp_path, "second", ["subject_3"])

    assert _run_index_data(tmp_path, dataset_dir, first).exit_code == 0
    result = _run_index_data(tmp_path, dataset_dir, second)
    assert result.exit_code == 0, result.output

    manifest = _read_raw_manifest(dataset_dir)
    assert len(manifest["sources"]) == 2
    ids = [subject["subject_id"] for subject in manifest["subjects"]]
    assert ids == ["subject_1", "subject_2", "subject_3"]
    assert manifest["created_at"] <= manifest["updated_at"]


def test_index_data_rejects_duplicate_subject_across_sources(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    first = _index_source(tmp_path, "first", ["subject_1"])
    second = _index_source(tmp_path, "second", ["subject_1"])

    assert _run_index_data(tmp_path, dataset_dir, first).exit_code == 0
    result = _run_index_data(tmp_path, dataset_dir, second)
    assert result.exit_code != 0
    assert "subject_1" in str(result.exception)

    # The failed run must not have clobbered the existing manifest.
    manifest = _read_raw_manifest(dataset_dir)
    assert len(manifest["sources"]) == 1


def test_index_data_source_id_namespaces_subjects(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    first = _index_source(tmp_path, "first", ["subject_1"])
    second = _index_source(tmp_path, "second", ["subject_1"])

    assert _run_index_data(tmp_path, dataset_dir, first, "siteA").exit_code == 0
    result = _run_index_data(tmp_path, dataset_dir, second, "siteB")
    assert result.exit_code == 0, result.output

    manifest = _read_raw_manifest(dataset_dir)
    ids = [subject["subject_id"] for subject in manifest["subjects"]]
    assert ids == ["siteA_subject_1", "siteB_subject_1"]
    assert [source["source_id"] for source in manifest["sources"]] == ["siteA", "siteB"]


def test_index_data_rejects_invalid_source_id(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    source = _index_source(tmp_path, "first", ["subject_1"])

    result = _run_index_data(tmp_path, dataset_dir, source, "site/A")
    assert result.exit_code != 0
    assert "source_id" in result.output


def _train_config_dict(dataset_dir: Path, experiment_dir: Path) -> dict:
    detector_architecture = {"initial_channels": 64, "output_classes": 2}
    classifier_architecture = {**detector_architecture, "dropout_rate": 0.5}
    return {
        "dataset_dir": str(dataset_dir),
        "experiment_dir": str(experiment_dir),
        "detector": {
            "architecture": detector_architecture,
            "patch_size": 48,
            "augmentation_factor": 10,
            "probability_threshold": 0.0,
        },
        "teacher": {
            "architecture": classifier_architecture,
            "patch_size": 24,
            "augmentation_factor": 5,
        },
        "student": {
            "architecture": classifier_architecture,
            "patch_size": 24,
            "augmentation_factor": 5,
            "probability_threshold": 0.0,
            "temperature": 4.0,
            "alpha": 0.4,
            "beta": 0.6,
        },
        "trainer": {
            "learning_rate": 0.001,
            "adam_epsilon": 0.0001,
            "batch_size": 8,
            "max_epochs": 100,
            "patience": 20,
            "learning_rate_factor": 0.1,
            "learning_rate_period": 2,
            "minimum_learning_rate": 0.000001,
        },
    }


def _write_preprocessed_manifest(manifest_dir: Path, subjects: list[dict]) -> None:
    """Write a schema-valid preprocessed manifest for the given subjects."""
    from microbleednet.orchestration import manifests
    from microbleednet.orchestration.manifests import (
        ManifestStatus,
        PreprocessedDatasetManifest,
        PreprocessedSubject,
    )

    now = manifests.timestamp()
    manifest = PreprocessedDatasetManifest(
        status=ManifestStatus.COMPLETE,
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
    subjects = [
        {"subject_id": f"s{index}", "mask_path": str(tmp_path / f"s{index}.nii.gz")}
        for index in range(2)
    ]
    _write_preprocessed_manifest(manifest_dir, subjects)
    experiment_dir = tmp_path / "experiment"
    config_path = tmp_path / "train.toml"
    _write_config(config_path, _train_config_dict(dataset_dir, experiment_dir))

    result = runner.invoke(app, ["train", "--config", str(config_path), "--dry-run"])
    assert result.exit_code == 0, result.output
    assert not experiment_dir.exists()


def test_train_rejects_subjects_without_masks(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    manifest_dir = dataset_dir / "manifests"
    manifest_dir.mkdir(parents=True)
    subjects = [
        {"subject_id": "s0", "mask_path": str(tmp_path / "s0.nii.gz")},
        {"subject_id": "s1"},  # no mask_path
    ]
    _write_preprocessed_manifest(manifest_dir, subjects)
    experiment_dir = tmp_path / "experiment"
    config_path = tmp_path / "train.toml"
    _write_config(config_path, _train_config_dict(dataset_dir, experiment_dir))

    result = runner.invoke(app, ["train", "--config", str(config_path), "--dry-run"])
    assert result.exit_code != 0
    assert "requires masks" in result.output
    assert "s1" in result.output
    assert not experiment_dir.exists()


def test_run_stage_writes_failed_manifest_and_reraises(tmp_path: Path) -> None:
    import pytest
    import torch

    from microbleednet.orchestration import patching
    from microbleednet.orchestration.configs import TrainConfig
    from microbleednet.orchestration.pipes import train

    experiment_dir = tmp_path / "experiment"
    settings = TrainConfig.model_validate(
        _train_config_dict(experiment_dir, experiment_dir)
    )
    runtime = train.stage_runtime(experiment_dir, torch.device("cpu"), "detector")

    class _FailingPatcher(patching.Patcher):
        def collect(self, *_args, **_kwargs):
            raise ValueError("synthetic patcher failure")

    failing_patcher = _FailingPatcher(patch_size=1)
    with pytest.raises(ValueError, match="synthetic patcher failure"):
        train.run_stage(
            runtime,
            [{"subject_id": "s1"}],
            [{"subject_id": "s2"}],
            settings,
            model=None,
            task=None,
            patcher=failing_patcher,
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

    from microbleednet.orchestration.configs import TrainConfig
    from microbleednet.orchestration.pipes import train

    dataset_dir = tmp_path / "dataset"
    manifest_dir = dataset_dir / "manifests"
    manifest_dir.mkdir(parents=True)
    subjects = [
        {"subject_id": f"s{index}", "mask_path": str(tmp_path / f"s{index}.nii.gz")}
        for index in range(5)
    ]
    _write_preprocessed_manifest(manifest_dir, subjects)
    experiment_dir = tmp_path / "experiment"
    settings = TrainConfig.model_validate(
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

    from microbleednet.orchestration.configs import DataSplitConfig
    from microbleednet.orchestration.manifests import PreprocessedSubject
    from microbleednet.orchestration.pipes import train

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
    train.persist_split(path, _subjects(["a", "b"]), _subjects(["c"]), datasplit)
    first = path.read_text(encoding="utf-8")

    # An identical rewrite is a no-op.
    train.persist_split(path, _subjects(["a", "b"]), _subjects(["c"]), datasplit)
    assert path.read_text(encoding="utf-8") == first

    # A conflicting split is a hard error, never a silent overwrite.
    with pytest.raises(ValueError, match="different split"):
        train.persist_split(path, _subjects(["a"]), _subjects(["b", "c"]), datasplit)


def test_infer_emits_provenance_before_prediction(tmp_path: Path) -> None:
    from microbleednet.orchestration.configs import InferConfig
    from microbleednet.orchestration.pipes import infer
    from microbleednet.orchestration.records import PredictionSummary

    output_dir = tmp_path / "prediction"
    settings = InferConfig.model_validate(
        {
            "volume_path": str(tmp_path / "volume.nii.gz"),
            "output_dir": str(output_dir),
            "detector_checkpoint": str(tmp_path / "detector.pth"),
            "student_checkpoint": str(tmp_path / "student.pth"),
            "detector": {
                "architecture": {"initial_channels": 64, "output_classes": 2},
                "patch_size": 48,
                "augmentation_factor": 10,
                "probability_threshold": 0.0,
            },
            "student": {
                "architecture": {
                    "initial_channels": 64,
                    "output_classes": 2,
                    "dropout_rate": 0.5,
                },
                "patch_size": 24,
                "augmentation_factor": 5,
                "probability_threshold": 0.0,
                "temperature": 4.0,
                "alpha": 0.4,
                "beta": 0.6,
            },
            "seed": 11,
        }
    )

    calls: dict[str, bool] = {}

    class _FakeModel:
        def to(self, *_args):
            return self

        def eval(self):
            return self

    def _fake_predict(*_args, **_kwargs):
        # Provenance must already be on disk before any prediction is produced.
        calls["provenance_first"] = (output_dir / "provenance.json").exists()
        return PredictionSummary(
            subject_id="volume",
            mask_path=output_dir / "volume_prediction.nii.gz",
            probability_path=output_dir / "volume_probability.nii.gz",
            components_path=output_dir / "volume_components.json",
            component_count=0,
        )

    original = infer.predict_and_write
    infer.predict_and_write = _fake_predict  # type: ignore[assignment]
    original_detector = infer.CandidateDetector
    original_student = infer.CandidateDiscriminatorStudent
    original_load = infer.core_io.load_model_weights
    infer.CandidateDetector = lambda *_args: _FakeModel()  # type: ignore[assignment]
    infer.CandidateDiscriminatorStudent = lambda *_args: _FakeModel()  # type: ignore[assignment]
    infer.core_io.load_model_weights = lambda *_args: None  # type: ignore[assignment]
    try:
        infer.execute(settings)
    finally:
        infer.predict_and_write = original  # type: ignore[assignment]
        infer.CandidateDetector = original_detector  # type: ignore[assignment]
        infer.CandidateDiscriminatorStudent = original_student  # type: ignore[assignment]
        infer.core_io.load_model_weights = original_load  # type: ignore[assignment]

    assert calls["provenance_first"] is True
    record = json.loads((output_dir / "provenance.json").read_text(encoding="utf-8"))
    assert record["seed"] == 11
    assert record["configuration"] == settings.model_dump(mode="json")


def test_apply_postprocessing_drops_subthreshold_components() -> None:
    from microbleednet.orchestration.configs import PostprocessingConfig
    from microbleednet.orchestration.pipes.infer import _apply_postprocessing

    # A compact cube (large volume, low eccentricity) survives; a single stray
    # voxel is dropped for being under the minimum volume.
    component_mask = np.zeros((12, 12, 12), dtype=np.uint8)
    component_mask[2:6, 2:6, 2:6] = 1
    component_mask[9, 9, 9] = 1
    probability_map = np.where(component_mask > 0, 0.8, 0.0).astype(np.float32)
    brain_mask = np.ones_like(component_mask)
    postprocessing = PostprocessingConfig(
        minimum_volume_mm3=2.5,
        maximum_eccentricity=1.0,
        minimum_boundary_distance_mm=0.0,
    )

    accepted_mask, accepted_probability, filtered = _apply_postprocessing(
        component_mask,
        probability_map,
        brain_mask,
        spacing=(1.0, 1.0, 1.0),
        postprocessing=postprocessing,
        subject_id="synthetic",
    )

    # The cube (64 voxels) survives; the single voxel does not.
    assert accepted_mask[3, 3, 3] == 1
    assert accepted_mask[9, 9, 9] == 0
    assert accepted_probability[9, 9, 9] == 0.0
    reasons = {
        component["component_id"]: component["rejection_reasons"]
        for component in filtered
    }
    assert any("volume" in value for value in reasons.values())


def test_write_provenance_captures_config_and_seed(tmp_path: Path) -> None:
    import torch

    from microbleednet.orchestration.provenance import (
        ProvenanceRecord,
        write_provenance,
    )

    config = {"modality": "T2*-GRE"}
    path = write_provenance(tmp_path, config, seed=1234, device=torch.device("cpu"))
    record = ProvenanceRecord.model_validate_json(path.read_text(encoding="utf-8"))
    assert record.seed == 1234
    assert record.configuration == config
    assert record.device == "cpu"
