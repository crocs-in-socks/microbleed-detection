"""Tests for the CLI command table: registration, wiring, and the generic
``describe`` machinery, plus end-to-end coverage of the ``index-data``
command it currently registers.
"""

import json
from pathlib import Path

import typer
from click.testing import Result
from pydantic import BaseModel, Field
from typer.testing import CliRunner

from microbleednet.cli.commands import CommandSpec, build_describe_command
from microbleednet.cli.entrypoint import app
from tests.support import create_volume_source, write_index_config

runner = CliRunner()


class _Nested(BaseModel):
    knob: int = Field(default=1, description="A nested knob.")


class _FakeConfig(BaseModel):
    top_level: str = Field(description="A top-level field.")
    nested: _Nested = Field(default_factory=_Nested)
    trailing: bool = Field(default=True, description="A trailing top-level field.")


def test_describe_renders_a_section_header_for_nested_config() -> None:
    """``build_describe_command`` groups fields by nested-model section.

    Exercised against a scratch Typer app with a synthetic nested config,
    since ``IndexDataConfig`` is flat and never triggers this grouping.
    """
    scratch_app = typer.Typer()

    # A second command keeps Typer from collapsing this scratch app down to a
    # single-command app, which would swallow the "describe" subcommand name.
    @scratch_app.command("noop")
    def noop() -> None:
        pass

    spec = CommandSpec(
        name="fake",
        help="Fake command for testing.",
        config=_FakeConfig,
        pipe="unused",
        dry_run_message=lambda s: "",
        success_message=lambda s, r: "",
    )
    build_describe_command(scratch_app, [spec])

    result = runner.invoke(scratch_app, ["describe", "fake"])
    assert result.exit_code == 0, result.output
    assert "nested" in result.output
    assert "knob" in result.output
    assert "trailing" in result.output


def test_index_data_help_points_to_describe() -> None:
    result = runner.invoke(app, ["index-data", "--help"])
    assert result.exit_code == 0, result.output
    assert "describe index-data" in " ".join(result.output.split())


def test_describe_index_data_lists_config_keys() -> None:
    result = runner.invoke(app, ["describe", "index-data"])
    assert result.exit_code == 0, result.output
    assert "dataset_dir" in result.output
    assert "volume_pattern" in result.output


def test_describe_rejects_unknown_command() -> None:
    result = runner.invoke(app, ["describe", "bogus"])
    assert result.exit_code != 0
    assert "unknown command" in result.output


def test_index_data_dry_run_does_not_write_manifest(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    source = create_volume_source(tmp_path, "first", ["subject_1"])
    config_path = tmp_path / "index.toml"
    write_index_config(
        config_path,
        dataset_dir=dataset_dir,
        input_dir=source,
        label_dir=tmp_path / "first_masks",
    )

    result = runner.invoke(
        app, ["index-data", "--config", str(config_path), "--dry-run"]
    )
    assert result.exit_code == 0, result.output
    assert "dry run" in result.output
    assert not (dataset_dir / "manifests" / "raw.json").exists()


def test_index_data_rejects_missing_label_dir(tmp_path: Path) -> None:
    config_path = tmp_path / "index.toml"
    write_index_config(
        config_path,
        label_dir=tmp_path / "missing_labels",
        mask_pattern="{subject_id}_mask.nii.gz",
    )

    result = runner.invoke(app, ["index-data", "--config", str(config_path)])
    assert result.exit_code != 0
    assert "label_dir" in result.output


def test_index_data_rejects_unmatched_subjects_when_masks_required(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "volumes"
    label_dir = tmp_path / "masks"
    input_dir.mkdir()
    label_dir.mkdir()
    (input_dir / "subject_1_volume.nii.gz").write_bytes(b"")
    (label_dir / "subject_2_mask.nii.gz").write_bytes(b"")
    config_path = tmp_path / "index.toml"
    write_index_config(
        config_path,
        input_dir=input_dir,
        label_dir=label_dir,
        mask_pattern="{subject_id}_mask.nii.gz",
    )

    result = runner.invoke(app, ["index-data", "--config", str(config_path)])
    assert result.exit_code != 0
    assert "unmatched subjects" in str(result.exception)


def test_preprocess_dry_run_accepts_indexed_dataset(tmp_path: Path) -> None:
    config_path = tmp_path / "preprocess.toml"
    config_path.write_text(
        f"dataset_dir = {json.dumps(str(tmp_path))}\n", encoding="utf-8"
    )

    result = runner.invoke(
        app, ["preprocess", "--config", str(config_path), "--dry-run"]
    )

    assert result.exit_code == 0, result.output
    assert "dry run" in result.output


def test_preprocess_rejects_missing_dataset_dir(tmp_path: Path) -> None:
    config_path = tmp_path / "preprocess.toml"
    config_path.write_text(
        f"dataset_dir = {json.dumps(str(tmp_path / 'missing'))}\n", encoding="utf-8"
    )

    result = runner.invoke(app, ["preprocess", "--config", str(config_path)])

    assert result.exit_code != 0
    assert "dataset_dir" in result.output


def _run_index_data(
    tmp_path: Path,
    dataset_dir: Path,
    input_dir: Path,
    source_id: str = "source",
) -> Result:
    config_path = tmp_path / f"index_{input_dir.name}.toml"
    write_index_config(
        config_path,
        dataset_dir=dataset_dir,
        input_dir=input_dir,
        label_dir=input_dir.with_name(f"{input_dir.name}_masks"),
        source_id=source_id,
    )
    return runner.invoke(app, ["index-data", "--config", str(config_path)])


def _read_raw_manifest(dataset_dir: Path) -> dict:
    manifest_path = dataset_dir / "manifests" / "raw.json"
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def test_index_data_accumulates_sources(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    first = create_volume_source(tmp_path, "first", ["subject_1", "subject_2"])
    second = create_volume_source(tmp_path, "second", ["subject_3"])

    assert _run_index_data(tmp_path, dataset_dir, first).exit_code == 0
    result = _run_index_data(tmp_path, dataset_dir, second)
    assert result.exit_code == 0, result.output

    manifest = _read_raw_manifest(dataset_dir)
    assert len(manifest["sources"]) == 2
    ids = [subject["subject_id"] for subject in manifest["subjects"]]
    assert ids == ["source_subject_1", "source_subject_2", "source_subject_3"]
    assert manifest["created_at"] <= manifest["updated_at"]


def test_index_data_rejects_duplicate_subject_across_sources(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    first = create_volume_source(tmp_path, "first", ["subject_1"])
    second = create_volume_source(tmp_path, "second", ["subject_1"])

    assert _run_index_data(tmp_path, dataset_dir, first).exit_code == 0
    result = _run_index_data(tmp_path, dataset_dir, second)
    assert result.exit_code != 0
    assert "source_subject_1" in str(result.exception)

    # The failed run must not have clobbered the existing manifest.
    manifest = _read_raw_manifest(dataset_dir)
    assert len(manifest["sources"]) == 1


def test_index_data_source_id_namespaces_subjects(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    first = create_volume_source(tmp_path, "first", ["subject_1"])
    second = create_volume_source(tmp_path, "second", ["subject_1"])

    assert _run_index_data(tmp_path, dataset_dir, first, "siteA").exit_code == 0
    result = _run_index_data(tmp_path, dataset_dir, second, "siteB")
    assert result.exit_code == 0, result.output

    manifest = _read_raw_manifest(dataset_dir)
    ids = [subject["subject_id"] for subject in manifest["subjects"]]
    assert ids == ["siteA_subject_1", "siteB_subject_1"]
    assert [source["source_id"] for source in manifest["sources"]] == [
        "siteA",
        "siteB",
    ]
