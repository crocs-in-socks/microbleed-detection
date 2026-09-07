from pathlib import Path

import pytest
from pydantic import ValidationError

from microbleednet.orchestration import atomic_io
from microbleednet.orchestration.manifests import (
    Manifest,
    ManifestStatus,
    RawDatasetManifest,
    timestamp,
)


def _envelope(**overrides: object) -> dict:
    now = timestamp()
    base = {
        "schema_version": 1,
        "status": ManifestStatus.COMPLETE.value,
        "created_at": now,
        "updated_at": now,
    }
    base.update(overrides)
    return base


def test_failed_manifest_requires_error_message() -> None:
    with pytest.raises(ValidationError, match="nonempty error string"):
        Manifest.model_validate(_envelope(status=ManifestStatus.FAILED.value))


def test_non_failed_manifest_rejects_error_message() -> None:
    with pytest.raises(ValidationError, match="only a failed manifest"):
        Manifest.model_validate(_envelope(error="boom"))


def test_raw_dataset_manifest_rejects_duplicate_subject_ids() -> None:
    subject = {
        "subject_id": "s1",
        "source_id": "source",
        "volume_path": "/a",
    }
    with pytest.raises(ValidationError, match="duplicate subject ID"):
        RawDatasetManifest.model_validate(
            _envelope(
                manifest_type="raw_dataset", sources=[], subjects=[subject, subject]
            )
        )


def test_read_manifest_rejects_unversioned_payload(tmp_path: Path) -> None:
    path = tmp_path / "raw.json"
    atomic_io.write_json_atomic(path, {"status": "complete"})
    with pytest.raises(ValueError, match="not a versioned manifest"):
        RawDatasetManifest.read(path)


def test_read_manifest_rejects_incomplete_status(tmp_path: Path) -> None:
    path = tmp_path / "raw.json"
    atomic_io.write_json_atomic(
        path,
        _envelope(
            status=ManifestStatus.RUNNING.value,
            manifest_type="raw_dataset",
            sources=[],
            subjects=[],
        ),
    )
    with pytest.raises(ValueError, match="a consumer may only read a complete"):
        RawDatasetManifest.read(path)


def test_read_manifest_rejects_payload_that_fails_schema_validation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "raw.json"
    # schema_version is present, but required manifest fields are missing.
    atomic_io.write_json_atomic(path, _envelope())
    with pytest.raises(ValueError, match="invalid manifest at"):
        RawDatasetManifest.read(path)
