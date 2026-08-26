from pathlib import Path

import pytest

from microbleednet.orchestration.manifests import (
    ManifestStatus,
    RawDatasetManifest,
    RawSource,
    RawSubject,
)
from microbleednet.orchestration.pipes.index_data import (
    build_subject_map,
    compute_paths,
    extract_subject_id,
    index_source,
    merge_source,
)
from tests.support import create_volume_source, make_index_config


def test_extract_subject_id_matches_relative_pattern(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "subject_1_volume.nii.gz"

    subject_id = extract_subject_id(tmp_path, path, "nested/{subject_id}_volume.nii.gz")

    assert subject_id == "subject_1"


def test_extract_subject_id_returns_none_for_nonmatch(tmp_path: Path) -> None:
    path = tmp_path / "subject_1_mask.nii.gz"

    subject_id = extract_subject_id(tmp_path, path, "{subject_id}_volume.nii.gz")

    assert subject_id is None


def test_compute_paths_treats_pattern_text_literally(tmp_path: Path) -> None:
    expected = tmp_path / "scan[1]_subject_1.nii.gz"
    expected.write_bytes(b"")
    (tmp_path / "scan1_subject_2.nii.gz").write_bytes(b"")

    paths = compute_paths(tmp_path, "scan[1]_{subject_id}.nii.gz")

    assert paths == [expected]


def test_build_subject_map_rejects_empty_subject_id(tmp_path: Path) -> None:
    path = tmp_path / "_volume.nii.gz"
    path.write_bytes(b"")
    with pytest.raises(ValueError, match="empty ID"):
        build_subject_map(tmp_path, [path], "{subject_id}_volume.nii.gz")


def test_build_subject_map_rejects_duplicate_subject_id(tmp_path: Path) -> None:
    path = tmp_path / "subject_1_volume.nii.gz"
    path.write_bytes(b"")
    with pytest.raises(ValueError, match="duplicate subject ID"):
        build_subject_map(tmp_path, [path, path], "{subject_id}_volume.nii.gz")


def test_index_source_sorts_naturally_and_namespaces_subjects(tmp_path: Path) -> None:
    source_dir = create_volume_source(tmp_path, "source", ["subject_10", "subject_2"])
    config = make_index_config(
        tmp_path, input_dir=source_dir, source_id="siteA", modality="SWI"
    )

    source, subjects, unmatched_volumes, unmatched_masks = index_source(
        config, "2026-01-01T00:00:00+00:00"
    )

    assert [subject.subject_id for subject in subjects] == [
        "siteA_subject_2",
        "siteA_subject_10",
    ]
    assert source.source_id == "siteA"
    assert source.modality == "SWI"
    assert {subject.source_id for subject in subjects} == {"siteA"}
    assert set(unmatched_volumes) == {"siteA_subject_2", "siteA_subject_10"}
    assert unmatched_masks == []


def test_merge_source_preserves_creation_and_combines_state() -> None:
    existing_source = _source("first", "2026-01-01T00:00:00+00:00")
    existing = RawDatasetManifest(
        status=ManifestStatus.COMPLETE,
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
        sources=[existing_source],
        subjects=[
            RawSubject(
                subject_id="subject_10",
                source_id="first",
                volume_path="/subject_10",
            )
        ],
        unmatched_volumes=["subject_10"],
    )
    new_source = _source("second", "2026-01-02T00:00:00+00:00")

    merged = merge_source(
        existing,
        source=new_source,
        subjects=[
            RawSubject(
                subject_id="subject_2",
                source_id="second",
                volume_path="/subject_2",
            )
        ],
        unmatched_volumes=["subject_2"],
        unmatched_masks=["mask_only"],
        now="2026-01-02T00:00:00+00:00",
    )

    assert merged.created_at == existing.created_at
    assert merged.updated_at == "2026-01-02T00:00:00+00:00"
    assert [subject.subject_id for subject in merged.subjects] == [
        "subject_2",
        "subject_10",
    ]
    assert merged.unmatched_volumes == ["subject_10", "subject_2"]
    assert merged.unmatched_masks == ["mask_only"]


def _source(name: str, added_on: str) -> RawSource:
    return RawSource(
        input_dir=f"/{name}",
        volume_pattern="{subject_id}_volume.nii.gz",
        source_id=name,
        added_on=added_on,
    )
