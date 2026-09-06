from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from microbleednet.core.datamodels import PreprocessResult
from microbleednet.orchestration.configs import PreprocessConfig
from microbleednet.orchestration.layouts import DatasetLayout
from microbleednet.orchestration.manifests import (
    ManifestStatus,
    PreprocessedDatasetManifest,
    RawDatasetManifest,
    RawSource,
    RawSubject,
    read_manifest,
    timestamp,
    write_manifest,
)
from microbleednet.orchestration.pipes import preprocess


def _write_raw_dataset(dataset_dir: Path, with_unmasked_subject: bool = True) -> None:
    raw_dir = dataset_dir / "raw"
    raw_dir.mkdir(parents=True)
    volume_path = raw_dir / "masked_volume.nii.gz"
    mask_path = raw_dir / "masked_mask.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((2, 2, 2)), np.eye(4)), volume_path)
    nib.save(nib.Nifti1Image(np.ones((2, 2, 2)), np.eye(4)), mask_path)
    subjects = [
        RawSubject(
            subject_id="source_masked",
            source_id="source",
            volume_path=str(volume_path),
            mask_path=str(mask_path),
        )
    ]
    if with_unmasked_subject:
        unmasked_path = raw_dir / "unmasked_volume.nii.gz"
        nib.save(nib.Nifti1Image(np.ones((2, 2, 2)), np.eye(4)), unmasked_path)
        subjects.append(
            RawSubject(
                subject_id="source_unmasked",
                source_id="source",
                volume_path=str(unmasked_path),
            )
        )
    now = timestamp()
    manifest = RawDatasetManifest(
        status=ManifestStatus.COMPLETE,
        created_at=now,
        updated_at=now,
        sources=[
            RawSource(
                input_dir=str(raw_dir),
                volume_pattern="{subject_id}_volume.nii.gz",
                source_id="source",
                modality="QSM",
                added_on=now,
            )
        ],
        subjects=subjects,
    )
    layout = DatasetLayout(dataset_dir=dataset_dir)
    write_manifest(layout.raw_manifest_path(), manifest)


def test_execute_writes_volumes_masks_and_complete_manifest(
    tmp_path: Path, monkeypatch
) -> None:
    dataset_dir = tmp_path / "dataset"
    _write_raw_dataset(dataset_dir)

    def fake_preprocess(volume, mask, modality) -> PreprocessResult:
        assert modality == "QSM"
        output_mask = None if mask is None else np.ones((1, 1, 1), dtype=np.uint8)
        return PreprocessResult(np.ones((1, 1, 1)), output_mask, np.eye(4))

    monkeypatch.setattr(preprocess.processor, "preprocess", fake_preprocess)

    preprocess.execute(PreprocessConfig(dataset_dir=dataset_dir))

    layout = DatasetLayout(dataset_dir=dataset_dir)
    manifest = read_manifest(
        layout.preprocessed_manifest_path(), PreprocessedDatasetManifest
    )
    assert manifest.status is ManifestStatus.COMPLETE
    assert [subject.subject_id for subject in manifest.subjects] == [
        "source_masked",
        "source_unmasked",
    ]
    assert manifest.subjects[0].mask_path is not None
    assert manifest.subjects[1].mask_path is None
    for subject in manifest.subjects:
        assert Path(subject.volume_path).is_file()
        if subject.mask_path is not None:
            assert Path(subject.mask_path).is_file()


def test_execute_rejects_missing_processed_mask(tmp_path: Path, monkeypatch) -> None:
    dataset_dir = tmp_path / "dataset"
    _write_raw_dataset(dataset_dir, with_unmasked_subject=False)
    monkeypatch.setattr(
        preprocess.processor,
        "preprocess",
        lambda volume, mask, modality: PreprocessResult(
            np.ones((1, 1, 1)), None, np.eye(4)
        ),
    )

    with pytest.raises(ValueError, match="returned no mask for source_masked"):
        preprocess.execute(PreprocessConfig(dataset_dir=dataset_dir))