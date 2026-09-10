from typing import cast

import nibabel as nib

from ...core import io
from ...core.datamodels import Modality
from ...core.engines import processor
from .. import manifests
from ..configs import PreprocessConfig
from ..layouts import DatasetLayout
from ..manifests import (
    ManifestStatus,
    PreprocessedDatasetManifest,
    PreprocessedSubject,
    RawDatasetManifest,
)


def execute(config: PreprocessConfig) -> None:
    layout = DatasetLayout(dataset_dir=config.dataset_dir)
    raw_manifest = RawDatasetManifest.read(layout.raw_manifest_path())

    volumes_dir = layout.preprocessed_volumes_path()
    volumes_dir.mkdir(parents=True, exist_ok=True)

    masks_dir = layout.preprocessed_masks_path()
    masks_dir.mkdir(parents=True, exist_ok=True)

    source_modalities = {
        source.source_id: source.modality for source in raw_manifest.sources
    }
    preprocessed_subjects = []

    for subject in raw_manifest.subjects:
        subject_id = subject.subject_id

        raw_volume = io.load_volume(subject.volume_path)
        raw_mask = io.load_volume(subject.mask_path)

        modality = cast(Modality, source_modalities[subject.source_id])
        preprocess_result = processor.preprocess(raw_volume, raw_mask, modality)

        preprocessed_volume = nib.Nifti1Image(
            preprocess_result.image, preprocess_result.affine
        )
        preprocessed_volume_path = (
            volumes_dir / f"{subject_id}{layout.volume_suffix}"
        )
        io.save_volume(preprocessed_volume, preprocessed_volume_path)

        preprocessed_mask = nib.Nifti1Image(
            preprocess_result.mask, preprocess_result.affine
        )
        preprocessed_mask_path = masks_dir / f"{subject_id}{layout.mask_suffix}"
        io.save_volume(preprocessed_mask, preprocessed_mask_path)

        preprocessed_subjects.append(
            PreprocessedSubject(
                subject_id=subject_id,
                volume_path=str(preprocessed_volume_path.resolve()),
                mask_path=str(preprocessed_mask_path.resolve()),
            )
        )

    # Publish the manifest once, after every subject is on disk.
    now = manifests.timestamp()
    preprocessed_manifest = PreprocessedDatasetManifest(
        status=ManifestStatus.COMPLETE,
        created_at=now,
        updated_at=now,
        subjects=preprocessed_subjects,
    )
    preprocessed_manifest.write(layout.preprocessed_manifest_path())
