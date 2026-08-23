from pathlib import Path

import nibabel as nib

from ...core import io
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
    layout = DatasetLayout()
    raw_manifest = manifests.read_manifest(
        config.dataset_dir / layout.raw_manifest, RawDatasetManifest
    )

    volumes_dir = config.dataset_dir / layout.preprocessed_volumes_dir
    volumes_dir.mkdir(parents=True, exist_ok=True)

    masks_dir = config.dataset_dir / layout.preprocessed_masks_dir
    masks_dir.mkdir(parents=True, exist_ok=True)

    preprocessed_subjects = []

    for subject in raw_manifest.subjects:
        subject_id = subject.subject_id

        raw_volume = io.load_volume(Path(subject.volume_path))
        raw_mask = (
            io.load_volume(Path(subject.mask_path)) if subject.mask_path else None
        )

        output = processor.preprocess(raw_volume, raw_mask, config.preprocessing)

        preprocessed_volume = nib.Nifti1Image(output.image, output.geometry.affine)
        preprocessed_volume_path = (
            volumes_dir / f"{subject_id}{layout.volume_suffix}"
        )
        io.save_volume(preprocessed_volume, preprocessed_volume_path)

        preprocessed_mask_path = None
        if raw_mask is not None:
            if output.mask is None:
                raise ValueError(f"preprocessing returned no mask for {subject_id}")
            preprocessed_mask = nib.Nifti1Image(output.mask, output.geometry.affine)
            preprocessed_mask_path = (
                masks_dir / f"{subject_id}{layout.mask_suffix}"
            )
            io.save_volume(preprocessed_mask, preprocessed_mask_path)

        preprocessed_subjects.append(
            PreprocessedSubject(
                subject_id=subject_id,
                volume_path=str(preprocessed_volume_path.resolve()),
                mask_path=(
                    str(preprocessed_mask_path.resolve())
                    if preprocessed_mask_path is not None
                    else None
                ),
                bounding_box=[
                    list(bounds)
                    for bounds in zip(
                        output.transform.crop_start, output.transform.crop_stop
                    )
                ],
            )
        )

    # Publish the manifest once, after every subject is on disk.
    now = manifests.timestamp()
    preprocessed_manifest = PreprocessedDatasetManifest(
        status=ManifestStatus.COMPLETE,
        created_at=now,
        updated_at=now,
        preprocess_parameters={"modality": config.preprocessing},
        subjects=preprocessed_subjects,
    )
    manifests.write_manifest(
        config.dataset_dir / layout.preprocessed_manifest, preprocessed_manifest
    )
