from pathlib import Path

import nibabel as nib

from .. import manifests
from ..config import PreprocessingConfig
from ..core import utils
from ..core.engines import processor
from ..manifests import (
    PreprocessedDatasetManifest,
    PreprocessedSubject,
    RawDatasetManifest,
)
from . import constants


def execute(
    dataset_dir: Path,
    preprocessing: PreprocessingConfig,
) -> None:
    preprocessor_parameters = preprocessing.model_dump()
    raw_manifest = manifests.read_manifest(
        dataset_dir / constants.manifests.raw, RawDatasetManifest
    )

    volumes_dir = dataset_dir / constants.preprocess.volumes_dir
    volumes_dir.mkdir(parents=True, exist_ok=True)

    masks_dir = dataset_dir / constants.preprocess.masks_dir
    masks_dir.mkdir(parents=True, exist_ok=True)

    preprocessed_subjects = []

    for subject in raw_manifest.subjects:
        subject_id = subject.subject_id
        raw_volume_path = subject.volume_path
        raw_mask_path = subject.mask_path

        raw_volume = utils.load_volume(Path(raw_volume_path))
        raw_mask = utils.load_volume(Path(raw_mask_path)) if raw_mask_path else None

        output = processor.preprocess(raw_volume, raw_mask, **preprocessor_parameters)

        preprocessed_volume = nib.Nifti1Image(output.image, output.geometry.affine)
        preprocessed_volume_path = volumes_dir / f"{subject_id}{constants.preprocess.volume_suffix}"
        utils.save_volume(preprocessed_volume, preprocessed_volume_path)

        preprocessed_mask_path = None
        if raw_mask is not None:
            if output.mask is None:
                raise ValueError(f"preprocessing returned no mask for {subject_id}")
            preprocessed_mask = nib.Nifti1Image(output.mask, output.geometry.affine)
            preprocessed_mask_path = masks_dir / f"{subject_id}{constants.preprocess.mask_suffix}"
            utils.save_volume(preprocessed_mask, preprocessed_mask_path)

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
        status=manifests.ManifestStatus.COMPLETE,
        created_at=now,
        updated_at=now,
        preprocess_parameters=preprocessor_parameters,
        subjects=preprocessed_subjects,
    )
    manifests.write_manifest(
        dataset_dir / constants.manifests.preprocessed, preprocessed_manifest
    )