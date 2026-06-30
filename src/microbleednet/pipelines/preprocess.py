import json
from pathlib import Path
from datetime import datetime 

from . import constants
from ..core import utils
from ..core.engines import processor

def execute(
    dataset_dir: Path,
    preprocessor_parameters: dict
) -> None:
    raw_manifest_path = dataset_dir / constants.manifests.raw
    with open(raw_manifest_path, "r") as raw_manifest_file:
        raw_manifest_content = json.load(raw_manifest_file)
        raw_subjects = raw_manifest_content.get("subjects", [])

    volumes_dir = dataset_dir / constants.preprocess.volumes_dir
    volumes_dir.mkdir(parents=True, exist_ok=True)

    masks_dir = dataset_dir / constants.preprocess.masks_dir
    masks_dir.mkdir(parents=True, exist_ok=True)

    preprocessed_subjects = []

    try:
        for subject in raw_subjects:
            subject_id = subject["subject_id"]
            raw_volume_path = subject["volume_path"]
            raw_mask_path = subject.get("mask_path")

            raw_volume = utils.load_volume(raw_volume_path)
            raw_mask = utils.load_volume(raw_mask_path) if raw_mask_path else None

            output = processor.preprocess(raw_volume, raw_mask, **preprocessor_parameters)

            preprocessed_volume = utils.numpy_to_nifti(output["volume"])
            preprocessed_volume_path = volumes_dir / f"{subject_id}{constants.preprocess.volume_suffix}"
            utils.save_volume(preprocessed_volume, preprocessed_volume_path)

            if raw_mask is not None:
                preprocessed_mask = utils.numpy_to_nifti(output["mask"])
                preprocessed_mask_path = masks_dir / f"{subject_id}{constants.preprocess.mask_suffix}"
                utils.save_volume(preprocessed_mask, preprocessed_mask_path)

            preprocessed_subject = {
                "subject_id": subject_id,
                "volume_path": str(preprocessed_volume_path.resolve()),
                "mask_path": str(preprocessed_mask_path.resolve()) if raw_mask is not None else None,
                "bounding_box": output["bounding_box"]
            }

            preprocessed_subjects.append(preprocessed_subject)

    finally:
        preprocessed_manifest_path = dataset_dir / constants.manifests.preprocessed
        preprocessed_manifest_data = {
            "stage": "preprocessed",
            "created_on": datetime.now().isoformat(),
            "preprocess_parameters": preprocessor_parameters,
            "subjects": preprocessed_subjects
        }

        with open(preprocessed_manifest_path, mode="w") as preprocessed_manifest_file:
            json.dump(preprocessed_manifest_data, preprocessed_manifest_file)