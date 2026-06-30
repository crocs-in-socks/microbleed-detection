from pathlib import Path


class manifests:
    raw = Path("manifests/raw.json")
    preprocessed = Path("manifests/preprocessed.json")

class preprocess:
    volumes_dir = Path("/preprocessed/volumes")
    masks_dir = Path("/preprocessed/masks")

    volume_suffix = "_volume.nii.gz"
    mask_suffix = "_mask.nii.gz"

class index_data:
    subject_id_placeholder = "{subject_id}"