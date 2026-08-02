"""On-disk dataset layout shared by the pipeline commands.

These are the relative paths and filename suffixes that define how an indexed
dataset directory is structured: where ``index-data`` writes the raw manifest,
where ``preprocess`` reads it and writes preprocessed volumes/masks, and where
``train`` looks for the preprocessed manifest. They belong to no config model —
they are a filesystem contract between commands, not user-tunable values — so
they live here rather than on a datamodel.
"""

from pathlib import Path


class manifests:
    raw = Path("manifests/raw.json")
    preprocessed = Path("manifests/preprocessed.json")


class preprocess:
    volumes_dir = Path("preprocessed/volumes")
    masks_dir = Path("preprocessed/masks")

    volume_suffix = "_volume.nii.gz"
    mask_suffix = "_mask.nii.gz"
