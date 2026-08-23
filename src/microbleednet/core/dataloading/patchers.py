from pathlib import Path

import numpy as np

from microbleednet.core.datamodels import ExtractedPatch, PatchRecord
from microbleednet.core.io import save_array_atomic
from microbleednet.core.transforms import patch

# No augmentation multiplier unless a caller asks for one.
DEFAULT_AUGMENTATION_FACTOR = 1


def nonoverlapping_patcher(
    volume: np.ndarray, mask: np.ndarray, patch_size: int
) -> list[ExtractedPatch]:
    volume_patches = patch.get_nonoverlapping_patches(volume, patch_size)
    mask_patches = patch.get_nonoverlapping_patches(mask, patch_size)

    return [
        ExtractedPatch(volume=volume_patch, mask=mask_patch)
        for volume_patch, mask_patch in zip(volume_patches, mask_patches)
    ]


def target_centered_patcher(
    volume: np.ndarray,
    mask: np.ndarray,
    centers: list[tuple[int, int, int]],
    patch_size: int,
) -> list[ExtractedPatch]:
    volume_patches = patch.extract_centered_patches(volume, centers, patch_size)
    mask_patches = patch.extract_centered_patches(mask, centers, patch_size)

    return [
        ExtractedPatch(volume=volume_patch, mask=mask_patch)
        for volume_patch, mask_patch in zip(volume_patches, mask_patches)
    ]


def materialize_patches(
    patches: list[ExtractedPatch],
    patch_dir: Path,
    volume_identifier: str,
    augmentation_factor: int = DEFAULT_AUGMENTATION_FACTOR,
) -> list[PatchRecord]:
    patch_dir.mkdir(parents=True, exist_ok=True)

    if not patches:
        return []

    # Stack the subject's fixed-shape patches into two (N, P, P, P) arrays and
    # write one .npy file each, rather than a tiny file per patch. The datasets
    # memory-map these and slice one patch by index, so random access stays O(1)
    # while the file count drops from 2*patches to 2 per subject. Uncompressed:
    # patches are written once and re-read every epoch, and float intensity data
    # compresses poorly, so we trade disk for no per-read decompression.
    volumes = np.stack([patch.volume for patch in patches])
    masks = np.stack([patch.mask for patch in patches])

    volume_path = (patch_dir / f"volumes_{volume_identifier}.npy").resolve()
    mask_path = (patch_dir / f"masks_{volume_identifier}.npy").resolve()
    save_array_atomic(volumes, volume_path)
    save_array_atomic(masks, mask_path)

    records: list[PatchRecord] = []
    for index, mask in enumerate(masks):
        record = PatchRecord(
            volume_path=str(volume_path),
            mask_path=str(mask_path),
            patch_index=index,
            has_microbleed=bool(np.any(mask > 0)),
        )
        # Inflate the training set by augmentation_factor without duplicating the
        # patch on disk: each patch is stored once, then its record is referenced
        # factor times. The dataset seeds augmentation on each record's position
        # in the final list, so the copies land at distinct positions and yield
        # distinct augmentations of the same stored patch. Validation passes
        # factor=1, so it is never inflated.
        records.extend(record for _ in range(augmentation_factor))

    return records
