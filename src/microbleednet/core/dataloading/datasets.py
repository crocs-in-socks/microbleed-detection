from typing import NamedTuple

import numpy as np
import torch
from torch.utils.data import Dataset

from microbleednet.core.datamodels import (
    AugmentationConfig,
    LoadedPatch,
    PatchRecord,
)
from microbleednet.core.io import load_array_mmap
from microbleednet.core.transforms.augmentations import augment

# The collated batches these datasets produce and the tasks consume. Each is a
# NamedTuple so PyTorch's default collate stacks it field-by-field into the same
# type (no custom ``collate_fn``), and a task reads its fields by attribute
# rather than by string key. Every batch carries ``volume`` (the input image
# patch); segmentation adds ``mask`` and classification adds ``label``.


class SegmentationBatch(NamedTuple):
    volume: torch.Tensor
    mask: torch.Tensor


class SegmentationClassificationBatch(NamedTuple):
    volume: torch.Tensor
    mask: torch.Tensor
    label: torch.Tensor


class ClassificationBatch(NamedTuple):
    volume: torch.Tensor
    label: torch.Tensor


class BasePatchDataset(Dataset):
    def __init__(
        self,
        patches: list[PatchRecord],
        perform_augmentation: bool = False,
        augmentation: AugmentationConfig | None = None,
    ):
        self.patches = patches
        self.perform_augmentation = perform_augmentation
        self.augmentation = augmentation or AugmentationConfig()
        # Memory-mapped per-subject arrays, opened lazily and cached by path.
        # DataLoader workers fork after construction, so each worker builds its
        # own cache and holds its own mmaps -- no handles are shared across the
        # fork.
        self._mmaps: dict[str, np.ndarray] = {}

    def __len__(self):
        return len(self.patches)

    def _mmap(self, path: str) -> np.ndarray:
        array = self._mmaps.get(path)
        if array is None:
            array = load_array_mmap(path)
            self._mmaps[path] = array
        return array

    def load_patch(self, idx: int) -> LoadedPatch:
        record = self.patches[idx]
        # Slice one patch out of the memory-mapped stacks. Copy so the returned
        # arrays own their data (the augmentation step writes in place) and no
        # mmap page is pinned beyond this call.
        volume = np.array(self._mmap(record.volume_path)[record.patch_index])
        mask = np.array(self._mmap(record.mask_path)[record.patch_index])

        return LoadedPatch(
            volume=volume, mask=mask, has_microbleed=record.has_microbleed
        )

    def __getitem__(self, idx: int):
        raise NotImplementedError("Subclasses must implement the __getitem__ method.")


class SegmentationPatchDataset(BasePatchDataset):
    def __getitem__(self, idx: int):
        patch = self.load_patch(idx)

        volume = patch.volume
        mask = patch.mask
        if self.perform_augmentation:
            volume, mask = augment(
                volume,
                mask,
                augmentation=self.augmentation,
                mask_indices=(1,),
                rng=np.random.default_rng(idx),
            )

        volume = np.expand_dims(volume, axis=0)  # Shape: (1, H, W, D)

        return SegmentationBatch(
            volume=torch.from_numpy(volume).float(),
            mask=torch.from_numpy(mask.astype(np.int64, copy=False)),
        )


class SegmentationClassificationPatchDataset(BasePatchDataset):
    def __getitem__(self, idx: int):
        patch = self.load_patch(idx)

        volume = patch.volume
        mask = patch.mask
        label = patch.has_microbleed
        if self.perform_augmentation:
            volume, mask = augment(
                volume,
                mask,
                augmentation=self.augmentation,
                mask_indices=(1,),
                rng=np.random.default_rng(idx),
            )

        volume = np.expand_dims(volume, axis=0)  # Shape: (1, H, W, D)
        return SegmentationClassificationBatch(
            volume=torch.from_numpy(volume).float(),
            mask=torch.from_numpy(mask.astype(np.int64, copy=False)),
            label=torch.tensor(int(label), dtype=torch.int64),
        )


class ClassificationPatchDataset(BasePatchDataset):
    def __getitem__(self, idx):
        patch = self.load_patch(idx)

        volume = patch.volume
        label = patch.has_microbleed
        if self.perform_augmentation:
            (volume,) = augment(volume, augmentation=self.augmentation)  # Unpack

        volume = np.expand_dims(volume, axis=0)  # Shape: (1, H, W, D)
        return ClassificationBatch(
            volume=torch.from_numpy(volume).float(),
            label=torch.tensor(int(label), dtype=torch.int64),
        )
