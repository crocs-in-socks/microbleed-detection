import hashlib

import numpy as np
import torch
from torch.utils.data import Dataset

from microbleednet.config import AugmentationConfig
from microbleednet.core.transforms.augmentations import augment


class BasePatchDataset(Dataset):
    def __init__(
        self,
        patches: list,
        perform_augmentation: bool = False,
        augmentation: AugmentationConfig | None = None,
    ):
        self.patches = patches
        self.perform_augmentation = perform_augmentation
        self.augmentation = augmentation or AugmentationConfig()

    def __len__(self):
        return len(self.patches)

    def load_patch(self, idx: int):

        patch = self.patches[idx]
        patch_path = patch["patch_path"]
        with np.load(patch_path) as patch_data:
            patch_dict = {key: patch_data[key] for key in patch_data.files}

        expected_checksum = patch.get("checksum")
        if expected_checksum:
            with open(patch_path, "rb") as patch_file:
                actual_checksum = hashlib.sha256(patch_file.read()).hexdigest()
            if actual_checksum != expected_checksum:
                raise ValueError(f"corrupt patch file: {patch_path}")

        patch_dict["has_microbleed"] = patch["has_microbleed"]
        patch_dict["is_augmented"] = patch.get("augmentation_version", 0) != 0

        return patch_dict

    def __getitem__(self, idx: int):
        raise NotImplementedError("Subclasses must implement the __getitem__ method.")


class SegmentationPatchDataset(BasePatchDataset):
    def __getitem__(self, idx: int):
        patch = self.load_patch(idx)

        volume = patch["volume"]
        mask = patch["mask"]
        if self.perform_augmentation:
            volume, mask = augment(
                volume,
                mask,
                augmentation=self.augmentation,
                mask_indices=(1,),
                rng=np.random.default_rng(idx),
            )

        volume = np.expand_dims(volume, axis=0)  # Shape: (1, H, W, D)

        return {
            "volume": torch.from_numpy(volume).float(),
            "mask": torch.from_numpy(mask.astype(np.int64, copy=False)),
        }


class SegmentationClassificationPatchDataset(BasePatchDataset):
    def __getitem__(self, idx: int):
        patch = self.load_patch(idx)

        volume = patch["volume"]
        mask = patch["mask"]
        label = patch["has_microbleed"]
        if self.perform_augmentation:
            volume, mask = augment(
                volume,
                mask,
                augmentation=self.augmentation,
                mask_indices=(1,),
                rng=np.random.default_rng(idx),
            )

        volume = np.expand_dims(volume, axis=0)  # Shape: (1, H, W, D)
        return {
            "volume": torch.from_numpy(volume).float(),
            "mask": torch.from_numpy(mask.astype(np.int64, copy=False)),
            "label": torch.tensor(int(label), dtype=torch.int64),
        }


class ClassificationPatchDataset(BasePatchDataset):
    def __getitem__(self, idx):
        patch = self.load_patch(idx)

        volume = patch["volume"]
        label = patch["has_microbleed"]
        if self.perform_augmentation:
            (volume,) = augment(volume, augmentation=self.augmentation)  # Unpack

        volume = np.expand_dims(volume, axis=0)  # Shape: (1, H, W, D)
        return {
            "volume": torch.from_numpy(volume).float(),
            "label": torch.tensor(int(label), dtype=torch.int64),
        }
