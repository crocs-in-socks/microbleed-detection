import torch
import numpy as np
from torch.utils.data import Dataset

from microbleednet.core.transforms.augmentations import augment


class BasePatchDataset(Dataset):
    def __init__(
        self,
        patches: list,
        perform_augmentation: bool = False
    ):
        self.patches = patches
        self.perform_augmentation = perform_augmentation

    def __len__(self):
        return len(self.patches)

    def load_patch(self, idx: int):

        patch = self.patches[idx]
        patch_path = patch.get("patch_path")
        has_microbleed = patch.get("has_microbleed")
        is_augmented = patch.get("is_augmented")
        
        with np.load(patch_path) as patch_data:
            volume = patch_data.get("volume")
            mask = patch_data.get("mask")
            voxel_weights = patch_data.get("voxel_weights")

        return volume, mask, voxel_weights, has_microbleed, is_augmented

    def __getitem__(self, idx: int):
        raise NotImplementedError("Subclasses must implement the __getitem__ method.")
        

class SegmentationPatchDataset(BasePatchDataset):
    def __getitem__(self, idx: int):
        x, y, weights, _, is_augmented = self.load_patch(idx)

        if self.perform_augmentation and is_augmented:
            x, y, weights = augment(x, y, weights)

        x = np.expand_dims(x, axis=0) # Shape: (1, H, W, D)
        y_one_hot = np.stack((1 - y, y), axis=0) # Shape: (2, H, W, D)
        weights = np.expand_dims(weights, axis=0) # Shape: (1, H, W, D)

        return {
            "x": torch.from_numpy(x).float(),
            "y": torch.from_numpy(y_one_hot).float(),
            "weights": torch.from_numpy(weights).float()
        }

class SegmentationClassificationPatchDataset(BasePatchDataset):
    def __getitem__(self, idx: int):
        volume, mask, weights, label, is_augmented = self.load_patch(idx)

        if self.perform_augmentation and is_augmented:
            volume, mask, weights = augment(volume, mask, weights)

        volume = np.expand_dims(volume, axis=0)
        mask_one_hot = np.stack((1 - mask, mask), axis=0)
        weights = np.expand_dims(weights, axis=0)

        label_one_hot = np.array([1 - int(label), int(label)])

        return {
            "volume": torch.from_numpy(volume).float(),
            "mask": torch.from_numpy(mask_one_hot).float(),
            "weights": torch.from_numpy(weights).float(),
            "label": torch.from_numpy(label_one_hot).float()
        }