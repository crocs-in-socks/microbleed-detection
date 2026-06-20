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
        patch_path = patch["patch_path"]
        has_microbleed = patch["has_microbleed"]
        is_augmented = patch["is_augmented"]
        
        with np.load(patch_path) as patch_data:
            patch_dict = {key: patch_data[key] for key in patch_data.files}

        patch_dict["has_microbleed"] = has_microbleed
        patch_dict["is_augmented"] = is_augmented

        return patch_dict

    def __getitem__(self, idx: int):
        raise NotImplementedError("Subclasses must implement the __getitem__ method.")
        

class SegmentationPatchDataset(BasePatchDataset):
    def __getitem__(self, idx: int):
        patch = self.load_patch(idx)

        x = patch["volume"]
        y = patch["mask"]
        weights = patch["voxel_weights"]
        is_augmented = patch["is_augmented"]

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
        patch = self.load_patch(idx)

        volume = patch["volume"]
        mask = patch["mask"]
        weights = patch["voxel_weights"]
        label = patch["has_microbleed"]
        is_augmented = patch["is_augmented"]

        if self.perform_augmentation and is_augmented:
            volume, mask, weights = augment(volume, mask, weights)

        volume = np.expand_dims(volume, axis=0) # Shape: (1, H, W, D)
        mask_one_hot = np.stack((1 - mask, mask), axis=0) # Shape: (2, H, W, D)
        weights = np.expand_dims(weights, axis=0) # Shape: (1, H, W, D)

        label_one_hot = np.array([1 - int(label), int(label)])

        return {
            "volume": torch.from_numpy(volume).float(),
            "mask": torch.from_numpy(mask_one_hot).float(),
            "weights": torch.from_numpy(weights).float(),
            "label": torch.from_numpy(label_one_hot).float()
        }

class ClassificationPatchDataset(BasePatchDataset):
    def __getitem__(self, idx):
        patch = self.load_patch(idx)

        x = patch["volume"]
        y = patch["has_microbleed"]
        is_augmented = patch["is_augmented"]

        if self.perform_augmentation and is_augmented:
            (x,) = augment(x)  # Unpack the tuple returned by augment

        x = np.expand_dims(x, axis=0) # Shape: (1, H, W, D)
        y_one_hot = np.array([1 - int(y), int(y)]) 

        return {
            "x": torch.from_numpy(x).float(),
            "y": torch.from_numpy(y_one_hot).float()
        }