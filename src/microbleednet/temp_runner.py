from pathlib import Path

import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.ndimage import gaussian_filter
from sklearn.model_selection import train_test_split

from microbleednet.core import utils
from microbleednet.core.engines import processor
from microbleednet.core.engines.trainers import Trainer
from microbleednet.core.common.tasks import SegmentationTask, SegmentationClassificationTask, KnowledgeDistillationClassificationTask
from microbleednet.core.common.models import CandidateDetector, CandidateDiscriminatorTeacher, CandidateDiscriminatorStudent
from microbleednet.core.dataloading import patchers
from microbleednet.core.dataloading.samplers import EqualBatchSampler
from microbleednet.core.dataloading.datasets import SegmentationPatchDataset, SegmentationClassificationPatchDataset, ClassificationPatchDataset


def main():
    inputs = [
        Path("/home/gouri/workspace/ephemeral/samples/preprocessed/volumes/volume_1.nii.gz"),
        Path("/home/gouri/workspace/ephemeral/samples/preprocessed/volumes/volume_0.nii.gz"),
    ]
    labels = [
        Path("/home/gouri/workspace/ephemeral/samples/preprocessed/masks/mask_1.nii.gz"),
        Path("/home/gouri/workspace/ephemeral/samples/preprocessed/masks/mask_0.nii.gz"),
    ]

    subjects = list(zip(inputs, labels))

    # constants
    device = torch.device("cuda")
    pin_memory = True
    train_proportion = 0.8
    random_state = 42
    num_workers = 4
    batch_size = 64
    patch_size = 24
    augmentation_factor = 5
    optimizer_parameters = {
        "lr": 1e-3,
        "eps": 1e-4,
    }
    scheduler_parameters = {
        "milestones": [2, 4, 6],
        "gamma": 0.1,
    }
    detector_threshold = 0
    detector_checkpoint_path = Path("/home/gouri/workspace/ephemeral/samples/checkpoints/detector/best_model.pth")
    discriminator_teacher_checkpoint_path = Path("/home/gouri/workspace/ephemeral/samples/checkpoints/discriminator_teacher/best_model.pth")
    checkpoint_dir = Path("/home/gouri/workspace/ephemeral/samples/checkpoints/discriminator_student")
    train_patch_path = Path("/home/gouri/workspace/ephemeral/samples/patches/train")
    test_patch_path = Path("/home/gouri/workspace/ephemeral/samples/patches/test")

    train_subjects, test_subjects = train_test_split(subjects, train_size=train_proportion, random_state=random_state)

    detector_model = CandidateDetector(2, 2, 64).to(device)
    utils.load_model_weights(detector_model, device, detector_checkpoint_path)

    train_patches = []
    for idx, subject in enumerate(train_subjects):
        volume_path, mask_path = subject

        volume = utils.nifti_to_numpy(utils.load_volume(volume_path))
        mask = utils.nifti_to_numpy(utils.load_volume(mask_path))

        detector_logits = processor.infer(detector_model, device, volume)
        detector_output = F.softmax(detector_logits, dim=1)
        detector_output = detector_output.cpu().numpy()[0, 1]
        detector_output = (detector_output > detector_threshold).astype(int)

        subject_patches = patchers.target_centered_patcher(volume, mask, detector_output, patch_size)
        subject_patches = patchers.materialize_patches(subject_patches, train_patch_path, str(idx), augmentation_factor)
        train_patches.extend(subject_patches)

    test_patches = []
    for idx, subject in enumerate(test_subjects):
        volume_path, mask_path = subject

        volume = utils.nifti_to_numpy(utils.load_volume(volume_path))
        mask = utils.nifti_to_numpy(utils.load_volume(mask_path))

        detector_logits = processor.infer(detector_model, device, volume)
        detector_output = F.softmax(detector_logits, dim=1)
        detector_output = detector_output.cpu().numpy()[0, 1]
        detector_output = (detector_output > detector_threshold).astype(int)

        subject_patches = patchers.target_centered_patcher(volume, mask, detector_output, patch_size)
        subject_patches = patchers.materialize_patches(subject_patches, test_patch_path, str(idx), augmentation_factor)
        test_patches.extend(subject_patches)

    # Free detector model - no longer needed
    del detector_model
    torch.cuda.empty_cache()

    train_set = ClassificationPatchDataset(train_patches, perform_augmentation=True)
    train_sampler = EqualBatchSampler(train_patches, batch_size=batch_size)
    train_loader = DataLoader(
        dataset=train_set,
        # batch_sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_set = ClassificationPatchDataset(test_patches)
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    teacher_model = CandidateDiscriminatorTeacher(2, 2, 64, 0.2).to(device)
    utils.load_model_weights(teacher_model, device, discriminator_teacher_checkpoint_path)

    student_model = CandidateDiscriminatorStudent(2, 2, 64, 0.2).to(device)
    
    task = KnowledgeDistillationClassificationTask(teacher_model)
    trainer = Trainer(student_model, device, optimizer_parameters, scheduler_parameters, task, checkpoint_dir)

    trainer.fit(train_loader, test_loader, n_epochs=10)


def temp_train_discriminator_teacher():
    inputs = [
        Path("/home/gouri/workspace/ephemeral/samples/preprocessed/volumes/volume_0.nii.gz"),
        Path("/home/gouri/workspace/ephemeral/samples/preprocessed/volumes/volume_1.nii.gz")
    ]
    labels = [
        Path("/home/gouri/workspace/ephemeral/samples/preprocessed/masks/mask_0.nii.gz"),
        Path("/home/gouri/workspace/ephemeral/samples/preprocessed/masks/mask_1.nii.gz")
    ]

    subjects = list(zip(inputs, labels))

    # constants
    device = torch.device("cuda")
    pin_memory = True
    train_proportion = 0.8
    random_state = 42
    num_workers = 4
    batch_size = 32
    patch_size = 24
    augmentation_factor = 10
    optimizer_parameters = {
        "lr": 1e-3,
        "eps": 1e-4,
    }
    scheduler_parameters = {
        "milestones": [2, 4, 6],
        "gamma": 0.1,
    }
    detector_checkpoint_path = Path("/home/gouri/workspace/ephemeral/samples/checkpoints/detector/best_model.pth")
    checkpoint_dir = Path("/home/gouri/workspace/ephemeral/samples/checkpoints/discriminator_teacher")
    train_patch_path = Path("/home/gouri/workspace/ephemeral/samples/patches/train")
    test_patch_path = Path("/home/gouri/workspace/ephemeral/samples/patches/test")

    train_subjects, test_subjects = train_test_split(subjects, train_size=train_proportion, random_state=random_state)

    train_patches = []
    for idx, subject in enumerate(train_subjects):
        volume_path, mask_path = subject

        volume = utils.nifti_to_numpy(utils.load_volume(volume_path))
        mask = utils.nifti_to_numpy(utils.load_volume(mask_path))

        subject_patches = patchers.nonoverlapping_patcher(volume, mask, patch_size)
        subject_patches = patchers.materialize_patches(subject_patches, train_patch_path, str(idx), augmentation_factor)
        train_patches.extend(subject_patches)

    test_patches = []
    for idx, subject in enumerate(test_subjects):
        volume_path, mask_path = subject

        volume = utils.nifti_to_numpy(utils.load_volume(volume_path))
        mask = utils.nifti_to_numpy(utils.load_volume(mask_path))

        subject_patches = patchers.nonoverlapping_patcher(volume, mask, patch_size)
        subject_patches = patchers.materialize_patches(subject_patches, test_patch_path, str(idx), 1)
        test_patches.extend(subject_patches)

    train_set = SegmentationClassificationPatchDataset(train_patches, perform_augmentation=True)
    train_sampler = EqualBatchSampler(train_patches, batch_size=batch_size)
    train_loader = DataLoader(
        dataset=train_set,
        batch_sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_set = SegmentationClassificationPatchDataset(test_patches)
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    model = CandidateDiscriminatorTeacher(2, 2, 64, 0.2).to(device)
    utils.load_model_weights(model, device, detector_checkpoint_path)
    
    task = SegmentationClassificationTask()
    trainer = Trainer(model, device, optimizer_parameters, scheduler_parameters, task, checkpoint_dir)

    trainer.fit(train_loader, test_loader, n_epochs=10)


def temp_train_detector():
    inputs = [
        Path("/home/gouri/workspace/ephemeral/samples/preprocessed/volumes/volume_0.nii.gz"),
        Path("/home/gouri/workspace/ephemeral/samples/preprocessed/volumes/volume_1.nii.gz")
    ]
    labels = [
        Path("/home/gouri/workspace/ephemeral/samples/preprocessed/masks/mask_0.nii.gz"),
        Path("/home/gouri/workspace/ephemeral/samples/preprocessed/masks/mask_1.nii.gz")
    ]

    subjects = list(zip(inputs, labels))

    # constants
    device = torch.device("cuda")
    pin_memory = True
    train_proportion = 0.8
    random_state = 42
    num_workers = 4
    batch_size = 16
    patch_size = 48
    augmentation_factor = 10
    optimizer_parameters = {
        "lr": 1e-3,
        "eps": 1e-4,
    }
    scheduler_parameters = {
        "milestones": [2, 4, 6],
        "gamma": 0.1,
    }
    checkpoint_dir = Path("/home/gouri/workspace/ephemeral/samples/checkpoints/detector")
    train_patch_path = Path("/home/gouri/workspace/ephemeral/samples/patches/train")
    test_patch_path = Path("/home/gouri/workspace/ephemeral/samples/patches/test")

    train_subjects, test_subjects = train_test_split(subjects, train_size=train_proportion, random_state=random_state)

    train_patches = []
    for idx, subject in enumerate(train_subjects):
        volume_path, mask_path = subject

        volume = utils.nifti_to_numpy(utils.load_volume(volume_path))
        mask = utils.nifti_to_numpy(utils.load_volume(mask_path))

        subject_patches = patchers.nonoverlapping_patcher(volume, mask, patch_size)
        subject_patches = patchers.materialize_patches(subject_patches, train_patch_path, str(idx), augmentation_factor)
        train_patches.extend(subject_patches)

    test_patches = []
    for idx, subject in enumerate(test_subjects):
        volume_path, mask_path = subject

        volume = utils.nifti_to_numpy(utils.load_volume(volume_path))
        mask = utils.nifti_to_numpy(utils.load_volume(mask_path))

        subject_patches = patchers.nonoverlapping_patcher(volume, mask, patch_size)
        subject_patches = patchers.materialize_patches(subject_patches, train_patch_path, str(idx), 1)
        test_patches.extend(subject_patches)

    train_set = SegmentationPatchDataset(train_patches, perform_augmentation=True)
    train_sampler = EqualBatchSampler(train_patches, batch_size=batch_size)
    train_loader = DataLoader(
        dataset=train_set,
        batch_sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_set = SegmentationPatchDataset(test_patches)
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    model = CandidateDetector(2, 2, 64).to(device)
    task = SegmentationTask()
    trainer = Trainer(model, device, optimizer_parameters, scheduler_parameters, task, checkpoint_dir)

    trainer.fit(train_loader, test_loader, n_epochs=10)


def temp_preprocess():
    inputs = [
        Path("/home/gouri/workspace/ephemeral/samples/raw/volumes/swan_COG0432.nii.gz"),
        Path("/home/gouri/workspace/ephemeral/samples/raw/volumes/swan_COG0894.nii.gz")
    ]
    labels = [
        Path("/home/gouri/workspace/ephemeral/samples/raw/masks/mIP_swan_COG0432_allSlices_mask.nii.gz"),
        Path("/home/gouri/workspace/ephemeral/samples/raw/masks/mIP_swan_COG0894_allSlices_mask.nii.gz")
    ]

    raw_subjects = zip(inputs, labels)
    for idx, raw_subject in enumerate(raw_subjects):
        input_volume_path, label_mask_path = raw_subject
        volume = utils.load_volume(input_volume_path)
        mask = utils.load_volume(label_mask_path)

        output_volume, output_mask, bounding_box = processor.run(volume, mask, True, True, True, True, True)

        output_volume = utils.numpy_to_nifti(output_volume, volume)
        output_volume_path = Path(f"/home/gouri/workspace/ephemeral/samples/preprocessed/volumes/volume_{idx}.nii.gz")
        utils.save_volume(output_volume, output_volume_path)

        output_mask = utils.numpy_to_nifti(output_mask, mask)
        output_mask_path = Path(f"/home/gouri/workspace/ephemeral/samples/preprocessed/masks/mask_{idx}.nii.gz")
        utils.save_volume(output_mask, output_mask_path)