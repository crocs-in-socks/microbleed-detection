from pathlib import Path


class manifests:
    raw = Path("manifests/raw.json")
    preprocessed = Path("manifests/preprocessed.json")

class preprocess:
    volumes_dir = Path("preprocessed/volumes")
    masks_dir = Path("preprocessed/masks")

    volume_suffix = "_volume.nii.gz"
    mask_suffix = "_mask.nii.gz"

class index_data:
    subject_id_placeholder = "{subject_id}"

class train:
    detector_patches_dir = Path("train/detector/patches")
    detector_checkpoints_dir = Path("train/detector/checkpoints")

    discriminator_teacher_patches_dir = Path("train/discriminator_teacher/patches")
    discriminator_teacher_checkpoints_dir = Path("train/discriminator_teacher/checkpoints")

    discriminator_student_patches_dir = Path("train/discriminator_student/patches")
    discriminator_student_checkpoints_dir = Path("train/discriminator_student/checkpoints")

    class default:
        detector_threshold = 0.0