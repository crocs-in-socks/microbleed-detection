from pathlib import Path

from microbleednet.core import utils
from microbleednet.core import preprocess

def main():
    input_path = Path("/home/gouri/workspace/ephemeral/swan_COG0432.nii.gz")
    label_path = Path("/home/gouri/workspace/ephemeral/mIP_swan_COG0432_allSlices_mask.nii.gz")

    volume = utils.load_volume(input_path)
    mask = utils.load_volume(label_path)

    output_volume, output_mask, bounding_box = preprocess.run(volume, mask, True, True, True, True, True)

    output_volume = utils.numpy_to_nifti(output_volume, volume)
    output_path = Path("/home/gouri/workspace/ephemeral/preprocessed_volume.nii.gz")
    utils.save_volume(output_volume, output_path)
    
    output_mask = utils.numpy_to_nifti(output_mask, mask)
    output_mask_path = Path("/home/gouri/workspace/ephemeral/preprocessed_mask.nii.gz")
    utils.save_volume(output_mask, output_mask_path)