import random
import numpy as np
from skimage.util import random_noise
from scipy.ndimage import gaussian_filter

from .. import constants


def translate_array(array, offsetx: int, offsety: int):
    """Shifts array by integer offsets and pads with zeros."""
    shifted_array = np.roll(array, shift=offsetx, axis=0)
    shifted_array = np.roll(shifted_array, shift=offsety, axis=1) 

    # Zero out the regions that rolled around
    if offsety > 0:
        shifted_array[:offsety, :, :] = 0  
    elif offsety < 0:
        shifted_array[offsety:, :, :] = 0  

    if offsetx > 0:
        shifted_array[:, :offsetx, :] = 0  
    elif offsetx < 0:
        shifted_array[:, offsetx:, :] = 0  

    return shifted_array

def translate(*volumes, **kwargs):
    """
    Translation: x-offset: [-15, 15], y-offset: [-15, 15] voxels
    Applied to ALL provided volumes equally.
    (kwargs swallows 'intensity_indices' passed by the main loop)
    """
    offsetx = random.randint(*constants.transforms.augmentation.translation_offset_range)
    offsety = random.randint(*constants.transforms.augmentation.translation_offset_range)

    translated_volumes = tuple(translate_array(vol, offsetx, offsety) for vol in volumes)
    
    return translated_volumes

def add_noise(*volumes, intensity_indices=(0,)):
    """
    Random noise injection: Distribution - Gaussian, mu = 0, sigma^2 = [0.01, 0.04]
    Applied ONLY to the volumes specified by intensity_indices.
    """
    variance = random.uniform(*constants.transforms.augmentation.noise_variance_range)
    
    result = list(volumes)
    
    for idx in intensity_indices:
        result[idx] = random_noise(result[idx], mode='gaussian', mean=0, var=variance)
        
    return tuple(result)

def blur(*volumes, intensity_indices=(0,)):
    """
    Gaussian filtering: sigma = [0.1, 0.2] voxels
    Applied ONLY to the volumes specified by intensity_indices.
    """
    sigma = random.uniform(*constants.transforms.augmentation.blur_sigma_range)
    
    result = list(volumes)
    
    for idx in intensity_indices:
        result[idx] = gaussian_filter(result[idx], sigma)
        
    return tuple(result)

def augment(*volumes, intensity_indices=(0,)):
    """
    Applies a random combination of transformations to an arbitrary number of volumes.
    
    Args:
        *volumes: Any number of numpy arrays (e.g., image, label, weights)
        intensity_indices: Tuple of integers indicating which volumes get blur/noise. Defaults to (0,), meaning only the first volume is altered.
    """
    available_transformations = {
        'translate': translate, 
        'noise': add_noise, 
        'blur': blur
    }
    
    num_transformations_to_apply = random.randint(1, len(available_transformations))
    transformations = random.sample(list(available_transformations.values()), num_transformations_to_apply)
    
    transformed_volumes = volumes

    for func in transformations:
        transformed_volumes = func(*transformed_volumes, intensity_indices=intensity_indices)

    return transformed_volumes