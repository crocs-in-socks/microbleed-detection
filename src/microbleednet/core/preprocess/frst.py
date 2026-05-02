import numpy as np
import scipy.ndimage as ndimage

from microbleednet.core.utils import *

from microbleednet.constants import FRST_RADII
from microbleednet.constants import FRST_ALPHA
from microbleednet.constants import FRST_FACTOR_STD
from microbleednet.constants import FRST_BRIGHT
from microbleednet.constants import FRST_DARK


def apply(volume: np.ndarray) -> np.ndarray:
    """
    Apply Fast Radial Symmetry Transform (FRST) to the given volume.
    """

    frst_volume = np.zeros_like(volume)
    _, _, depth = volume.shape
        
    for slice_idx in range(depth):
        frst_volume[:, :, slice_idx] = fast_radial_symmetry_transform(
            volume[:, :, slice_idx],
            radii=FRST_RADII,
            alpha=FRST_ALPHA,
            factor_std=FRST_FACTOR_STD,
            bright=FRST_BRIGHT,
            dark=FRST_DARK,
        )

    return frst_volume


def fast_radial_symmetry_transform(
    slice: np.ndarray,
    radii: list,
    alpha: float,
    factor_std: float,
    bright: bool,
    dark: bool,
) -> np.ndarray:

    dy, dx = np.gradient(slice)
    g_norm = np.sqrt(dx**2 + dy**2)

    significant = g_norm > 0
    dx_significant = dx[significant]
    dy_significant = dy[significant]
    g_norm_significant = g_norm[significant]

    yy, xx = np.nonzero(significant)

    offset = int(np.ceil(np.max(radii)))
    output_shape = (slice.shape[0] + 2 * offset, slice.shape[1] + 2 * offset)

    output = np.zeros(output_shape)

    for radius in radii:
        O_n = np.zeros(output_shape)
        M_n = np.zeros(output_shape)
        gp_y = np.round((dy_significant / g_norm_significant) * radius).astype(int)
        gp_x = np.round((dx_significant / g_norm_significant) * radius).astype(int)

        if bright:
            pos_y = yy + gp_y + offset
            pos_x = xx + gp_x + offset
            np.add.at(O_n, (pos_y, pos_x), 1)
            np.add.at(M_n, (pos_y, pos_x), g_norm_significant)

        if dark:
            neg_y = yy - gp_y + offset
            neg_x = xx - gp_x + offset
            np.add.at(O_n, (neg_y, neg_x), -1)
            np.add.at(M_n, (neg_y, neg_x), -g_norm_significant)

        O_n = np.abs(O_n)
        O_n = normalize_volume(O_n)

        M_n = np.abs(M_n)
        M_n = normalize_volume(M_n)

        S_n = (O_n**alpha) * M_n
        output += ndimage.gaussian_filter(S_n, radius * factor_std)

    return output[offset:-offset, offset:-offset]