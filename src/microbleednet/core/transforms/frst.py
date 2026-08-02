import math
from collections.abc import Sequence

import torch
import torchvision.transforms.functional as F

# Fast Radial Symmetry Transform defaults. Structural parameters of the
# transform; not exposed through the config models.
_DEFAULT_RADII = [2, 3, 4, 6]
_DEFAULT_ALPHA = 2
_DEFAULT_FACTOR_STD = 0.1
_DEFAULT_BRIGHT = True
_DEFAULT_DARK = False


def apply(
    volumes: torch.Tensor,
    radii: Sequence[float] = _DEFAULT_RADII,
    alpha: float = _DEFAULT_ALPHA,
    factor_std: float = _DEFAULT_FACTOR_STD,
    bright: bool = _DEFAULT_BRIGHT,
    dark: bool = _DEFAULT_DARK,
) -> torch.Tensor:
    """
    Batched 3D FRST on GPU.
    Input: volumes (Batch, 1, H, W, D)
    Output: frst_volumes (Batch, 1, H, W, D)
    """
    if volumes.ndim != 5 or volumes.shape[1] != 1:
        raise ValueError("volumes must have shape (B, 1, H, W, D)")
    if not volumes.is_floating_point():
        raise TypeError("volumes must use a floating-point dtype")
    if not radii or any(radius <= 0 for radius in radii):
        raise ValueError("radii must contain positive values")

    batch_size, channels, height, width, depth = volumes.shape

    slices = volumes.permute(0, 4, 2, 3, 1).reshape(-1, height, width)
    slice_count = slices.shape[0]

    grad_y, grad_x = torch.gradient(slices, dim=(1, 2))
    g_norm = torch.sqrt(grad_x**2 + grad_y**2)

    significant = g_norm > 0
    grad_x_significant = grad_x[significant]
    grad_y_significant = grad_y[significant]
    g_norm_significant = g_norm[significant]

    coords = torch.nonzero(significant)
    nn = coords[:, 0]
    yy = coords[:, 1]
    xx = coords[:, 2]

    offset = int(math.ceil(max(radii))) if len(radii) > 0 else 0
    out_height = height + 2 * offset
    out_width = width + 2 * offset

    output = torch.zeros(
        (slice_count, out_height, out_width), device=volumes.device, dtype=volumes.dtype
    )

    for radius in radii:
        orientation = torch.zeros(
            (slice_count, out_height, out_width),
            device=volumes.device,
            dtype=volumes.dtype,
        )
        magnitude = torch.zeros(
            (slice_count, out_height, out_width),
            device=volumes.device,
            dtype=volumes.dtype,
        )
        gp_y = torch.round((grad_y_significant / g_norm_significant) * radius).long()
        gp_x = torch.round((grad_x_significant / g_norm_significant) * radius).long()

        if bright:
            pos_y = yy + gp_y + offset
            pos_x = xx + gp_x + offset
            if (
                torch.any(pos_y < 0)
                or torch.any(pos_y >= out_height)
                or torch.any(pos_x < 0)
                or torch.any(pos_x >= out_width)
            ):
                raise IndexError("FRST scatter index is outside the padded output")

            # Flatten 3D indices to 1D for scatter_add_
            idx_bright = nn * (out_height * out_width) + pos_y * out_width + pos_x
            orientation.view(-1).scatter_add_(
                0, idx_bright, torch.ones_like(idx_bright, dtype=volumes.dtype)
            )
            magnitude.view(-1).scatter_add_(0, idx_bright, g_norm_significant)

        if dark:
            neg_y = yy - gp_y + offset
            neg_x = xx - gp_x + offset

            idx_dark = nn * (out_height * out_width) + neg_y * out_width + neg_x
            orientation.view(-1).scatter_add_(
                0, idx_dark, -torch.ones_like(idx_dark, dtype=volumes.dtype)
            )
            magnitude.view(-1).scatter_add_(0, idx_dark, -g_norm_significant)

        orientation = torch.abs(orientation)
        orientation = normalize_tensor_slicewise(orientation)

        magnitude = torch.abs(magnitude)
        magnitude = normalize_tensor_slicewise(magnitude)

        response = (orientation**alpha) * magnitude

        sigma = radius * factor_std
        if sigma > 0:
            # Replicate SciPy's default kernel size (truncate=4.0)
            rad_int = int(4.0 * sigma + 0.5)
            kernel_size = 2 * rad_int + 1

            response = response.unsqueeze(1)
            response = F.gaussian_blur(
                response, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma]
            )
            output += response.squeeze(1)
        else:
            output += response

    output = output / len(radii)

    output = output[:, offset:-offset, offset:-offset]

    # Reshape back to (Batch, 1, Height, Width, Depth)
    output = (
        output.view(batch_size, depth, height, width)
        .unsqueeze(1)
        .permute(0, 1, 3, 4, 2)
    )
    return output


def prepend_frst_channel(volume: torch.Tensor) -> torch.Tensor:
    """Return ``volume`` with its FRST transform appended as a second channel.

    Every model in the pipeline consumes two input channels: the volume and its
    FRST response, concatenated along the channel axis. This is the single place
    that pairing is expressed, so tasks, inference, and the processor all agree.

    Input/output shape: (Batch, 1, H, W, D) -> (Batch, 2, H, W, D).
    """
    return torch.cat((volume, apply(volume)), dim=1)


def normalize_tensor_slicewise(tensor: torch.Tensor) -> torch.Tensor:
    # Assumes tensor shape is (N, H, W)
    t_min = tensor.amin(dim=(1, 2), keepdim=True)
    t_max = tensor.amax(dim=(1, 2), keepdim=True)

    tensor = tensor - t_min
    t_range = t_max - t_min

    # Divide only where the range is > 0 to avoid division by zero
    tensor = torch.where(t_range > 0, tensor / t_range, tensor)

    return tensor
