import math
import torch
import torchvision.transforms.functional as F

from .. import constants


def apply(
    volumes: torch.Tensor, 
    radii: list = constants.transforms.frst.radii, 
    alpha: float = constants.transforms.frst.alpha, 
    factor_std: float = constants.transforms.frst.factor_std, 
    bright: bool = constants.transforms.frst.bright, 
    dark: bool = constants.transforms.frst.dark
) -> torch.Tensor:
    """
    Batched 3D FRST on GPU.
    Input: volumes (Batch, 1, H, W, D)
    Output: frst_volumes (Batch, 1, H, W, D)
    """
    B, C, H, W, D = volumes.shape

    slices = volumes.permute(0, 4, 2, 3, 1).reshape(-1, H, W) # New Shape: (B*D, H, W)
    N = slices.shape[0] # Number of slices

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
    out_H = H + 2 * offset
    out_W = W + 2 * offset

    output = torch.zeros((N, out_H, out_W), device=volumes.device, dtype=volumes.dtype)

    for radius in radii:
        O_n = torch.zeros((N, out_H, out_W), device=volumes.device, dtype=volumes.dtype)
        M_n = torch.zeros((N, out_H, out_W), device=volumes.device, dtype=volumes.dtype)
        gp_y = torch.round((grad_y_significant / g_norm_significant) * radius).long()
        gp_x = torch.round((grad_x_significant / g_norm_significant) * radius).long()

        if bright:
            pos_y = yy + gp_y + offset
            pos_x = xx + gp_x + offset
            
            # Flatten 3D indices to 1D for scatter_add_
            idx_bright = nn * (out_H * out_W) + pos_y * out_W + pos_x
            O_n.view(-1).scatter_add_(0, idx_bright, torch.ones_like(idx_bright, dtype=volumes.dtype))
            M_n.view(-1).scatter_add_(0, idx_bright, g_norm_significant)

        if dark:
            neg_y = yy - gp_y + offset
            neg_x = xx - gp_x + offset
            
            idx_dark = nn * (out_H * out_W) + neg_y * out_W + neg_x
            O_n.view(-1).scatter_add_(0, idx_dark, -torch.ones_like(idx_dark, dtype=volumes.dtype))
            M_n.view(-1).scatter_add_(0, idx_dark, -g_norm_significant)

        O_n = torch.abs(O_n)
        O_n = normalize_tensor_slicewise(O_n)

        M_n = torch.abs(M_n)
        M_n = normalize_tensor_slicewise(M_n)

        S_n = (O_n ** alpha) * M_n

        sigma = radius * factor_std
        if sigma > 0:
            # Replicate SciPy's default kernel size (truncate=4.0)
            rad_int = int(4.0 * sigma + 0.5)
            kernel_size = 2 * rad_int + 1
            
            S_n = S_n.unsqueeze(1) # F.gaussian_blur expects (..., C, H, W). We add and remove a dummy channel dimension. 
            S_n = F.gaussian_blur(S_n, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])
            output += S_n.squeeze(1)
        else:
            output += S_n

    if len(radii) > 0:
        output = output / len(radii)

    if offset > 0:
        output = output[:, offset:-offset, offset:-offset]

    # Reshape back to (Batch, 1, Height, Width, Depth)
    output = output.view(B, D, H, W).unsqueeze(1).permute(0, 1, 3, 4, 2)
    return output


def normalize_tensor_slicewise(tensor: torch.Tensor) -> torch.Tensor:
    # Assumes tensor shape is (N, H, W)
    t_min = tensor.amin(dim=(1, 2), keepdim=True)
    t_max = tensor.amax(dim=(1, 2), keepdim=True)
    
    tensor = tensor - t_min
    t_range = t_max - t_min
    
    # Divide only where the range is > 0 to avoid division by zero
    tensor = torch.where(t_range > 0, tensor / t_range, tensor)
    
    return tensor