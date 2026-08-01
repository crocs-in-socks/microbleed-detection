import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import constants


class SingleConv(nn.Module):
    """
    convolution -> batch-normalization -> relu
    """

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, padding: int = constants.common.layers.single_conv.default.padding
    ):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class DoubleConv(nn.Module):
    """
    (convolution -> batch-normalization -> relu) twice
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size_1: int,
        kernel_size_2: int,
        intermediate_channels: int | None = None,
    ):
        super().__init__()

        if intermediate_channels is None:
            intermediate_channels = out_channels

        padding_1 = (kernel_size_1 - 1) // 2
        padding_2 = (kernel_size_2 - 1) // 2

        self.layer = nn.Sequential(
            SingleConv(in_channels, intermediate_channels, kernel_size_1, padding_1),
            SingleConv(intermediate_channels, out_channels, kernel_size_2, padding_2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class DownConv(nn.Module):
    """
    maxpool with kernel_size 2 -> (convolution -> batch_normalization -> relu) twice
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size_1: int,
        kernel_size_2: int,
    ):
        super().__init__()
        self.layer = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels, kernel_size_1, kernel_size_2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class UpConv(nn.Module):
    """
    convolution transpose (upsampling) -> concatenation with skip connection -> (convolution -> batch-normalization -> relu) twice
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.upsample = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size, stride=2)
        self.conv = DoubleConv(in_channels, out_channels, 3, 1)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:

        x1 = self.upsample(x1)

        diff_z = x2.size()[2] - x1.size()[2]
        diff_y = x2.size()[3] - x1.size()[3]
        diff_x = x2.size()[4] - x1.size()[4]

        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x1 = F.pad(
            x1,
            [
                diff_x // 2,
                diff_x - diff_x // 2,
                diff_y // 2,
                diff_y - diff_y // 2,
                diff_z // 2,
                diff_z - diff_z // 2,
            ],
        )

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """
    convolution with kernel_size=1
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layer = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)
