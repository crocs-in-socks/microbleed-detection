import torch.nn as nn

from . import layers


class CandidateDetector(nn.Module):
    def __init__(self, in_channels: int, n_classes: int, initial_channels: int):
        super().__init__()

        level_channels = [
            3,
            initial_channels,
            initial_channels * 2,
            initial_channels * 4,
        ]

        self.feature_extractor = FeatureExtractor(in_channels, level_channels)
        self.segmentor = Segmentor(level_channels, n_classes)

    def forward(self, x):
        features = self.feature_extractor(x)
        logits = self.segmentor(features)
        return logits
    

class FeatureExtractor(nn.Module):
    def __init__(self, in_channels: int, level_channels: list[int]):
        super().__init__()

        self.in_conv = layers.OutConv(in_channels, level_channels[0])
        self.conv_1 = layers.DoubleConv(level_channels[0], level_channels[1], 3, 1)
        self.down_1 = layers.DownConv(level_channels[1], level_channels[2], 3, 1)
        self.down_2 = layers.DownConv(level_channels[2], level_channels[3], 3, 1)

    def forward(self, x):
        x0 = self.in_conv(x)
        x1 = self.conv_1(x0)
        x2 = self.down_1(x1)
        x3 = self.down_2(x2)

        return {"x1": x1, "x2": x2, "x3": x3}


class Segmentor(nn.Module):
    def __init__(self, level_channels: list[int], n_classes: int):
        super().__init__()

        self.up_2 = layers.UpConv(level_channels[3], level_channels[2], 3)
        self.up_1 = layers.UpConv(level_channels[2], level_channels[1], 3)
        self.out_conv = layers.OutConv(level_channels[1], n_classes)

    def forward(self, features):
        x1 = features.get("x1")
        x2 = features.get("x2")
        x3 = features.get("x3")

        x = self.up_2(x3, x2)
        x = self.up_1(x, x1)
        logits = self.out_conv(x)

        return logits