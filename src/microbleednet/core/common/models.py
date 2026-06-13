import torch
import torch.nn as nn

from . import layers

def weight_init(model):
    """Applies truncated normal initialization."""
    if isinstance(model, (nn.Conv3d, nn.Linear)):
        # PyTorch has a built-in truncated normal initializer
        nn.init.trunc_normal_(model.weight, std=0.05)
        if model.bias is not None:
            nn.init.constant_(model.bias, 0.1)


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

        self.apply(weight_init)

    def forward(self, x):
        features = self.feature_extractor(x)
        logits = self.segmentor(features)
        return logits
    

class CandidateDiscriminatorTeacher(nn.Module):
    def __init__(
        self,
        in_channels: int,
        n_classes: int,
        initial_channels: int,
        dropout_rate: float,
    ):
        super().__init__()

        level_channels = [
            3,
            initial_channels,
            initial_channels * 2,
            initial_channels * 4,
        ]

        self.feature_extractor = FeatureExtractor(in_channels, level_channels)
        self.segmentor = Segmentor(level_channels, n_classes)
        self.classifier = Classifier(level_channels[3], n_classes, dropout_rate)

        self.apply(weight_init)

    def forward(self, x):
        features = self.feature_extractor(x)
        segmentation_logits = self.segmentor(features)
        classification_logits = self.classifier(features)
        return segmentation_logits, classification_logits

class CandidateDiscriminatorStudent(nn.Module):
    def __init__(
        self,
        in_channels: int,
        n_classes: int,
        initial_channels: int,
        dropout_rate: float,
    ):
        super().__init__()

        level_channels = [
            3,
            initial_channels,
            initial_channels * 2,
            initial_channels * 4,
        ]

        self.feature_extractor = FeatureExtractor(in_channels, level_channels)
        self.classifier = Classifier(level_channels[3], n_classes, dropout_rate)

        self.apply(weight_init)

    def forward(self, x):
        features = self.feature_extractor(x)
        logits = self.classifier(features)
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
    

class Classifier(nn.Module):
    def __init__(self, in_channels: int, n_classes: int, dropout_rate: float):
        super().__init__()

        level_channels = [in_channels, in_channels // 2]

        linear_nodes = [1024, 128, 32, n_classes]

        self.in_conv = layers.SingleConv(level_channels[0], level_channels[1], 1)
        self.down_1 = layers.DownConv(level_channels[1], level_channels[1], 3, 3)
        self.down_2 = layers.DownConv(level_channels[1], level_channels[1], 3, 3)
        self.fc_1 = nn.Linear(linear_nodes[0], linear_nodes[1])
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc_2 = nn.Linear(linear_nodes[1], linear_nodes[2])
        self.fc_3 = nn.Linear(linear_nodes[2], linear_nodes[3])

    def forward(self, features):
        x3 = features.get("x3")
        x = self.in_conv(x3)
        x = self.down_1(x)
        x = self.down_2(x)
        x = torch.flatten(x, 1)
        x = self.fc_1(x)
        x = self.dropout(x)
        x = self.fc_2(x)
        logits = self.fc_3(x)
        return logits
