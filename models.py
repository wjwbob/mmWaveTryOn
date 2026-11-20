"""The six mmWaveTryOn models from the original Table 3 training runs.

Radar input is [batch, time, RGB, height, width]. Skeleton input is
[batch, time, 13, 2]. Fusion models receive radar first and skeleton second.
Layer names and tensor dimensions follow the original trained weights.
"""

import torch
from torch import nn
from torch.nn import functional as F


class _BatchNorm1d(nn.BatchNorm1d):
    """Use running statistics when a training batch has one value per channel."""

    def forward(self, x):
        if self.training and x.numel() == x.size(1):
            return F.batch_norm(
                x, self.running_mean, self.running_var, self.weight, self.bias,
                training=False, eps=self.eps,
            )
        return super().forward(x)


class CNN1D(nn.Module):
    """Encode a single skeleton frame by convolving across the joints."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(2, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(128, 128)

    def forward(self, frame):
        x = frame.transpose(1, 2)
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.bn3(F.relu(self.conv3(x)))
        return self.proj(self.gap(x).squeeze(-1))


class Skeleton1DCNN_MLP(nn.Module):
    """Convolve across time with the joint coordinates as input channels."""

    def __init__(self, num_classes):
        super().__init__()
        self.conv_layers = nn.ModuleList()
        for in_channels, out_channels in ((26, 32), (32, 64), (64, 128)):
            self.conv_layers.append(nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
                _BatchNorm1d(out_channels),
                nn.ReLU(),
            ))
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.1)

    def forward(self, skeletons):
        x = skeletons.flatten(2).transpose(1, 2)
        for layer in self.conv_layers:
            x = layer(x)
        features = self.global_avg_pool(x).flatten(1)
        features = F.relu(self.fc1(self.dropout(features)))
        return self.fc2(self.dropout(features))


class SkelCNN_LSTM_MLP(nn.Module):
    """Combine frame-level joint features using a unidirectional LSTM."""

    def __init__(self, num_classes):
        super().__init__()
        self.frame_cnn = CNN1D()
        self.lstm = nn.LSTM(128, 256, batch_first=True)
        self.fc_2 = nn.Linear(256, 128)
        self.fc_3 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.1)

    def forward(self, skeletons):
        batch, time = skeletons.shape[:2]
        frames = self.frame_cnn(skeletons.flatten(0, 1))
        sequence, _ = self.lstm(frames.reshape(batch, time, -1))
        features = F.relu(self.fc_2(self.dropout(sequence[:, -1])))
        return self.fc_3(self.dropout(features))


class Image2DCNN_MLP(nn.Module):
    """Pool spatial features within each radar frame, then across time."""

    def __init__(self, num_classes):
        super().__init__()
        self.conv_layers = nn.ModuleList()
        for in_channels, out_channels in ((3, 32), (32, 64), (64, 128)):
            self.conv_layers.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.MaxPool2d(2),
            ))
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.1)

    def forward(self, images):
        batch, time = images.shape[:2]
        x = images.flatten(0, 1)
        for layer in self.conv_layers:
            x = layer(x)
        sequence = self.gap(x).reshape(batch, time, -1)
        features = self.temporal_pool(sequence.transpose(1, 2)).squeeze(-1)
        features = F.relu(self.fc1(self.dropout(features)))
        return self.fc2(self.dropout(features))


class CNN_Net(nn.Module):
    """Use the final LSTM output after frame-wise radar feature extraction."""

    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.maxpool = nn.MaxPool2d(2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.lstm = nn.LSTM(128, 512, batch_first=True)
        self.fc_2 = nn.Linear(512, 128)
        self.fc_3 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.1)

    def extract_frame_features(self, frame):
        x = self.maxpool(F.relu(self.bn1(self.conv1(frame))))
        x = self.maxpool(F.relu(self.bn2(self.conv2(x))))
        x = self.maxpool(F.relu(self.bn3(self.conv3(x))))
        return self.gap(x).flatten(1)

    def forward(self, images):
        # The original recurrent radar encoder updates batch normalization per frame.
        frames = [self.extract_frame_features(frame) for frame in images.unbind(1)]
        sequence, _ = self.lstm(torch.stack(frames, dim=1))
        features = self.fc_2(self.dropout(sequence[:, -1]))
        return self.fc_3(self.dropout(features))


class MultiModalCNN_LSTM_MLP(nn.Module):
    """The concatenation and attention variants used different classifier heads."""

    def __init__(self, num_classes, attention):
        super().__init__()
        self.attention = attention
        self.image_conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.image_bn1 = nn.BatchNorm2d(32)
        self.image_conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.image_bn2 = nn.BatchNorm2d(64)
        self.image_conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.image_bn3 = nn.BatchNorm2d(128)
        self.image_maxpool = nn.MaxPool2d(2)
        self.image_gap = nn.AdaptiveAvgPool2d(1)
        self.image_proj = nn.Linear(128, 128)
        self.skeleton_cnn = CNN1D()
        if attention:
            self.image_attention = nn.Sequential(nn.Linear(128, 64), nn.Tanh(), nn.Linear(64, 1))
            self.skeleton_attention = nn.Sequential(nn.Linear(128, 64), nn.Tanh(), nn.Linear(64, 1))
        self.lstm = nn.LSTM(256, 256, batch_first=True)
        self.fc_2 = nn.Linear(256, 128)
        if attention:
            self.bn_fc2 = _BatchNorm1d(128)
        self.fc_3 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.1)
        if attention:
            self.dropout_heavy = nn.Dropout(0.2)

    def extract_image_frame_features(self, frame):
        x = self.image_maxpool(F.relu(self.image_bn1(self.image_conv1(frame))))
        x = self.image_maxpool(F.relu(self.image_bn2(self.image_conv2(x))))
        x = self.image_maxpool(F.relu(self.image_bn3(self.image_conv3(x))))
        return self.image_proj(self.image_gap(x).flatten(1))

    def forward(self, images, skeletons):
        if images.shape[:2] != skeletons.shape[:2]:
            raise ValueError("Radar and skeleton inputs must share batch size and sequence length.")
        image_features = [self.extract_image_frame_features(frame) for frame in images.unbind(1)]
        skeleton_features = [self.skeleton_cnn(frame) for frame in skeletons.unbind(1)]
        fused = []
        for image, skeleton in zip(image_features, skeleton_features):
            if self.attention:
                scores = torch.cat((self.image_attention(image), self.skeleton_attention(skeleton)), dim=1)
                weights = F.softmax(scores, dim=1)
                image = image * weights[:, :1]
                skeleton = skeleton * weights[:, 1:]
            fused.append(torch.cat((image, skeleton), dim=1))
        sequence, _ = self.lstm(torch.stack(fused, dim=1))
        features = self.fc_2(self.dropout(sequence[:, -1]))
        if self.attention:
            features = self.dropout_heavy(F.relu(self.bn_fc2(features)))
        else:
            features = self.dropout(F.relu(features))
        return self.fc_3(features)


MODEL_NAMES = ("skeleton-cnn", "skeleton-lstm", "radar-cnn", "radar-lstm", "concat", "attention")


def build_model(name, num_classes=11):
    """Build a Table 3 architecture and return its untrained classifier."""
    if name in ("concat", "attention"):
        return MultiModalCNN_LSTM_MLP(num_classes, attention=name == "attention")
    constructors = {
        "skeleton-cnn": Skeleton1DCNN_MLP,
        "skeleton-lstm": SkelCNN_LSTM_MLP,
        "radar-cnn": Image2DCNN_MLP,
        "radar-lstm": CNN_Net,
    }
    if name not in constructors:
        raise ValueError(f"Unknown model {name!r}. Choose from {', '.join(MODEL_NAMES)}.")
    return constructors[name](num_classes)
