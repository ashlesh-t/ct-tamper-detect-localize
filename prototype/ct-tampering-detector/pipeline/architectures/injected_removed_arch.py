# pipeline/architectures/injected_removed_arch.py
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b2, densenet121

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)

class EnhancedDualStreamFakeCT(nn.Module):
    """
    Architecture for injected vs removed classification
    This matches your training code exactly
    """
    def __init__(self, num_classes=2):
        super().__init__()
        # Stream 1: EfficientNet-B2
        backbone1 = efficientnet_b2(weights=None)
        self.stream1_features = backbone1.features
        self.stream1_attention = ChannelAttention(1408)  # EfficientNet-B2 feature channels
        self.stream1_pool = nn.AdaptiveAvgPool2d(1)

        # Stream 2: DenseNet121
        backbone2 = densenet121(weights=None)
        self.stream2_features = backbone2.features
        self.stream2_attention = ChannelAttention(1024)  # DenseNet121 feature channels
        self.stream2_pool = nn.AdaptiveAvgPool2d(1)

        # Fusion classifier
        self.fusion = nn.Sequential(
            nn.Linear(1408 + 1024, 1024), 
            nn.ReLU(), 
            nn.BatchNorm1d(1024), 
            nn.Dropout(0.5),
            nn.Linear(1024, 512), 
            nn.ReLU(), 
            nn.BatchNorm1d(512), 
            nn.Dropout(0.4),
            nn.Linear(512, 256), 
            nn.ReLU(), 
            nn.BatchNorm1d(256), 
            nn.Dropout(0.3),
            nn.Linear(256, 128), 
            nn.ReLU(), 
            nn.BatchNorm1d(128), 
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # Stream 1: Uses CT channel (channel 0) repeated to 3 channels
        x1 = x[:, 0:1].repeat(1, 3, 1, 1)
        f1 = self.stream1_features(x1)
        f1 = self.stream1_pool(f1 * self.stream1_attention(f1)).view(f1.size(0), -1)

        # Stream 2: Uses ROI and FFT channels (channels 1-2) + ROI repeated
        x2 = torch.cat([x[:, 1:3], x[:, 1:2]], dim=1)  # Creates 3 channels: ROI, FFT, ROI
        f2 = self.stream2_features(x2)
        f2 = self.stream2_pool(f2 * self.stream2_attention(f2)).view(f2.size(0), -1)

        return self.fusion(torch.cat([f1, f2], dim=1))