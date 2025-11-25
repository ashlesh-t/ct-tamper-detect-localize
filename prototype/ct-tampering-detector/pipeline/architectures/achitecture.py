# pipeline/architectures/architecture.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_v2_m

# Copy the exact same architecture definitions from your testing code
# (ChannelAttention, SpatialAttention, CBAM, MultiStreamCTModel)

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        out = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * out

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv(x_cat))
        return x * out

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(in_channels, reduction)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x

class MultiStreamCTModel(nn.Module):
    def __init__(self, num_classes=2, pretrained=False):
        super().__init__()
        # Stream 1: CT
        self.ct_backbone = efficientnet_v2_m(weights=None)
        self.ct_backbone.features[0][0] = nn.Conv2d(1, 24, kernel_size=3, stride=2, padding=1, bias=False)
        self.ct_features = nn.Sequential(*list(self.ct_backbone.children())[:-1])

        # Stream 2: ROI
        self.roi_backbone = efficientnet_v2_m(weights=None)
        self.roi_backbone.features[0][0] = nn.Conv2d(1, 24, kernel_size=3, stride=2, padding=1, bias=False)
        self.roi_features = nn.Sequential(*list(self.roi_backbone.children())[:-1])

        # Stream 3: FFT
        self.fft_backbone = efficientnet_v2_m(weights=None)
        self.fft_backbone.features[0][0] = nn.Conv2d(1, 24, kernel_size=3, stride=2, padding=1, bias=False)
        self.fft_features = nn.Sequential(*list(self.fft_backbone.children())[:-1])

        feature_dim = 1280
        self.ct_cbam = CBAM(feature_dim)
        self.roi_cbam = CBAM(feature_dim)
        self.fft_cbam = CBAM(feature_dim)
        self.cross_attention = nn.MultiheadAttention(feature_dim, num_heads=8, batch_first=True)

        self.fusion = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_dim * 3, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

        # Aux layers (needed to load weights successfully, even if unused)
        self.ct_aux = nn.Linear(feature_dim, num_classes)
        self.roi_aux = nn.Linear(feature_dim, num_classes)
        self.fft_aux = nn.Linear(feature_dim, num_classes)

    def forward(self, x, return_aux=False):
        ct_img = x[:, 0:1, :, :]  # Channel 0: CT
        roi_img = x[:, 1:2, :, :]  # Channel 1: ROI  
        fft_img = x[:, 2:3, :, :]  # Channel 2: FFT

        ct_feat = self.ct_features(ct_img)
        roi_feat = self.roi_features(roi_img)
        fft_feat = self.fft_features(fft_img)

        ct_feat = self.ct_cbam(ct_feat)
        roi_feat = self.roi_cbam(roi_feat)
        fft_feat = self.fft_cbam(fft_feat)

        b, c, h, w = ct_feat.shape
        ct_flat = ct_feat.view(b, c, -1).permute(0, 2, 1)
        roi_flat = roi_feat.view(b, c, -1).permute(0, 2, 1)
        fft_flat = fft_feat.view(b, c, -1).permute(0, 2, 1)

        ct_attended, _ = self.cross_attention(ct_flat, roi_flat, roi_flat)
        roi_attended, _ = self.cross_attention(roi_flat, fft_flat, fft_flat)
        fft_attended, _ = self.cross_attention(fft_flat, ct_flat, ct_flat)

        ct_attended = ct_attended.permute(0, 2, 1).view(b, c, h, w)
        roi_attended = roi_attended.permute(0, 2, 1).view(b, c, h, w)
        fft_attended = fft_attended.permute(0, 2, 1).view(b, c, h, w)

        fused_feat = torch.cat([ct_attended, roi_attended, fft_attended], dim=1)
        fused = self.fusion(fused_feat)
        output = self.classifier(fused)
        return output