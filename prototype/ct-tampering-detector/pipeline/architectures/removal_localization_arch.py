# pipeline/architectures/removal_localization_arch.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiChannelUNet(nn.Module):
    """
    U-Net for removal localization with 4 input channels
    Matches your training code exactly
    """
    def __init__(self, n_channels=4, n_classes=1):
        super(MultiChannelUNet, self).__init__()
        
        def double_conv(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )
        
        # Encoder
        self.inc = double_conv(n_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), double_conv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), double_conv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), double_conv(256, 512))
        
        # Decoder
        self.up1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv1 = double_conv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv2 = double_conv(256, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv3 = double_conv(128, 64)
        
        # Output
        self.outc = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        # Decoder with skip connections
        x = self.up1(x4)
        x = torch.cat([x, x3], dim=1)
        x = self.conv1(x)
        
        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv2(x)
        
        x = self.up3(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv3(x)
        
        return self.outc(x)