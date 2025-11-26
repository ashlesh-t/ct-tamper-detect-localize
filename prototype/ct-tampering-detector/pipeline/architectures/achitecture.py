# pipeline/architectures/architecture.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_v2_m, EfficientNet_V2_M_Weights

class CBAM(nn.Module):
    def __init__(self, in_ch, reduction=16):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_ch, in_ch//reduction), 
            nn.ReLU(), 
            nn.Linear(in_ch//reduction, in_ch)
        )
        self.sig = nn.Sigmoid()
        self.spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        
    def forward(self, x):
        b, c, _, _ = x.shape
        avg = self.avg(x).view(b, c)
        maxv = self.max(x).view(b, c)
        att = self.sig(self.fc(avg) + self.fc(maxv)).view(b, c, 1, 1)
        x = x * att
        
        a = torch.mean(x, dim=1, keepdim=True)
        m, _ = torch.max(x, dim=1, keepdim=True)
        x = x * self.sig(self.spatial(torch.cat([a, m], dim=1)))
        return x

class MultiStreamCTModel(nn.Module):
    def __init__(self, pretrained=True, feature_dim=1280, num_classes=2):
        super().__init__()
        weights = EfficientNet_V2_M_Weights.DEFAULT if pretrained else None
        
        # Helper to make single-channel backbone
        def make_stream():
            m = efficientnet_v2_m(weights=weights)
            # Replace first conv to accept 1 channel
            first = list(m.features.children())[0][0]
            new_conv = nn.Conv2d(
                1, first.out_channels, 
                kernel_size=first.kernel_size, 
                stride=first.stride, 
                padding=first.padding, 
                bias=False
            )
            with torch.no_grad():
                w = first.weight
                w_mean = w.mean(dim=1, keepdim=True)
                new_conv.weight.copy_(w_mean)
            m.features[0][0] = new_conv
            feat = nn.Sequential(*list(m.features.children()))
            return feat
        
        self.ct_stream = make_stream()
        self.roi_stream = make_stream() 
        self.fft_stream = make_stream()
        
        self.cbam1 = CBAM(feature_dim)
        self.cbam2 = CBAM(feature_dim)
        self.cbam3 = CBAM(feature_dim)
        
        self.cross_attn = nn.MultiheadAttention(feature_dim, num_heads=8, batch_first=True)
        
        self.fusion = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), 
            nn.Flatten(),
            nn.BatchNorm1d(feature_dim * 3),
            nn.Linear(feature_dim * 3, 1024), 
            nn.ReLU(), 
            nn.Dropout(0.4),
            nn.Linear(1024, 512), 
            nn.ReLU(), 
            nn.Dropout(0.3)
        )
        
        self.classifier = nn.Linear(512, num_classes)
        
        # Auxiliary classifiers (needed for loading weights)
        self.aux_ct = nn.Linear(feature_dim, num_classes)
        self.aux_roi = nn.Linear(feature_dim, num_classes)
        self.aux_fft = nn.Linear(feature_dim, num_classes)

    def forward(self, x, return_aux=False):
        ct = x[:, 0:1, :, :]    # Channel 0: CT
        roi = x[:, 1:2, :, :]   # Channel 1: ROI  
        fft = x[:, 2:3, :, :]   # Channel 2: FFT

        fct = self.ct_stream(ct)
        froi = self.roi_stream(roi) 
        ffft = self.fft_stream(fft)
        
        fct = self.cbam1(fct)
        froi = self.cbam2(froi)
        ffft = self.cbam3(ffft)
        
        b, c, h, w = fct.shape
        
        def to_seq(t): 
            return t.view(b, c, -1).permute(0, 2, 1)
            
        def from_seq(seq):
            return seq.permute(0, 2, 1).view(b, c, h, w)
        
        seq_ct = to_seq(fct)
        seq_roi = to_seq(froi)
        seq_fft = to_seq(ffft)
        
        ct_att, _ = self.cross_attn(seq_ct, seq_roi, seq_roi)
        roi_att, _ = self.cross_attn(seq_roi, seq_fft, seq_fft)
        fft_att, _ = self.cross_attn(seq_fft, seq_ct, seq_ct)
        
        ct_att = from_seq(ct_att)
        roi_att = from_seq(roi_att)
        fft_att = from_seq(fft_att)
        
        fused = torch.cat([ct_att, roi_att, fft_att], dim=1)
        out_feat = self.fusion(fused)
        logits = self.classifier(out_feat)
        
        if return_aux:
            aux_ct = self.aux_ct(F.adaptive_avg_pool2d(fct, 1).flatten(1))
            aux_roi = self.aux_roi(F.adaptive_avg_pool2d(froi, 1).flatten(1))
            aux_fft = self.aux_fft(F.adaptive_avg_pool2d(ffft, 1).flatten(1))
            return logits, (aux_ct, aux_roi, aux_fft), out_feat
            
        return logits, out_feat
# Add this to the end of architecture.py (after MultiStreamCTModel)

from torchvision.models import densenet121, DenseNet121_Weights

class DenseNetBinary(nn.Module):
    def __init__(self, pretrained=True, num_classes=2):
        super().__init__()
        weights = DenseNet121_Weights.DEFAULT if pretrained else None
        m = densenet121(weights=weights)
        m.classifier = nn.Linear(m.classifier.in_features, num_classes)
        self.model = m

    def forward(self, x):
        return self.model(x)