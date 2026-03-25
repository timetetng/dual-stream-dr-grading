import torch
import torch.nn as nn
import torchvision.models as models

class SpatialBranch(nn.Module):
    def __init__(self, embed_dim=512):
        super(SpatialBranch, self).__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V1
        resnet = models.resnet50(weights=weights)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.num_ftrs = resnet.fc.in_features
        
        self.projector = nn.Sequential(
            nn.Linear(self.num_ftrs, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3)  # 新增 Dropout 缓解小规模数据集过拟合
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        feat = self.projector(x)
        return feat

class FrequencyBranch(nn.Module):
    def __init__(self, in_channels=3, base_filters=64, embed_dim=512):
        super(FrequencyBranch, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, base_filters, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(base_filters, base_filters * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_filters * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters * 2, base_filters * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_filters * 4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.num_ftrs = base_filters * 4
        
        self.projector = nn.Sequential(
            nn.Linear(self.num_ftrs, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3) # 新增 Dropout
        )

    def forward_fft(self, x):
        fft_x = torch.fft.fft2(x, dim=(-2, -1))
        fft_shift = torch.fft.fftshift(fft_x, dim=(-2, -1))
        magnitude_spectrum = torch.abs(fft_shift)
        epsilon = 1e-8
        log_magnitude = torch.log(magnitude_spectrum + epsilon)
        return log_magnitude

    def forward(self, x):
        freq_inputs = self.forward_fft(x)
        x = self.features(freq_inputs)
        x = torch.flatten(x, 1)
        feat = self.projector(x)
        return feat

class DynamicGatedFusion(nn.Module):
    def __init__(self, embed_dim=512):
        super(DynamicGatedFusion, self).__init__()
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim // 2, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, feat_spatial, feat_freq):
        concat_feat = torch.cat([feat_spatial, feat_freq], dim=1)
        weights = self.gate(concat_feat)
        w_spatial = weights[:, 0].unsqueeze(1)
        w_freq = weights[:, 1].unsqueeze(1)
        fused_feat = w_spatial * feat_spatial + w_freq * feat_freq
        return fused_feat, w_spatial, w_freq

class ConcatFusion(nn.Module):
    def __init__(self, embed_dim=512):
        super(ConcatFusion, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, feat_spatial, feat_freq):
        concat_feat = torch.cat([feat_spatial, feat_freq], dim=1)
        fused_feat = self.fc(concat_feat)
        return fused_feat, None, None

class DualStreamNet(nn.Module):
    def __init__(self, num_classes=5, embed_dim=512, use_freq=True, fusion_type='gated'):
        super(DualStreamNet, self).__init__()
        self.use_freq = use_freq
        self.fusion_type = fusion_type
        
        self.spatial_branch = SpatialBranch(embed_dim=embed_dim)
        
        if self.use_freq:
            self.freq_branch = FrequencyBranch(embed_dim=embed_dim)
            if self.fusion_type == 'gated':
                self.fusion_module = DynamicGatedFusion(embed_dim=embed_dim)
            elif self.fusion_type == 'concat':
                self.fusion_module = ConcatFusion(embed_dim=embed_dim)
            else:
                raise ValueError("fusion_type must be 'gated' or 'concat'")
                
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        feat_spatial = self.spatial_branch(x)
        
        if not self.use_freq:
            return self.classifier(feat_spatial)
            
        feat_freq = self.freq_branch(x)
        fused_feat, _, _ = self.fusion_module(feat_spatial, feat_freq)
        
        out = self.classifier(fused_feat)
        return out
