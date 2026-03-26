# src/models/fusion.py
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
            nn.Dropout(p=0.3)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        feat = self.projector(x)
        return feat

# =================================================================
# 旧架构：直接吃进对数幅度谱 (完全保留你原来的逻辑，用于 Baseline 对比)
# =================================================================
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
            nn.Dropout(p=0.3)
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

# =================================================================
# 新架构：数学先验驱动的病灶感知分支 (Math-Prior Driven)
# =================================================================
class HighFreqLesionBranch(nn.Module):
    def __init__(self, in_channels=1, base_filters=64, embed_dim=512, radius=10):
        super(HighFreqLesionBranch, self).__init__()
        self.radius = radius
        # 由于 iFFT 得到的是单通道物理残差图，这里的 in_channels 是 1
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
            nn.Dropout(p=0.3)
        )

    def apply_math_filter(self, x):
        """数学计算层：GPU 实时处理 FFT -> 高斯高通滤波 -> iFFT"""
        # 转灰度，剥离颜色干扰，只看物理结构突变
        if x.size(1) == 3:
            x_gray = 0.299 * x[:, 0:1, :, :] + 0.587 * x[:, 1:2, :, :] + 0.114 * x[:, 2:3, :, :]
        else:
            x_gray = x

        B, C, H, W = x_gray.shape
        fft_x = torch.fft.fft2(x_gray, dim=(-2, -1))
        fft_shift = torch.fft.fftshift(fft_x, dim=(-2, -1))

        # 动态生成高斯掩膜 (无需反向传播)
        y = torch.arange(H, device=x.device).view(-1, 1) - H // 2
        x_coord = torch.arange(W, device=x.device).view(1, -1) - W // 2
        d_sq = x_coord**2 + y**2 
        mask = 1.0 - torch.exp(-d_sq / (2 * self.radius**2))
        mask = mask.view(1, 1, H, W).to(fft_shift.dtype)

        fshift_hpf = fft_shift * mask
        f_ishift_hpf = torch.fft.ifftshift(fshift_hpf, dim=(-2, -1))
        img_back_hpf = torch.abs(torch.fft.ifft2(f_ishift_hpf, dim=(-2, -1)))
        
        # 归一化以稳定 CNN 训练
        max_vals = img_back_hpf.view(B, -1).max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        img_back_hpf = img_back_hpf / (max_vals + 1e-8)

        return img_back_hpf

    def forward(self, x):
        with torch.no_grad(): # 滤波过程是纯数学运算，不计算梯度
            high_freq_map = self.apply_math_filter(x)
            
        x = self.features(high_freq_map)
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
    def __init__(self, num_classes=5, embed_dim=512, use_freq=True, fusion_type='gated', freq_type='math_prior'):
        super(DualStreamNet, self).__init__()
        self.use_freq = use_freq
        self.fusion_type = fusion_type
        self.freq_type = freq_type # 新增：用于决定实例化哪种频域分支
        
        self.spatial_branch = SpatialBranch(embed_dim=embed_dim)
        
        if self.use_freq:
            # 动态选择频域分支
            if self.freq_type == 'math_prior':
                self.freq_branch = HighFreqLesionBranch(embed_dim=embed_dim, radius=40)
            else:
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
