# src/models/fusion.py
import torch
import torch.nn as nn
import torchvision.models as models

class AdaptiveConcatPool2d(nn.Module):
    """
    混合池化层：将全局最大池化和全局平均池化的结果在通道维度拼接。
    对于 DR 任务，最大池化抓取突变病灶（如出血点），平均池化保留整体视网膜背景信息。
    """
    def __init__(self, sz=(1, 1)):
        super(AdaptiveConcatPool2d, self).__init__()
        self.ap = nn.AdaptiveAvgPool2d(sz)
        self.mp = nn.AdaptiveMaxPool2d(sz)
        
    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], dim=1)

# 空域分支，尝试加上多层混合池化
class SpatialBranch(nn.Module):
    def __init__(self, embed_dim=512):
        super(SpatialBranch, self).__init__()
        # 使用泛化性能更强的 V2 预训练权重
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        resnet = models.resnet50(weights=weights)
        
        # 将 ResNet50 拆解，以便后续提取多尺度特征
        self.stem = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        )
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3  # Layer3 输出通道数为 1024
        self.layer4 = resnet.layer4  # Layer4 输出通道数为 2048
        
        # 使用混合池化，注意这会使原本的通道数翻倍
        self.pool = AdaptiveConcatPool2d((1, 1))
        
        # 计算融合后的总通道数：
        # Layer3: 1024 通道 -> ConcatPool -> 2048
        # Layer4: 2048 通道 -> ConcatPool -> 4096
        # 拼接后总和: 2048 + 4096 = 6144
        self.num_ftrs = 6144
        
        # 优化后的投影头，采用两层结构和 SiLU 激活函数以实现更好的非线性降维
        self.projector = nn.Sequential(
            nn.Linear(self.num_ftrs, 1024),
            nn.BatchNorm1d(1024),
            nn.SiLU(inplace=True),
            nn.Dropout(p=0.4),  # 特征维度变大，适当增加 Dropout 防止过拟合
            nn.Linear(1024, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        # 前置基础特征提取
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        
        # 提取多尺度特征
        feat3 = self.layer3(x)      # (Batch, 1024, H/16, W/16)
        feat4 = self.layer4(feat3)  # (Batch, 2048, H/32, W/32)
        
        # 分别进行混合池化并展平
        p3 = torch.flatten(self.pool(feat3), 1)  # (Batch, 2048)
        p4 = torch.flatten(self.pool(feat4), 1)  # (Batch, 4096)
        
        # 在通道维度拼接多尺度特征
        concat_feat = torch.cat([p3, p4], dim=1) # (Batch, 6144)
        
        # 降维映射到目标 embed_dim
        feat = self.projector(concat_feat)
        
        return feat

# =================================================================
# 旧架构：直接吃进对数幅度谱 
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
# 新架构：数学先验驱动的病灶感知分支 
# =================================================================
class HighFreqLesionBranch(nn.Module):
    def __init__(self, in_channels=1, base_filters=64, embed_dim=512, init_radius=40.0):
        super(HighFreqLesionBranch, self).__init__()
        
        # 【核心修改 1】：将 radius 注册为可学习的 Parameter
        self.radius = nn.Parameter(torch.tensor(float(init_radius)))
        
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
        """数学计算层：GPU 实时处理 FFT -> 可学习高斯高通滤波 -> iFFT"""
        # 转灰度，剥离颜色干扰
        if x.size(1) == 3:
            x_gray = 0.299 * x[:, 0:1, :, :] + 0.587 * x[:, 1:2, :, :] + 0.114 * x[:, 2:3, :, :]
        else:
            x_gray = x

        B, C, H, W = x_gray.shape
        fft_x = torch.fft.fft2(x_gray, dim=(-2, -1))
        fft_shift = torch.fft.fftshift(fft_x, dim=(-2, -1))

        # 动态生成高斯掩膜
        y = torch.arange(H, device=x.device).view(-1, 1) - H // 2
        x_coord = torch.arange(W, device=x.device).view(1, -1) - W // 2
        d_sq = x_coord**2 + y**2 
        
        # 确保 radius 始终为正，防止除以0
        actual_radius = torch.abs(self.radius) + 1e-6 
        mask = 1.0 - torch.exp(-d_sq / (2 * actual_radius**2))
        mask = mask.view(1, 1, H, W).to(fft_shift.dtype)

        fshift_hpf = fft_shift * mask
        f_ishift_hpf = torch.fft.ifftshift(fshift_hpf, dim=(-2, -1))
        img_back_hpf = torch.abs(torch.fft.ifft2(f_ishift_hpf, dim=(-2, -1)))
        
        # 归一化以稳定 CNN 训练
        max_vals = img_back_hpf.view(B, -1).max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        img_back_hpf = img_back_hpf / (max_vals + 1e-8)

        return img_back_hpf

    def forward(self, x):
        # 【核心修改 2】：移除 torch.no_grad()，让 FFT 滤波过程的梯度能够回传到 self.radius
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
    def __init__(self, num_classes=5, embed_dim=512, use_freq=True, fusion_type='gated', freq_type='math_prior', init_radius=40.0):
        super(DualStreamNet, self).__init__()
        self.use_freq = use_freq
        self.fusion_type = fusion_type
        self.freq_type = freq_type
        
        self.spatial_branch = SpatialBranch(embed_dim=embed_dim)
        
        if self.use_freq:
            if self.freq_type == 'math_prior':
                # 【核心修改 3】：接收并传递 init_radius 参数
                self.freq_branch = HighFreqLesionBranch(embed_dim=embed_dim, init_radius=init_radius)
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
