import torch
import torch.nn as nn

class FrequencyBranch(nn.Module):
    def __init__(self, in_channels=3, base_filters=64, num_classes=5):
        super(FrequencyBranch, self).__init__()
        
        # 频域特征提取网络：使用轻量级 CNN 处理幅度谱
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
        
        # 用于独立收敛测试的分类头
        self.fc = nn.Linear(base_filters * 4, num_classes)

    def forward_fft(self, x):
        """
        核心操作：对图像进行 2D FFT 变换并获取中心化的对数幅度谱
        """
        # 1. 2D FFT 变换
        fft_x = torch.fft.fft2(x, dim=(-2, -1))
        
        # 2. 将低频分量移到频谱中心处 (fftshift)
        fft_shift = torch.fft.fftshift(fft_x, dim=(-2, -1))
        
        # 3. 获取幅度谱
        magnitude_spectrum = torch.abs(fft_shift)
        
        # 4. 对幅度谱进行对数变换，将极大的值域压缩，防止梯度爆炸
        epsilon = 1e-8
        log_magnitude = torch.log(magnitude_spectrum + epsilon)
        
        return log_magnitude

    def forward(self, x):
        # 1. 提取频域对数幅度谱
        freq_inputs = self.forward_fft(x)
        
        # 2. 提取频域深层特征
        x = self.features(freq_inputs)
        
        # 3. 展平特征向量 (Batch, base_filters * 4)
        feature_vector = torch.flatten(x, 1)
        
        # 4. 输出预测结果
        out = self.fc(feature_vector)
        
        # 返回 out 用于独立计算 loss 验证收敛，返回 feature_vector 用于后续与空域融合
        return out, feature_vector

