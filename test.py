import os
import cv2
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def apply_math_filter_standalone(img_tensor, radius):
    # 转灰度
    if img_tensor.size(1) == 3:
        x_gray = 0.299 * img_tensor[:, 0:1, :, :] + 0.587 * img_tensor[:, 1:2, :, :] + 0.114 * img_tensor[:, 2:3, :, :]
    else:
        x_gray = img_tensor

    B, C, H, W = x_gray.shape
    fft_x = torch.fft.fft2(x_gray, dim=(-2, -1))
    fft_shift = torch.fft.fftshift(fft_x, dim=(-2, -1))

    # 生成高斯掩膜
    y = torch.arange(H, device=img_tensor.device).view(-1, 1) - H // 2
    x_coord = torch.arange(W, device=img_tensor.device).view(1, -1) - W // 2
    d_sq = x_coord**2 + y**2 
    mask = 1.0 - torch.exp(-d_sq / (2 * radius**2))
    mask = mask.view(1, 1, H, W).to(fft_shift.dtype)

    # 滤波与逆变换
    fshift_hpf = fft_shift * mask
    f_ishift_hpf = torch.fft.ifftshift(fshift_hpf, dim=(-2, -1))
    img_back_hpf = torch.abs(torch.fft.ifft2(f_ishift_hpf, dim=(-2, -1)))
    
    # 归一化
    max_vals = img_back_hpf.view(B, -1).max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
    img_back_hpf = img_back_hpf / (max_vals + 1e-8)

    return img_back_hpf

def visualize_radii(img_path, save_path="fft_radius_comparison.png"):
    """
    读取 384x384 图像并对比不同 radius 的滤波结果
    """
    if not os.path.exists(img_path):
        print(f"找不到图像: {img_path}")
        return

    # 读取并转为 RGB
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 转换为 PyTorch Tensor，形状 (1, 3, 384, 384)，范围 0-1
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    
    # 设置要对比的半径（10 保留了较多中低频，80 则极其严苛只留锐利边缘）
    radii_to_test = [10, 20, 30, 40, 60]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # 绘制原图
    axes[0].imshow(img)
    axes[0].set_title("Original Image (384x384)")
    axes[0].axis('off')
    
    # 绘制不同 radius 的滤波结果
    for i, r in enumerate(radii_to_test):
        filtered_tensor = apply_math_filter_standalone(img_tensor, radius=r)
        
        # 转换回 numpy 格式用于绘图 (H, W)
        filtered_img = filtered_tensor[0, 0].cpu().numpy()
        
        ax = axes[i + 1]
        # 使用 cmap='gray' 更好的观察结构特征
        ax.imshow(filtered_img, cmap='gray')
        ax.set_title(f"Math Prior (radius={r})")
        ax.axis('off')
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"可视化结果已保存至: {save_path}")
    plt.close()

if __name__ == "__main__":
    # 请将此处替换为你 data/processed/train_images/ 中的某张患病图像路径
    # 建议挑选一张包含较多"棉絮斑(软性渗出)"或"大片出血"的 3 级或 4 级样本
    SAMPLE_IMG_PATH = "data/processed/train_images/0a1076183736.png" 
    visualize_radii(SAMPLE_IMG_PATH)
