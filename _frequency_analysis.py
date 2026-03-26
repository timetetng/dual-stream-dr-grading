import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_sample_images(csv_path, img_dir):
    """从 CSV 中自动为每个诊断类别 (0-4) 寻找一张示例图片"""
    df = pd.read_csv(csv_path)
    samples = []
    # 遍历类别 0 到 4
    for label in range(5):
        # 提取该类别的第一张图像
        sample_row = df[df['diagnosis'] == label].iloc[0]
        img_id = sample_row['id_code']
        img_path = os.path.join(img_dir, f"{img_id}.png")
        samples.append((img_path, label))
    return samples

def gaussian_filter(shape, radius, high_pass=False):
    """
    生成频域高斯滤波器，避免理想滤波器带来的振铃效应 (Gibbs phenomenon)。
    高斯函数的傅里叶变换依然是高斯函数，平滑且无波纹。
    """
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    y, x = np.ogrid[:rows, :cols]
    
    # 计算频域中每个点到中心的距离平方: D(u,v)^2
    d_sq = (x - ccol)**2 + (y - crow)**2
    
    # 低通高斯函数: H(u,v) = e^(-D^2 / (2 * D0^2))
    mask = np.exp(-d_sq / (2 * radius**2))
    
    # 高通高斯函数: H(u,v) = 1 - e^(-D^2 / (2 * D0^2))
    if high_pass:
        mask = 1 - mask
        
    return mask

def analyze_and_reconstruct_fft(image_path, label, output_dir='./outputs/reports/fft_results', radius=10):
    """
    核心分析函数：读取图像 -> 2D FFT -> 高斯滤波 -> iFFT -> 热力图可视化
    """
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # 1. 读取原图 (灰度)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"无法读取图像: {image_path}")
        return
        
    rows, cols = img.shape

    # 2. 2D 傅里叶变换
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    
    # 幅度谱 (对数变换，防止低频能量淹没高频)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)

    # 3. 构造高斯滤波器 (取代原本毫无意义的黑白 Mask)
    mask_lpf = gaussian_filter((rows, cols), radius, high_pass=False)
    mask_hpf = gaussian_filter((rows, cols), radius, high_pass=True)

    # 4. 频域滤波与逆傅里叶变换 (iFFT)
    # -- 低频重建 (背景、光照不均)
    fshift_lpf = fshift * mask_lpf
    img_back_lpf = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift_lpf)))
    
    # -- 高频重建 (血管、出血点、硬性渗出等细节)
    fshift_hpf = fshift * mask_hpf
    img_back_hpf = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift_hpf)))

    # ==========================
    # 5. 可视化布局 (2x2 结构，直接舍弃无意义的 Mask 图)
    # ==========================
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle(f"Frequency Domain Analysis (Gaussian Filter) - Class {label}", fontsize=18, fontweight='bold', y=0.95)

    # (1) 带有标签的原图
    axes[0, 0].imshow(img, cmap='gray')
    axes[0, 0].set_title(f'Original Image\nDiagnosis Label: {label}', fontsize=14)
    axes[0, 0].axis('off')

    # (2) 频谱图 (换用 magma 伪彩，高能区域更醒目)
    axes[0, 1].imshow(magnitude_spectrum, cmap='magma')
    axes[0, 1].set_title('Log Magnitude Spectrum', fontsize=14)
    axes[0, 1].axis('off')

    # (3) 低频背景 (平滑的视网膜底色)
    axes[1, 0].imshow(img_back_lpf, cmap='gray')
    axes[1, 0].set_title('iFFT Low-Pass\n(Smooth Background / Illumination)', fontsize=14)
    axes[1, 0].axis('off')

    # (4) 高频病灶 (利用 hot 热力图，病灶和血管会像发光一样凸显)
    # 先做轻微的截断以增强对比度，否则个别极值会压暗整体
    hpf_display = np.clip(img_back_hpf, 0, np.percentile(img_back_hpf, 99.5))
    axes[1, 1].imshow(hpf_display, cmap='hot')
    axes[1, 1].set_title('iFFT High-Pass\n(Vessels & Lesions Highlight)', fontsize=14)
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.88) # 给主标题留空间
    
    # 按照要求命名：含有分类标签、ID
    plot_save_path = os.path.join(output_dir, f'Class_{label}_{base_name}_fft_analysis.png')
    plt.savefig(plot_save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[✔] 成功分析并保存: {plot_save_path}")

def main():
    csv_path = 'data/raw/train.csv'
    img_dir = 'data/processed/train_images'
    output_dir = './outputs/reports/fft_results'
    
    print("开始自动搜寻 0-4 级糖尿病视网膜样本...")
    try:
        samples = get_sample_images(csv_path, img_dir)
    except FileNotFoundError:
        print(f"错误: 找不到 CSV 文件 {csv_path}。请确保在项目根目录运行。")
        return
        
    for img_path, label in samples:
        if os.path.exists(img_path):
            analyze_and_reconstruct_fft(img_path, label, output_dir=output_dir, radius=40)
        else:
            print(f"警告: 图像不存在 {img_path}")
            
    print("所有分析完成！")

if __name__ == "__main__":
    main()
