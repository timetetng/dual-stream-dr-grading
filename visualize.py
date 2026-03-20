import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from src.models.fusion import DualStreamNet

def visualize_fft_spectrum(image_tensor):
    """
    计算并返回用于可视化的 FFT 对数幅度谱图像
    """
    # 转换为灰度图进行 FFT 可视化 (或直接对单通道进行)
    if image_tensor.dim() == 4:
        image_tensor = image_tensor.squeeze(0)
    
    # 提取 L 通道或取平均作为灰度
    gray_tensor = torch.mean(image_tensor, dim=0) 
    
    fft_x = torch.fft.fft2(gray_tensor)
    fft_shift = torch.fft.fftshift(fft_x)
    magnitude_spectrum = torch.abs(fft_shift)
    
    epsilon = 1e-8
    log_magnitude = torch.log(magnitude_spectrum + epsilon)
    
    # 归一化到 0-255 以便显示
    log_magnitude = log_magnitude.cpu().numpy()
    log_magnitude = (log_magnitude - log_magnitude.min()) / (log_magnitude.max() - log_magnitude.min())
    log_magnitude = np.uint8(log_magnitude * 255)
    
    return log_magnitude

def main():
    # 1. 配置路径与参数
    weights_path = 'outputs/weights/best_dual_stream_ordinal.pth'
    img_dir = 'data/processed/train_images'  # 使用处理好的高质量图像
    output_dir = 'outputs/reports/visualizations'
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 2. 加载双流网络模型与最佳权重
    print("Loading model and weights...")
    model = DualStreamNet(num_classes=5, embed_dim=512).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    
    # 3. 配置 Grad-CAM 目标层
    # 我们关注空域分支（ResNet50）的最后一层卷积网络，看看它学到了什么形态学特征
    target_layers = [model.spatial_branch.features[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    
    # 图像预处理流水线
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 4. 随机选取几张图像进行可视化测试 (你可以手动指定 id_code)
    # 这里假设目录下有这些图片，你需要替换成你真实存在的图片名
    sample_images = os.listdir(img_dir)[:5] 
    
    print(f"Generating visualizations for {len(sample_images)} images...")
    
    for img_name in sample_images:
        img_path = os.path.join(img_dir, img_name)
        
        # 读取原图
        rgb_img = Image.open(img_path).convert('RGB')
        rgb_img_np = np.array(rgb_img, dtype=np.float32) / 255.0  # 归一化到 [0, 1] 用于 Grad-CAM 叠加
        
        # 张量化
        input_tensor = transform(rgb_img).unsqueeze(0).to(device)
        
        # 获取预测结果
        with torch.no_grad():
            output = model(input_tensor)
            pred_class = torch.argmax(output, dim=1).item()
        
        # 生成 Grad-CAM 热力图
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
        cam_image = show_cam_on_image(rgb_img_np, grayscale_cam, use_rgb=True)
        
        # 生成 FFT 频谱图
        fft_image = visualize_fft_spectrum(input_tensor)
        
        # 5. 绘制并保存子图拼接结果
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(rgb_img)
        axes[0].set_title("Original Preprocessed Image")
        axes[0].axis('off')
        
        axes[1].imshow(fft_image, cmap='magma')
        axes[1].set_title("FFT Log Magnitude Spectrum")
        axes[1].axis('off')
        
        axes[2].imshow(cam_image)
        axes[2].set_title(f"Grad-CAM (Predicted Class: {pred_class})")
        axes[2].axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(output_dir, f"vis_{img_name}")
        plt.savefig(save_path, dpi=300)
        plt.close()
        
        print(f"Saved visualization for {img_name}")

if __name__ == '__main__':
    main()
