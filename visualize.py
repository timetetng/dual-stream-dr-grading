import os
import cv2
import torch
import numpy as np
import pandas as pd
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
    if image_tensor.dim() == 4:
        image_tensor = image_tensor.squeeze(0)
    
    gray_tensor = torch.mean(image_tensor, dim=0) 
    
    fft_x = torch.fft.fft2(gray_tensor)
    fft_shift = torch.fft.fftshift(fft_x)
    magnitude_spectrum = torch.abs(fft_shift)
    
    epsilon = 1e-8
    log_magnitude = torch.log(magnitude_spectrum + epsilon)
    
    log_magnitude = log_magnitude.cpu().numpy()
    log_magnitude = (log_magnitude - log_magnitude.min()) / (log_magnitude.max() - log_magnitude.min())
    log_magnitude = np.uint8(log_magnitude * 255)
    
    return log_magnitude

def main():
    weights_path = 'outputs/weights/best_dual_stream_ordinal.pth'
    img_dir = 'data/processed/train_images'
    csv_path = 'data/raw/train.csv'
    output_dir = 'outputs/reports/visualizations'
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Loading model and weights...")
    model = DualStreamNet(num_classes=5, embed_dim=512).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    
    target_layers = [model.spatial_branch.features[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print("Loading labels from CSV...")
    df = pd.read_csv(csv_path)
    
    # 修复 Pandas 警告，按 0-4 级各抽样 1 张
    sample_df = df.groupby('diagnosis').apply(lambda x: x.sample(1, random_state=42), include_groups=False).reset_index()
    
    # 创建 5 行 3 列 的对比大图
    fig, axes = plt.subplots(5, 3, figsize=(15, 25))
    print(f"Generating comprehensive grid for 5 classes...")
    
    for idx, row in sample_df.iterrows():
        img_id = row['id_code']
        true_label = row['diagnosis']  # 0, 1, 2, 3, 4
        img_name = f"{img_id}.png"
        img_path = os.path.join(img_dir, img_name)
        
        if not os.path.exists(img_path):
            print(f"Image {img_name} not found, skipping...")
            continue
            
        rgb_img = Image.open(img_path).convert('RGB')
        rgb_img_np = np.array(rgb_img, dtype=np.float32) / 255.0
        
        input_tensor = transform(rgb_img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(input_tensor)
            pred_class = torch.argmax(output, dim=1).item()
        
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
        cam_image = show_cam_on_image(rgb_img_np, grayscale_cam, use_rgb=True)
        
        fft_image = visualize_fft_spectrum(input_tensor)
        
        # 将图像画在对应的行 (0-4行对应0-4级)
        ax_orig = axes[true_label, 0]
        ax_fft = axes[true_label, 1]
        ax_cam = axes[true_label, 2]
        
        ax_orig.imshow(rgb_img)
        ax_orig.set_title(f"Class {true_label} Original\n({img_name})")
        ax_orig.axis('off')
        
        ax_fft.imshow(fft_image, cmap='magma')
        ax_fft.set_title(f"Class {true_label} FFT Spectrum")
        ax_fft.axis('off')
        
        ax_cam.imshow(cam_image)
        ax_cam.set_title(f"Class {true_label} Grad-CAM\n(Pred: {pred_class})")
        ax_cam.axis('off')
        
        print(f"Processed Class {true_label} -> {img_name}")

    plt.tight_layout()
    save_path = os.path.join(output_dir, "vis_all_classes_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved comprehensive visualization to {save_path}")

if __name__ == '__main__':
    main()
