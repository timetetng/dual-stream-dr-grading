# visualize.py
import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from src.models.fusion import DualStreamNet

def run_grad_cam(model_weights_path, image_path, output_path):
    """
    针对双流网络的空间分支进行 Grad-CAM 热力图生成，
    用于验证模型是否准确聚焦于 DR 病灶区域。
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. 初始化模型并加载权重
    model = DualStreamNet(num_classes=5, embed_dim=512, use_freq=True, fusion_type='gated').to(device)
    model.load_state_dict(torch.load(model_weights_path, map_location=device))
    model.eval()

    # 2. 指定想要可视化的目标层 (这里选择空间分支的 ResNet 的最后一层)
    # src/models/fusion.py 中 SpatialBranch 的 features 包含了 ResNet 去掉 fc 层的部分
    target_layers = [model.spatial_branch.features[-1]]

    # 3. 数据预处理 (需与训练时保持数学一致)
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_float = np.float32(img) / 255.0
    
    # 手动 normalize (mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    input_tensor = (img_float - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    input_tensor = torch.from_numpy(input_tensor).permute(2, 0, 1).unsqueeze(0).float().to(device)

    # 4. 初始化 Grad-CAM
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=torch.cuda.is_available())

    # 5. 生成热力图 (自动寻找网络预测概率最大的类别)
    targets = None 
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]

    # 6. 将热力图叠加在原图上
    visualization = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)

    # 7. 绘图保存
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(visualization)
    plt.title('Grad-CAM (Spatial Branch Focus)')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"热力图已保存至: {output_path}")

if __name__ == '__main__':
    # 请替换为你的实际路径
    # run_grad_cam('outputs/weights/best_dual_stream_ordinal.pth', 'data/processed/train_images/example.png', 'grad_cam_result.png')
    pass
