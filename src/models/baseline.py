import torch.nn as nn
import torchvision.models as models

def build_baseline_resnet(num_classes=5):
    """
    构建基于 ResNet50 的单流基线模型，用于 DR 0-4 级分类
    """
    # 加载预训练的 ResNet50 权重
    model = models.resnet50(pretrained=True)
    
    # 替换最后的全连接层以适应 5 分类任务
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model

