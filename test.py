import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, 
    recall_score, 
    f1_score, 
    roc_auc_score, 
    confusion_matrix,
    cohen_kappa_score
)

from src.dataset import APTOSDataset
from src.models.fusion import SpatialBranch, FrequencyBranch

# ----------------- 兼容旧权重的模型结构 (V2) -----------------
class DynamicGatedFusionV2(nn.Module):
    def __init__(self, embed_dim=512):
        super(DynamicGatedFusionV2, self).__init__()
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(32, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, feat_spatial, feat_freq):
        concat_feat = torch.cat([feat_spatial, feat_freq], dim=1)
        weights = self.gate(concat_feat)
        w_spatial = weights[:, 0].unsqueeze(1)
        w_freq = weights[:, 1].unsqueeze(1)
        fused_feat = w_spatial * feat_spatial + w_freq * feat_freq
        return fused_feat, w_spatial, w_freq

class DualStreamNetV2(nn.Module):
    def __init__(self, num_classes=5, embed_dim=512, use_freq=True, fusion_type='gated'):
        super(DualStreamNetV2, self).__init__()
        self.use_freq = use_freq
        self.spatial_branch = SpatialBranch(embed_dim=embed_dim)
        
        if self.use_freq:
            self.freq_branch = FrequencyBranch(embed_dim=embed_dim)
            self.fusion_module = DynamicGatedFusionV2(embed_dim=embed_dim)
            
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        feat_spatial = self.spatial_branch(x)
        if not self.use_freq:
            return self.classifier(feat_spatial)
            
        feat_freq = self.freq_branch(x)
        fused_feat, _, _ = self.fusion_module(feat_spatial, feat_freq)
        out = self.classifier(fused_feat)
        return out

# ----------------- 数据加载与评估函数 -----------------
def get_test_dataloader(csv_path, img_dir, batch_size=16, num_workers=4):
    df = pd.read_csv(csv_path)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_dataset = APTOSDataset(df, img_dir, transform=test_transform, is_train=True)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True
    )
    return test_loader

def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Testing")
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)

def calculate_comprehensive_metrics(y_true, y_pred, y_probs):
    acc = accuracy_score(y_true, y_pred)
    sensitivity = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred)
    num_classes = cm.shape[0]
    specificities = []
    for i in range(num_classes):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        fp = np.sum(cm[:, i]) - tp
        tn = np.sum(cm) - (tp + fp + fn)
        spec_i = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificities.append(spec_i)
    specificity = np.mean(specificities)
    
    try:
        auc = roc_auc_score(y_true, y_probs, multi_class='ovr', average='macro')
    except ValueError:
        auc = float('nan')
        
    qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')
    
    return {
        "Accuracy": acc,
        "Sensitivity (Macro)": sensitivity,
        "Specificity (Macro)": specificity,
        "F1 Score (Macro)": f1,
        "AUC (Macro OVR)": auc,
        "QWK": qwk
    }

# ----------------- 主函数 -----------------
def main():
    test_csv_path = 'data/raw/test.csv'  
    test_img_dir = 'data/processed/test_images'
    weights_path = 'outputs/weights/best_dual_stream_ordinal.pth' 
    
    model_config = {
        "use_freq": True, 
        "fusion_type": "gated"
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Loading test data...")
    test_loader = get_test_dataloader(test_csv_path, test_img_dir, batch_size=8, num_workers=4)
    
    print("Initializing model...")
    model = DualStreamNetV2(
        num_classes=5, 
        embed_dim=512, 
        use_freq=model_config['use_freq'], 
        fusion_type=model_config['fusion_type']
    ).to(device)
    
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"Loaded weights from {weights_path}")
    else:
        raise FileNotFoundError(f"Weights file not found: {weights_path}")
        
    y_true, y_pred, y_probs = evaluate_model(model, test_loader, device)
    metrics = calculate_comprehensive_metrics(y_true, y_pred, y_probs)
    
    print("\n" + "="*50)
    print("🏆 Test Set Evaluation Results 🏆")
    print("="*50)
    for metric_name, value in metrics.items():
        print(f"{metric_name:25s}: {value:.4f}")
    print("="*50)

if __name__ == '__main__':
    main()
