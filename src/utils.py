import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import (
    cohen_kappa_score, 
    confusion_matrix, 
    recall_score, 
    f1_score, 
    roc_auc_score
)

def calculate_qwk(y_true, y_pred):
    """计算二次加权 Kappa (QWK)"""
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')

def calculate_medical_metrics(y_true, y_pred, y_probs=None, num_classes=5):
    """
    计算多分类医学图像评价指标 (Macro Average)
    """
    labels = list(range(num_classes))
    
    # 召回率 (敏感度 Sensitivity) 和 F1 Score
    recall = recall_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)
    
    # 特异性 Specificity (多分类下利用混淆矩阵计算 Macro 特异性)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    specificities = []
    for i in range(num_classes):
        tn = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]
        fp = np.sum(cm[:, i]) - cm[i, i]
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificities.append(spec)
    specificity = np.mean(specificities)
    
    # AUC 曲线下面积 (OvR 多分类策略)
    auc = float('nan')
    if y_probs is not None:
        try:
            auc = roc_auc_score(y_true, y_probs, multi_class='ovr', average='macro')
        except ValueError:
            pass  # 如果 batch 太小导致某些类缺失，跳过报错
            
    return recall, specificity, f1, auc

def plot_confusion_matrix(y_true, y_pred, save_path):
    """生成并保存混淆矩阵图像，用于分析跨级误判"""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['0', '1', '2', '3', '4'],
                yticklabels=['0', '1', '2', '3', '4'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_training_curves(history, save_path):
    """绘制并保存训练过程的 Loss 和 QWK 曲线"""
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color='tab:red')
    ax1.plot(epochs, history['train_loss'], 'r--', label='Train Loss')
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    
    ax2 = ax1.twinx()  
    ax2.set_ylabel('QWK Score', color='tab:blue')  
    ax2.plot(epochs, history['val_qwk'], 'b-', marker='o', label='Val QWK')
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')
    
    plt.title('Training and Validation Metrics')
    fig.tight_layout()  
    plt.savefig(save_path, dpi=300)
    plt.close()
