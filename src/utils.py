# src/utils.py
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 强制使用非交互式后端，防止与 PyTorch 多进程 DataLoader 冲突
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import (
    cohen_kappa_score, 
    confusion_matrix, 
    recall_score, 
    f1_score, 
    roc_auc_score,
    accuracy_score
)

def calculate_qwk(y_true, y_pred):
    """计算二次加权 Kappa (QWK)"""
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')

def calculate_medical_metrics(y_true, y_pred, y_probs=None, num_classes=5):
    """
    计算多分类医学图像评价指标
    包含整体准确率 (Accuracy) 以及针对长尾分布的宏平均 (Macro Average) 指标
    """
    labels = list(range(num_classes))
    
    # 1. 整体准确率 (Accuracy)
    accuracy = accuracy_score(y_true, y_pred)
    
    # 2. 召回率 (敏感度 Sensitivity) 和 F1 Score (Macro)
    recall = recall_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)
    
    # 3. 特异性 Specificity (多分类下利用混淆矩阵计算 Macro 特异性)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    specificities = []
    for i in range(num_classes):
        tn = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]
        fp = np.sum(cm[:, i]) - cm[i, i]
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificities.append(spec)
    specificity = np.mean(specificities)
    
    # 4. 鲁棒的 AUC 曲线下面积计算 (OvR 多分类策略)
    auc = float('nan')
    if y_probs is not None:
        y_probs_arr = np.array(y_probs)
        y_true_arr = np.array(y_true)
        aucs = []
        
        # 遍历每一个类别，进行 OvR (One-vs-Rest) 计算
        for i in range(num_classes):
            # 只有当该类别在真实标签中同时存在"正例"和"负例"时，才计算 AUC
            if len(np.unique(y_true_arr == i)) == 2:
                class_auc = roc_auc_score((y_true_arr == i).astype(int), y_probs_arr[:, i])
                aucs.append(class_auc)
                
        # 取有效计算出的类别的 Macro 平均值
        if len(aucs) > 0:
            auc = np.mean(aucs)
            
    # 将 accuracy 作为第五个返回值
    return recall, specificity, f1, auc, accuracy

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
