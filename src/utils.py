import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

def calculate_qwk(y_true, y_pred):
    """计算二次加权 Kappa (QWK)"""
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')

def plot_confusion_matrix(y_true, y_pred, save_path):
    """
    生成并保存混淆矩阵图像，用于分析跨级误判
    """
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
    
    # 绘制 Loss 曲线 (左轴)
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color='tab:red')
    ax1.plot(epochs, history['train_loss'], 'r--', label='Train Loss')
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    
    # 绘制 QWK 曲线 (右轴)
    ax2 = ax1.twinx()  
    ax2.set_ylabel('QWK Score', color='tab:blue')  
    ax2.plot(epochs, history['val_qwk'], 'b-', marker='o', label='Val QWK')
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    
    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')
    
    plt.title('Training and Validation Metrics')
    fig.tight_layout()  
    plt.savefig(save_path, dpi=300)
    plt.close()
