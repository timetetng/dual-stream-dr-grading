import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
