import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import recall_score
from rich.console import Console

# 复用现有项目的模块
from configs.config import BaseConfig
from src.dataset import APTOSDataset, get_transforms
from src.models.fusion import DualStreamNet
from src.engine import train_one_epoch, evaluate
from src.utils import calculate_qwk, calculate_medical_metrics

console = Console()

def get_custom_dataloaders(strategy):
    """
    根据给定的策略动态生成 DataLoader
    strategy 可选: 'none' (无权重), 'inverse' (绝对反比 1/N), 'sqrt' (平方根反比 1/sqrt(N))
    """
    df = pd.read_csv(BaseConfig.CSV_PATH)
    
    # 切分独立测试集
    train_val_df, test_df = train_test_split(
        df, test_size=0.15, stratify=df['diagnosis'], random_state=42
    )
    train_val_df = train_val_df.reset_index(drop=True)
    
    # K-Fold 划分 (这里固定 fold 0 用于演示)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, val_idx = list(skf.split(train_val_df['id_code'], train_val_df['diagnosis']))[0]
    
    train_df = train_val_df.iloc[train_idx].copy()
    val_df = train_val_df.iloc[val_idx].copy()
    
    train_dataset = APTOSDataset(train_df, BaseConfig.IMG_DIR, transform=get_transforms('train'), is_train=True)
    val_dataset = APTOSDataset(val_df, BaseConfig.IMG_DIR, transform=get_transforms('val'), is_train=True)
    test_dataset = APTOSDataset(test_df, BaseConfig.IMG_DIR, transform=get_transforms('val'), is_train=True)
    
    # ---------------- 核心消融逻辑：构建不同的采样器 ----------------
    class_counts = train_df['diagnosis'].value_counts().sort_index().values
    
    if strategy == 'none':
        sampler = None
        shuffle = True
    elif strategy == 'inverse':
        class_weights = 1.0 / class_counts
        sample_weights = [class_weights[label] for label in train_df['diagnosis'].values]
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        shuffle = False
    elif strategy == 'sqrt':
        class_weights = 1.0 / np.sqrt(class_counts)
        sample_weights = [class_weights[label] for label in train_df['diagnosis'].values]
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        shuffle = False
    else:
        raise ValueError("Unknown strategy")

    num_workers = BaseConfig.NUM_WORKERS
    train_loader = DataLoader(train_dataset, batch_size=BaseConfig.BATCH_SIZE, sampler=sampler, shuffle=shuffle, num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BaseConfig.BATCH_SIZE, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=BaseConfig.BATCH_SIZE, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader

def run_experiment(strategy_name, strategy_type, device):
    """运行单组消融实验"""
    console.print(f"\n[bold yellow]>>> 正在运行重采样消融实验: {strategy_name} <<<[/bold yellow]")
    
    train_loader, val_loader, test_loader = get_custom_dataloaders(strategy_type)
    
    # 为排除结构干扰，使用纯空间基线网络
    model = DualStreamNet(num_classes=5, embed_dim=512, use_freq=False).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=BaseConfig.LR_PRETRAINED, weight_decay=BaseConfig.WEIGHT_DECAY)
    scaler = torch.amp.GradScaler('cuda')
    
    best_val_qwk = 0.0
    best_model_state = None
    
    for epoch in range(BaseConfig.NUM_EPOCHS):
        # 训练
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, BaseConfig.ACCUMULATION_STEPS)
        # 验证
        val_loss, val_qwk, _, _, _ = evaluate(model, val_loader, criterion, device, calculate_qwk)
        
        console.print(f"Epoch {epoch+1}/{BaseConfig.NUM_EPOCHS} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val QWK: {val_qwk:.4f}")
        
        if val_qwk > best_val_qwk:
            best_val_qwk = val_qwk
            best_model_state = model.state_dict().copy()
            
    # 加载最佳模型进行独立测试集评估
    model.load_state_dict(best_model_state)
    _, test_qwk, test_labels, test_preds, test_probs = evaluate(model, test_loader, criterion, device, calculate_qwk)
    
    # 获取通用指标
    _, _, test_f1, _, _ = calculate_medical_metrics(test_labels, test_preds, test_probs)
    
    # 专门提取 PDR (4级) 的召回率
    class_recalls = recall_score(test_labels, test_preds, labels=[0, 1, 2, 3, 4], average=None, zero_division=0)
    pdr_recall = class_recalls[4]
    
    return {
        "Strategy": strategy_name,
        "Val QWK": best_val_qwk,
        "Test QWK": test_qwk,
        "Macro F1": test_f1,
        "PDR Recall": pdr_recall
    }

def plot_ablation_results(results):
    """根据真实实验数据生成学术风格柱状图"""
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']
    plt.rcParams['axes.linewidth'] = 1.0
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    strategies = [r['Strategy'] for r in results]
    val_qwks = [r['Val QWK'] for r in results]
    macro_f1s = [r['Macro F1'] for r in results]
    pdr_recalls = [r['PDR Recall'] for r in results]

    x = np.arange(len(strategies))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 6))

    b1 = ax.bar(x - width, val_qwks, width, label='Overall Val QWK', color='#4C72B0', edgecolor='black', linewidth=1.2)
    b2 = ax.bar(x, macro_f1s, width, label='Macro F1 Score', color='#55A868', edgecolor='black', linewidth=1.2)
    b3 = ax.bar(x + width, pdr_recalls, width, label='PDR (Grade 4) Recall', color='#C44E52', edgecolor='black', linewidth=1.2)

    ax.set_ylabel('Performance Metric Score', fontsize=12)
    ax.set_title('Ablation Study of Resampling Strategies (Empirical Results)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, fontsize=11)
    ax.set_ylim(0, 1.05)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper left', frameon=False, fontsize=11)

    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)

    add_labels(b1)
    add_labels(b2)
    add_labels(b3)

    os.makedirs('outputs/reports', exist_ok=True)
    save_path = 'outputs/reports/real_resampling_ablation.png'
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    console.print(f"\n[bold green]图表已成功生成并保存至: {save_path}[/bold green]")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True if device.type == 'cuda' else False
    
    # 定义需要测试的三组实验
    experiments = [
        ("Baseline (No Weight)", "none"),
        ("Inverse Weight (1/N)", "inverse"),
        ("Inverse Sqrt (1/sqrt(N))", "sqrt")
    ]
    
    results = []
    for name, strategy_type in experiments:
        res = run_experiment(name, strategy_type, device)
        results.append(res)
        
    # 保存结果并画图
    df_results = pd.DataFrame(results)
    csv_path = "outputs/reports/real_weight_ablation.csv"
    df_results.to_csv(csv_path, index=False)
    console.print(f"\n[dim]实验数据已保存至 {csv_path}[/dim]")
    
    plot_ablation_results(results)

if __name__ == '__main__':
    main()
