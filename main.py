import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from src.dataset import get_dataloaders
from src.models.fusion import DualStreamNet
from src.models.loss import JointOrdinalLoss
from src.engine import train_one_epoch, evaluate
from src.utils import calculate_qwk, plot_confusion_matrix

def run_experiment(exp_name, config, train_loader, val_loader, device, output_dir, num_epochs=3):
    """运行单组实验并返回最佳 QWK"""
    print(f"\n{'='*50}")
    print(f"Starting Experiment: {exp_name}")
    print(f"{'='*50}")
    
    model = DualStreamNet(
        num_classes=5, 
        embed_dim=512, 
        use_freq=config['use_freq'], 
        fusion_type=config['fusion_type']
    ).to(device)
    
    if config['use_ordinal']:
        criterion = JointOrdinalLoss(alpha=0.5)
    else:
        criterion = nn.CrossEntropyLoss()
        
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    # 引入文档中设计的余弦退火学习率策略
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    best_qwk = 0.0
    os.makedirs(f"{output_dir}/weights", exist_ok=True)
    
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_qwk, val_labels, val_preds = evaluate(model, val_loader, criterion, device, calculate_qwk)
        
        # 更新学习率
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{num_epochs} | Val Loss: {val_loss:.4f} | Val QWK: {val_qwk:.4f}")
        
        if val_qwk > best_qwk:
            best_qwk = val_qwk
            
            # 补充保存模型权重的逻辑
            weight_filename = "best_dual_stream_ordinal.pth" if config['use_ordinal'] else f"best_{exp_name.replace(' ', '_')[:10]}.pth"
            weights_path = os.path.join(output_dir, "weights", weight_filename)
            torch.save(model.state_dict(), weights_path)
            
            cm_path = f"{output_dir}/reports/ablation_{exp_name}_best_cm.png"
            plot_confusion_matrix(val_labels, val_preds, cm_path)
            
    print(f"Experiment {exp_name} completed. Best QWK: {best_qwk:.4f}")
    return best_qwk

def main():
    csv_path = 'data/raw/train.csv'
    img_dir = 'data/processed/train_images' 
    output_dir = 'outputs'
    os.makedirs(f"{output_dir}/reports", exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_loader, val_loader = get_dataloaders(csv_path, img_dir, batch_size=4, num_workers=4)
    
    experiments = {
        "Baseline (Spatial Only)": {
            "use_freq": False, "fusion_type": "concat", "use_ordinal": False
        },
        "+ Freq (Concat Fusion)": {
            "use_freq": True,  "fusion_type": "concat", "use_ordinal": False
        },
        "+ Freq (Gated Fusion)": {
            "use_freq": True,  "fusion_type": "gated",  "use_ordinal": False
        },
        "+ Freq (Gated) + Ordinal (Full Model)": {
            "use_freq": True,  "fusion_type": "gated",  "use_ordinal": True
        }
    }
    
    results = []
    
    for exp_name, config in experiments.items():
        best_qwk = run_experiment(exp_name, config, train_loader, val_loader, device, output_dir, num_epochs=3)
        results.append({"Model Variant": exp_name, "Best QWK": best_qwk})
        
    # 优先保存到 CSV，确保数据安全
    df_results = pd.DataFrame(results)
    csv_save_path = f"{output_dir}/reports/ablation_results.csv"
    df_results.to_csv(csv_save_path, index=False)
    print(f"\nResults securely saved to {csv_save_path}")
    
    # 尝试打印 Markdown 表格
    print("\n\n" + "="*50)
    print("🏆 Ablation Study Results 🏆")
    print("="*50)
    try:
        print(df_results.to_markdown(index=False))
    except ImportError:
        print("Please run 'pip install tabulate' to view the formatted Markdown table.")
        print(df_results)

if __name__ == '__main__':
    main()
