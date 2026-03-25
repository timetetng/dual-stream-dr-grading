import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.console import Console
from rich.table import Table

from src.dataset import get_dataloaders
from src.models.fusion import DualStreamNet
from src.models.loss import JointOrdinalLoss
from src.engine import train_one_epoch, evaluate
from src.utils import calculate_qwk, calculate_medical_metrics, plot_confusion_matrix, plot_training_curves

console = Console()

def run_experiment(exp_name, config, train_loader, val_loader, test_loader, device, output_dir, progress, epoch_task, overall_task, num_epochs=30, accumulation_steps=4):
    torch.cuda.empty_cache()
    
    model = DualStreamNet(
        num_classes=5, 
        embed_dim=512, 
        use_freq=config['use_freq'], 
        fusion_type=config['fusion_type']
    ).to(device)
    
    if config['use_ordinal']:
        criterion = JointOrdinalLoss(alpha=0.4448)
    else:
        criterion = nn.CrossEntropyLoss()
        
    pretrained_params, new_params = [], []
    for name, param in model.named_parameters():
        if 'spatial_branch' in name:
            pretrained_params.append(param)
        else:
            new_params.append(param)
            
    optimizer = optim.Adam([
        {'params': pretrained_params, 'lr': 4.57e-5}, 
        {'params': new_params, 'lr': 1.92e-5}
    ], weight_decay=2.96e-5) 

    scaler = torch.amp.GradScaler('cuda')
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    best_val_qwk = 0.0
    patience = 12  
    epochs_no_improve = 0
    
    os.makedirs(f"{output_dir}/weights", exist_ok=True)
    history = {'train_loss': [], 'val_loss': [], 'val_qwk': []}
    
    weight_filename = "best_dual_stream_ordinal.pth" if config['use_ordinal'] else f"best_{exp_name.replace(' ', '_')[:10]}.pth"
    weights_path = os.path.join(output_dir, "weights", weight_filename)

    # ================= 核心计算逻辑 =================
    # 计算当前模型单个 Epoch 总计有多少个 Batch 需要处理（Train + Val）
    steps_per_epoch = len(train_loader) + len(val_loader)
    
    # 因为 epoch_task 的 total 是 num_epochs，所以单个 Batch 占 1 / steps_per_epoch
    advance_epoch_step = 1.0 / steps_per_epoch
    
    # 同理，overall_task 的 total 是 实验数量，单个 Batch 在整个实验中占的比重更小
    advance_overall_step = 1.0 / (num_epochs * steps_per_epoch)
    
    parent_advances = [
        (epoch_task, advance_epoch_step),
        (overall_task, advance_overall_step)
    ]
    # ===============================================
    
    for epoch in range(num_epochs):
        train_task = progress.add_task(f"[cyan]  ├─ Train (Ep {epoch+1})", total=len(train_loader))
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, 
            accumulation_steps, progress, train_task, parent_advances=parent_advances
        )
        progress.remove_task(train_task)
        
        val_task = progress.add_task(f"[magenta]  ├─ Val (Ep {epoch+1})", total=len(val_loader))
        val_loss, val_qwk, _, _, _ = evaluate(
            model, val_loader, criterion, device, calculate_qwk, 
            progress, val_task, parent_advances=parent_advances
        )
        progress.remove_task(val_task)
        
        scheduler.step()
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_qwk'].append(val_qwk)
        
        # 注意：此处不再调用 progress.update(epoch_task, advance=1)，因为内层已经平滑累加过了
        
        if val_qwk > best_val_qwk:
            best_val_qwk = val_qwk
            epochs_no_improve = 0  
            torch.save(model.state_dict(), weights_path)
            progress.update(epoch_task, description=f"[bold green]Running: {exp_name} | Best Val QWK: {best_val_qwk:.4f} (Saved!)[/]")
        else:
            epochs_no_improve += 1
            progress.update(epoch_task, description=f"[bold green]Running: {exp_name} | Best Val: {best_val_qwk:.4f} (No imp: {epochs_no_improve} ep)[/]")
            
        if epochs_no_improve >= patience:
            console.print(f"[bold red]Early stopping triggered for {exp_name} at epoch {epoch+1}.[/bold red]")
            # 补偿修正：如果早停，必须把剩下的“预计进度”直接加满，确保 Overall 进度条不会因为缺斤少两而错乱
            remaining_epochs = num_epochs - (epoch + 1)
            progress.advance(overall_task, advance=remaining_epochs * steps_per_epoch * advance_overall_step)
            break
            
    # --- 独立测试集评估 ---
    test_task = progress.add_task(f"[yellow]  └─ Independent Test Eval...", total=len(test_loader))
    model.load_state_dict(torch.load(weights_path))
    # 注意：测试集通常不计入 Overall 实验时间预估，所以不传 parent_advances
    _, test_qwk, test_labels, test_preds, test_probs = evaluate(model, test_loader, criterion, device, calculate_qwk, progress, test_task)
    progress.remove_task(test_task)
    
    test_recall, test_spec, test_f1, test_auc = calculate_medical_metrics(test_labels, test_preds, test_probs)
    
    safe_exp_name = exp_name.replace(' ', '_').replace('+', '').replace('(', '').replace(')', '')
    plot_confusion_matrix(test_labels, test_preds, f"{output_dir}/reports/ablation_{safe_exp_name}_test_cm.png")
    pd.DataFrame(history).to_csv(f"{output_dir}/reports/history_{safe_exp_name}.csv", index=False)
    plot_training_curves(history, f"{output_dir}/reports/curves_{safe_exp_name}.png")
            
    del model, optimizer, criterion
    torch.cuda.empty_cache()
    
    return best_val_qwk, test_qwk, test_recall, test_spec, test_f1, test_auc

def main():
    csv_path = 'data/raw/train.csv'
    img_dir = 'data/processed/train_images' 
    output_dir = 'outputs'
    os.makedirs(f"{output_dir}/reports", exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    console.print(f"[bold blue]Using device: {device}[/bold blue]")
    
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    
    train_loader, val_loader, test_loader = get_dataloaders(csv_path, img_dir, batch_size=8, num_workers=4)
    
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
        "+ Freq (Gated) + Ordinal": {
            "use_freq": True,  "fusion_type": "gated",  "use_ordinal": True
        }
    }
    
    results = []
    
    with Progress(
        TextColumn("[progress.description]{task.description}", justify="left"),
        BarColumn(bar_width=40),
        "[progress.percentage]{task.percentage:>3.1f}%",
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        expand=True
    ) as progress:
        
        overall_task = progress.add_task("[bold yellow]Overall Ablation Progress...", total=len(experiments))
        
        for exp_idx, (exp_name, config) in enumerate(experiments.items()):
            epoch_task = progress.add_task(f"[bold green]Running: {exp_name}", total=30)
            
            best_val, test_qwk, rec, spec, f1, auc = run_experiment(
                exp_name, config, train_loader, val_loader, test_loader, 
                device, output_dir, progress, epoch_task, overall_task, num_epochs=30, accumulation_steps=4
            )
            
            results.append({
                "Model Variant": exp_name, 
                "Val QWK": best_val, 
                "Test QWK": test_qwk,
                "Recall": rec,
                "Specificity": spec,
                "F1 Score": f1,
                "AUC": auc
            })
            
            # 实验结束后隐藏子进度条，并确保由于浮点数精度问题可能产生的微小差异被拉平
            progress.update(epoch_task, visible=False)
            progress.update(overall_task, completed=exp_idx + 1)
            
    df_results = pd.DataFrame(results)
    csv_save_path = f"{output_dir}/reports/ablation_results.csv"
    df_results.to_csv(csv_save_path, index=False)
    
    console.print("\n[bold yellow]🏆 Ablation Study Results 🏆[/bold yellow]")
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Model Variant", style="dim", width=35)
    table.add_column("Val QWK", justify="right")
    table.add_column("Test QWK", justify="right")
    table.add_column("Recall", justify="right")
    table.add_column("Spec", justify="right")
    table.add_column("F1", justify="right")
    table.add_column("AUC", justify="right")
    
    for row in results:
        table.add_row(
            row["Model Variant"],
            f"{row['Val QWK']:.4f}",
            f"{row['Test QWK']:.4f}",
            f"{row['Recall']:.4f}",
            f"{row['Specificity']:.4f}",
            f"{row['F1 Score']:.4f}",
            f"{row['AUC']:.4f}"
        )
        
    console.print(table)
    console.print(f"[dim]Results securely saved to {csv_save_path}[/dim]\n")

if __name__ == '__main__':
    main()
