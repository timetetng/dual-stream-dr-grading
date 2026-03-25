# main.py
import os
import torch
import pandas as pd
from rich.console import Console
from rich.table import Table

from src.dataset import get_dataloaders
from src.ui import create_progress_bar
from src.trainer import AblationTrainer
from configs.config import BaseConfig, EXPERIMENTS

console = Console()

def main():
    # 1. 基础环境设置
    os.makedirs(f"{BaseConfig.OUTPUT_DIR}/reports", exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    console.print(f"[bold blue]Using device: {device}[/bold blue]")
    
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    
    # 2. 准备数据
    train_loader, val_loader, test_loader = get_dataloaders(
        BaseConfig.CSV_PATH, 
        BaseConfig.IMG_DIR, 
        batch_size=BaseConfig.BATCH_SIZE, 
        num_workers=BaseConfig.NUM_WORKERS
    )
    
    # 3. 过滤出开启了的消融实验
    active_experiments = {name: cfg for name, cfg in EXPERIMENTS.items() if cfg.get("enabled", True)}
    
    if not active_experiments:
        console.print("[bold red]没有在 configs/config.py 中开启任何实验！[/bold red]")
        return
        
    results = []
    
    # 4. 运行实验组合
    with create_progress_bar() as progress:
        overall_task = progress.add_task("[bold yellow]Overall Ablation Progress...", total=len(active_experiments))
        
        for exp_idx, (exp_name, config) in enumerate(active_experiments.items()):
            epoch_task = progress.add_task(f"[bold green]Running: {exp_name}", total=BaseConfig.NUM_EPOCHS)
            
            # 初始化 Trainer
            trainer = AblationTrainer(
                exp_name=exp_name,
                config=config,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                device=device,
                progress=progress,
                epoch_task=epoch_task,
                overall_task=overall_task
            )
            
            # 执行单组实验
            exp_result = trainer.run()
            results.append(exp_result)
            
            # 实验结束后隐藏子进度条并推进总进度
            progress.update(epoch_task, visible=False)
            progress.update(overall_task, completed=exp_idx + 1)
            
    # 5. 结果汇总与展示
    df_results = pd.DataFrame(results)
    csv_save_path = f"{BaseConfig.OUTPUT_DIR}/reports/ablation_results.csv"
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
