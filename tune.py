import os
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from optuna.trial import TrialState
from src.dataset import get_dataloaders
from src.models.fusion import DualStreamNet
from src.models.loss import JointOrdinalLoss
from src.engine import train_one_epoch, evaluate
from src.utils import calculate_qwk
from src.ui import create_progress_bar

# 全局配置
CSV_PATH = 'data/raw/train.csv'
IMG_DIR = 'data/processed/train_images'
NUM_EPOCHS_PER_TRIAL = 15  # 搜索参数时不需要跑满，跑 15-20 个 epoch 足以看出潜力
BATCH_SIZE = 8
ACCUMULATION_STEPS = 4
NUM_WORKERS = 4

def objective(trial):
    """Optuna 的核心目标函数，定义搜索空间和训练逻辑"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()

    # 1. 定义超参数的搜索空间
    lr_pretrained = trial.suggest_float("lr_pretrained", 1e-6, 5e-5, log=True)
    lr_new = trial.suggest_float("lr_new", 1e-5, 5e-4, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    alpha = trial.suggest_float("alpha", 0.05, 0.5)

    # 2. 初始化模型与组件
    model = DualStreamNet(
        num_classes=5, 
        embed_dim=512, 
        use_freq=True, 
        fusion_type='gated',
        freq_type='math_prior'
    ).to(device)

    criterion = JointOrdinalLoss(alpha=alpha)

    pretrained_params = []
    new_params = []
    for name, param in model.named_parameters():
        if 'spatial_branch' in name:
            pretrained_params.append(param)
        else:
            new_params.append(param)

    optimizer = optim.Adam([
        {'params': pretrained_params, 'lr': lr_pretrained}, 
        {'params': new_params, 'lr': lr_new}
    ], weight_decay=weight_decay)

    scaler = torch.amp.GradScaler('cuda')
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS_PER_TRIAL)

    # 3. 获取数据
    train_loader, val_loader, _ = get_dataloaders(
        CSV_PATH, IMG_DIR, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
    )

    best_qwk = 0.0

    # 计算用于平滑联动外层进度条的步长
    steps_per_epoch = len(train_loader) + len(val_loader)
    advance_epoch_step = 1.0 / steps_per_epoch

    # 4. 训练与剪枝循环 (包裹在进度条上下文中)
    with create_progress_bar() as progress:
        # 创建当前 Trial 的总进度条
        trial_task = progress.add_task(f"[bold yellow]Trial {trial.number} Running...", total=NUM_EPOCHS_PER_TRIAL)

        for epoch in range(NUM_EPOCHS_PER_TRIAL):
            # 训练阶段进度条
            train_task = progress.add_task(f"[cyan]  ├─ Train (Ep {epoch+1})", total=len(train_loader))
            train_loss = train_one_epoch(
                model=model, 
                dataloader=train_loader, 
                criterion=criterion, 
                optimizer=optimizer, 
                scaler=scaler, 
                device=device, 
                accumulation_steps=ACCUMULATION_STEPS,
                progress=progress,          # 传入进度条对象
                task_id=train_task,         # 传入当前底层任务 ID
                parent_advances=[(trial_task, advance_epoch_step)] # 联动外层进度条
            )
            progress.remove_task(train_task)
            
            # 验证阶段进度条
            val_task = progress.add_task(f"[magenta]  ├─ Val (Ep {epoch+1})", total=len(val_loader))
            val_loss, val_qwk, _, _, _ = evaluate(
                model=model, 
                dataloader=val_loader, 
                criterion=criterion, 
                device=device, 
                metric_fn=calculate_qwk,
                progress=progress,          # 传入进度条对象
                task_id=val_task,           # 传入当前底层任务 ID
                parent_advances=[(trial_task, advance_epoch_step)] # 联动外层进度条
            )
            progress.remove_task(val_task)
            
            scheduler.step()

            if val_qwk > best_qwk:
                best_qwk = val_qwk
                
            # 实时更新当前 Trial 最好成绩的描述
            progress.update(trial_task, description=f"[bold green]Trial {trial.number} | Best QWK: {best_qwk:.4f}[/bold green]")

            # 将当前的验证集指标报告给 Optuna
            trial.report(val_qwk, epoch)

            # 检查是否需要提前终止 (剪枝)
            if trial.should_prune():
                del model, optimizer, criterion, scaler
                torch.cuda.empty_cache()
                # 抛出异常前，进度条上下文会自动安全退出
                raise optuna.exceptions.TrialPruned()

    # 清理内存
    del model, optimizer, criterion, scaler
    torch.cuda.empty_cache()
    
    return best_qwk


def main():
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3, interval_steps=1)
    
    study = optuna.create_study(direction="maximize", pruner=pruner, study_name="dr_grading_optimization")
    
    print(f"\n{'='*50}")
    print("🚀 Starting Hyperparameter Optimization with Optuna")
    print(f"{'='*50}\n")
    
    # 开启 Optuna 原生的全局搜索进度条
    study.optimize(objective, n_trials=30, show_progress_bar=True)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print(f"\n{'='*50}")
    print("🏆 Optimization Finished!")
    print(f"  Number of finished trials: {len(study.trials)}")
    print(f"  Number of pruned trials: {len(pruned_trials)}")
    print(f"  Number of complete trials: {len(complete_trials)}")

    print(f"\n🌟 Best Trial Details:")
    trial = study.best_trial
    print(f"  Value (Best QWK): {trial.value:.4f}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
        
    best_params_df = pd.DataFrame([trial.params])
    os.makedirs('outputs/reports', exist_ok=True)
    best_params_df.to_csv('outputs/reports/best_hyperparameters.csv', index=False)
    print("\nBest parameters securely saved to outputs/reports/best_hyperparameters.csv")

if __name__ == '__main__':
    main()
