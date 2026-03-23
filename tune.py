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

# 全局配置，方便调整
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
    # 空域分支（预训练）的学习率，范围设小一些
    lr_pretrained = trial.suggest_float("lr_pretrained", 1e-6, 5e-5, log=True)
    # 频域与融合分支（随机初始化）的学习率
    lr_new = trial.suggest_float("lr_new", 1e-5, 5e-4, log=True)
    # 权重衰减 (L2正则化)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    # 序数回归的 Alpha 惩罚系数
    alpha = trial.suggest_float("alpha", 0.05, 0.5)

    # 2. 初始化模型与组件 (仅针对效果最好的 Full Model 进行搜参)
    model = DualStreamNet(
        num_classes=5, 
        embed_dim=512, 
        use_freq=True, 
        fusion_type='gated'
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

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS_PER_TRIAL)

    # 3. 获取数据 (如果数据集极大，这里可以考虑传入参数使用子集)
    train_loader, val_loader = get_dataloaders(
        CSV_PATH, IMG_DIR, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
    )

    best_qwk = 0.0

    # 4. 训练与剪枝循环
    for epoch in range(NUM_EPOCHS_PER_TRIAL):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, accumulation_steps=ACCUMULATION_STEPS)
        val_loss, val_qwk, _, _ = evaluate(model, val_loader, criterion, device, calculate_qwk)
        scheduler.step()

        if val_qwk > best_qwk:
            best_qwk = val_qwk

        # 将当前的验证集指标报告给 Optuna
        trial.report(val_qwk, epoch)

        # 核心：检查是否需要提前终止这个没希望的实验 (剪枝)
        if trial.should_prune():
            del model, optimizer, criterion
            torch.cuda.empty_cache()
            raise optuna.exceptions.TrialPruned()

    del model, optimizer, criterion
    torch.cuda.empty_cache()
    
    # 目标是最大化 QWK
    return best_qwk

def main():
    # 使用中值停止规则作为剪枝器：如果在指定 epoch 后的指标低于所有已完成实验在该 epoch 的中位数，则砍掉
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3, interval_steps=1)
    
    study = optuna.create_study(direction="maximize", pruner=pruner, study_name="dr_grading_optimization")
    
    print(f"\n{'='*50}")
    print("🚀 Starting Hyperparameter Optimization with Optuna")
    print(f"{'='*50}\n")
    
    # 执行 30 次试验寻找最佳参数 (可以挂机一晚上跑 50 次)
    study.optimize(objective, n_trials=50)

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
        
    # 将最优参数保存到文件备查
    best_params_df = pd.DataFrame([trial.params])
    os.makedirs('outputs/reports', exist_ok=True)
    best_params_df.to_csv('outputs/reports/best_hyperparameters.csv', index=False)
    print("\nBest parameters securely saved to outputs/reports/best_hyperparameters.csv")

if __name__ == '__main__':
    main()
