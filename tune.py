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

# \u5168\u5c40\u914d\u7f6e
CSV_PATH = 'data/raw/train.csv'
IMG_DIR = 'data/processed/train_images'
NUM_EPOCHS_PER_TRIAL = 15  
BATCH_SIZE = 8
ACCUMULATION_STEPS = 4
NUM_WORKERS = 4

def objective(trial):
    """Optuna \u7684\u6838\u5fc3\u76ee\u6807\u51fd\u6570\uff0c\u5b9a\u4e49\u641c\u7d22\u7a7a\u95f4\u548c\u8bad\u7ec3\u903b\u8f91"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()

    # 1. \u5b9a\u4e49\u8d85\u53c2\u6570\u7684\u641c\u7d22\u7a7a\u95f4
    lr_pretrained = trial.suggest_float("lr_pretrained", 1e-6, 5e-5, log=True)
    lr_new = trial.suggest_float("lr_new", 1e-5, 5e-4, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    alpha = trial.suggest_float("alpha", 0.05, 0.5)
    
    # \u3010\u65b0\u589e\u3011\uff1a\u5c06\u6ee4\u6ce2\u5668\u7684\u521d\u59cb\u622a\u65ad\u9891\u7387\u7eb3\u5165\u8d85\u53c2\u641c\u7d22
    init_radius = trial.suggest_float("init_radius", 5.0, 100.0)

    # 2. \u521d\u59cb\u5316\u6a21\u578b\u4e0e\u7ec4\u4ef6
    model = DualStreamNet(
        num_classes=5, 
        embed_dim=512,
        use_freq=True, 
        fusion_type='gated',
        freq_type='math_prior',
        init_radius=init_radius  # \u4f20\u9012\u641c\u7d22\u51fa\u7684\u521d\u59cb\u503c
    ).to(device)

    criterion = JointOrdinalLoss(alpha=alpha)

    # \u6a21\u578b\u53c2\u6570\u5206\u7ec4\uff08\u53ef\u5b66\u4e60\u7684 radius \u81ea\u52a8\u4f1a\u88ab\u5212\u5206\u5230 new_params \u91cc\uff0c\u4eab\u53d7 lr_new \u7684\u5b66\u4e60\u7387\uff09
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

    # 3. \u83b7\u53d6\u6570\u636e
    train_loader, val_loader, _ = get_dataloaders(
        CSV_PATH, IMG_DIR, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
    )

    best_qwk = 0.0
    steps_per_epoch = len(train_loader) + len(val_loader)
    advance_epoch_step = 1.0 / steps_per_epoch

    # 4. \u8bad\u7ec3\u4e0e\u526a\u679d\u5faa\u73af
    with create_progress_bar() as progress:
        trial_task = progress.add_task(f"[bold yellow]Trial {trial.number} Running...", total=NUM_EPOCHS_PER_TRIAL)

        for epoch in range(NUM_EPOCHS_PER_TRIAL):
            train_task = progress.add_task(f"[cyan]  \u251c\u2500 Train (Ep {epoch+1})", total=len(train_loader))
            train_loss = train_one_epoch(
                model=model, 
                dataloader=train_loader, 
                criterion=criterion, 
                optimizer=optimizer, 
                scaler=scaler, 
                device=device, 
                accumulation_steps=ACCUMULATION_STEPS,
                progress=progress,          
                task_id=train_task,         
                parent_advances=[(trial_task, advance_epoch_step)] 
            )
            progress.remove_task(train_task)
            
            val_task = progress.add_task(f"[magenta]  \u251c\u2500 Val (Ep {epoch+1})", total=len(val_loader))
            val_loss, val_qwk, _, _, _ = evaluate(
                model=model, 
                dataloader=val_loader, 
                criterion=criterion, 
                device=device, 
                metric_fn=calculate_qwk,
                progress=progress,          
                task_id=val_task,           
                parent_advances=[(trial_task, advance_epoch_step)] 
            )
            progress.remove_task(val_task)
            
            scheduler.step()

            if val_qwk > best_qwk:
                best_qwk = val_qwk
                
            progress.update(trial_task, description=f"[bold green]Trial {trial.number} | Best QWK: {best_qwk:.4f}[/bold green]")
            trial.report(val_qwk, epoch)

            if trial.should_prune():
                del model, optimizer, criterion, scaler
                torch.cuda.empty_cache()
                raise optuna.exceptions.TrialPruned()

    del model, optimizer, criterion, scaler
    torch.cuda.empty_cache()
    
    return best_qwk

def main():
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3, interval_steps=1)
    study = optuna.create_study(direction="maximize", pruner=pruner, study_name="dr_grading_optimization")
    
    print(f"\n{'='*50}")
    print("\U0001f680 Starting Hyperparameter Optimization with Optuna")
    print(f"{'='*50}\n")
    
    study.optimize(objective, n_trials=30, show_progress_bar=True)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print(f"\n{'='*50}")
    print("\U0001f3c6 Optimization Finished!")
    print(f"  Number of finished trials: {len(study.trials)}")
    print(f"  Number of pruned trials: {len(pruned_trials)}")
    print(f"  Number of complete trials: {len(complete_trials)}")

    print(f"\n\U0001f31f Best Trial Details:")
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

