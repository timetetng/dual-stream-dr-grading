# src/trainer.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from rich.console import Console

from src.models.fusion import DualStreamNet
from src.models.loss import JointOrdinalLoss
from src.engine import train_one_epoch, evaluate
from src.utils import calculate_qwk, calculate_medical_metrics, plot_confusion_matrix, plot_training_curves
from configs.config import BaseConfig

console = Console()

class AblationTrainer:
    def __init__(self, exp_name, config, train_loader, val_loader, test_loader, device, progress, epoch_task, overall_task):
        self.exp_name = exp_name
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.progress = progress
        self.epoch_task = epoch_task
        self.overall_task = overall_task
        
        self.output_dir = BaseConfig.OUTPUT_DIR
        self.num_epochs = BaseConfig.NUM_EPOCHS
        self.accumulation_steps = BaseConfig.ACCUMULATION_STEPS
        self.patience = BaseConfig.PATIENCE
        
    def run(self):
        torch.cuda.empty_cache()
        
        # 1. 初始化模型
        model = DualStreamNet(
            num_classes=5, 
            embed_dim=512, 
            use_freq=self.config['use_freq'], 
            fusion_type=self.config['fusion_type'],
            freq_type=self.config.get('freq_type', 'magnitude')
        ).to(self.device)
        
        # 2. 初始化损失函数
        if self.config['use_ordinal']:
            criterion = JointOrdinalLoss(alpha=BaseConfig.ORDINAL_ALPHA)
        else:
            criterion = nn.CrossEntropyLoss()
            
        # 3. 配置优化器参数分组
        pretrained_params, new_params = [], []
        for name, param in model.named_parameters():
            if 'spatial_branch' in name:
                pretrained_params.append(param)
            else:
                new_params.append(param)
                
        optimizer = optim.Adam([
            {'params': pretrained_params, 'lr': BaseConfig.LR_PRETRAINED}, 
            {'params': new_params, 'lr': BaseConfig.LR_NEW}
        ], weight_decay=BaseConfig.WEIGHT_DECAY) 

        scaler = torch.amp.GradScaler('cuda')
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.num_epochs)
        
        best_val_qwk = 0.0
        epochs_no_improve = 0
        
        os.makedirs(f"{self.output_dir}/weights", exist_ok=True)
        history = {'train_loss': [], 'val_loss': [], 'val_qwk': []}
        
        weight_filename = "best_dual_stream_ordinal.pth" if self.config['use_ordinal'] else f"best_{self.exp_name.replace(' ', '_')[:10]}.pth"
        weights_path = os.path.join(self.output_dir, "weights", weight_filename)

        # ================= 核心进度条联动计算逻辑 =================
        steps_per_epoch = len(self.train_loader) + len(self.val_loader)
        advance_epoch_step = 1.0 / steps_per_epoch
        advance_overall_step = 1.0 / (self.num_epochs * steps_per_epoch)
        
        parent_advances = [
            (self.epoch_task, advance_epoch_step),
            (self.overall_task, advance_overall_step)
        ]
        # ==========================================================
        
        # 4. 训练循环
        for epoch in range(self.num_epochs):
            train_task = self.progress.add_task(f"[cyan]  ├─ Train (Ep {epoch+1})", total=len(self.train_loader))
            train_loss = train_one_epoch(
                model, self.train_loader, criterion, optimizer, scaler, self.device, 
                self.accumulation_steps, self.progress, train_task, parent_advances=parent_advances
            )
            self.progress.remove_task(train_task)
            
            val_task = self.progress.add_task(f"[magenta]  ├─ Val (Ep {epoch+1})", total=len(self.val_loader))
            val_loss, val_qwk, _, _, _ = evaluate(
                model, self.val_loader, criterion, self.device, calculate_qwk, 
                self.progress, val_task, parent_advances=parent_advances
            )
            self.progress.remove_task(val_task)
            
            scheduler.step()
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_qwk'].append(val_qwk)
            
            if val_qwk > best_val_qwk:
                best_val_qwk = val_qwk
                epochs_no_improve = 0  
                torch.save(model.state_dict(), weights_path)
                self.progress.update(self.epoch_task, description=f"[bold green]Running: {self.exp_name} | Best Val QWK: {best_val_qwk:.4f} (Saved!)[/]")
            else:
                epochs_no_improve += 1
                self.progress.update(self.epoch_task, description=f"[bold green]Running: {self.exp_name} | Best Val: {best_val_qwk:.4f} (No imp: {epochs_no_improve} ep)[/]")
                
            if epochs_no_improve >= self.patience:
                console.print(f"[bold red]Early stopping triggered for {self.exp_name} at epoch {epoch+1}.[/bold red]")
                remaining_epochs = self.num_epochs - (epoch + 1)
                self.progress.advance(self.overall_task, advance=remaining_epochs * steps_per_epoch * advance_overall_step)
                break
                
        # 5. 独立测试集评估
        test_task = self.progress.add_task(f"[yellow]  └─ Independent Test Eval...", total=len(self.test_loader))
        model.load_state_dict(torch.load(weights_path))
        _, test_qwk, test_labels, test_preds, test_probs = evaluate(
            model, self.test_loader, criterion, self.device, calculate_qwk, self.progress, test_task
        )
        self.progress.remove_task(test_task)
        
        # 接收新加入的准确率
        test_recall, test_spec, test_f1, test_auc, test_acc = calculate_medical_metrics(test_labels, test_preds, test_probs)
        
        # 6. 保存报告
        safe_exp_name = self.exp_name.replace(' ', '_').replace('+', '').replace('(', '').replace(')', '')
        plot_confusion_matrix(test_labels, test_preds, f"{self.output_dir}/reports/ablation_{safe_exp_name}_test_cm.png")
        pd.DataFrame(history).to_csv(f"{self.output_dir}/reports/history_{safe_exp_name}.csv", index=False)
        plot_training_curves(history, f"{self.output_dir}/reports/curves_{safe_exp_name}.png")
                
        del model, optimizer, criterion
        torch.cuda.empty_cache()
        
        return {
            "Model Variant": self.exp_name, 
            "Accuracy": test_acc, 
            "Val QWK": best_val_qwk, 
            "Test QWK": test_qwk,
            "Recall": test_recall,
            "Specificity": test_spec,
            "F1 Score": test_f1,
            "AUC": test_auc
        }
