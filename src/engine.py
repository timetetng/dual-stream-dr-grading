import torch
from tqdm import tqdm

def train_one_epoch(model, dataloader, criterion, optimizer, device, accumulation_steps=4):
    """
    包含 AMP (自动混合精度) 和梯度累加的完整单轮训练函数
    添加了 non_blocking=True 提升数据传输与计算的并行度
    加入梯度裁剪以防止序数回归带来的梯度爆炸与震荡
    """
    model.train()
    running_loss = 0.0
    
    scaler = torch.amp.GradScaler('cuda')
    optimizer.zero_grad()
    
    pbar = tqdm(dataloader, desc="Training")
    for i, (images, labels) in enumerate(pbar):
        # 加上 non_blocking=True 配合 pin_memory 异步传输
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        with torch.amp.autocast('cuda'):
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss = loss / accumulation_steps
        
        scaler.scale(loss).backward()
        
        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(dataloader):
            # --- 新增：AMP 下的安全梯度裁剪 ---
            # 1. 先将梯度取消缩放，恢复到真实大小
            scaler.unscale_(optimizer)
            # 2. 将所有梯度的最大范数限制为 1.0，防止梯度爆炸引发模型震荡
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # --------------------------------
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        # 因为引入了联合损失，loss 的数值会有变化
        real_loss = loss.item() * accumulation_steps
        running_loss += real_loss
        pbar.set_postfix({'loss': f"{real_loss:.4f}"})
        
    return running_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device, metric_fn):
    """
    完整的验证/测试函数
    添加了 non_blocking=True
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating")
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
                
            running_loss += loss.item()
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    epoch_loss = running_loss / len(dataloader)
    score = metric_fn(all_labels, all_preds)
    
    return epoch_loss, score, all_labels, all_preds
