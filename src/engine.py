import torch
from rich.progress import Progress, TaskID

def train_one_epoch(
    model, 
    dataloader, 
    criterion, 
    optimizer, 
    scaler, 
    device, 
    accumulation_steps=4,
    progress: Progress = None,
    task_id: TaskID = None,
    parent_advances: list = None
):
    """
    包含 AMP (自动混合精度) 和梯度累加的完整单轮训练函数
    加入了 parent_advances 以支持多级进度条的平滑联动和精确 ETA 计算
    """
    model.train()
    running_loss = 0.0
    
    optimizer.zero_grad()
    
    for i, (images, labels) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        with torch.amp.autocast('cuda'):
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss = loss / accumulation_steps
        
        scaler.scale(loss).backward()
        
        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(dataloader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        real_loss = loss.item() * accumulation_steps
        running_loss += real_loss
        
        if progress is not None:
            # 1. 推进最底层的 Batch 进度条
            if task_id is not None:
                progress.update(
                    task_id, 
                    advance=1, 
                    description=f"[cyan]训练中... Loss: {real_loss:.4f}[/cyan]"
                )
            # 2. 核心：通过极小浮点数步长，平滑推进外层的 Epoch 和 总体进度条
            if parent_advances is not None:
                for pid, amt in parent_advances:
                    progress.advance(pid, advance=amt)
            
    return running_loss / len(dataloader)


def evaluate(
    model, 
    dataloader, 
    criterion, 
    device, 
    metric_fn,
    progress: Progress = None,
    task_id: TaskID = None,
    parent_advances: list = None
):
    """
    完整的验证/测试函数
    加入了 parent_advances 以支持多级进度条的平滑联动和精确 ETA 计算
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []  
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
                
            running_loss += loss.item()
            
            probs = torch.softmax(outputs, dim=1)
            all_probs.extend(probs.cpu().numpy()) 
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            if progress is not None:
                # 1. 推进最底层的 Batch 进度条
                if task_id is not None:
                    progress.update(
                        task_id, 
                        advance=1, 
                        description=f"[magenta]验证中... Loss: {loss.item():.4f}[/magenta]"
                    )
                # 2. 核心：通过极小浮点数步长，平滑推进外层的 Epoch 和 总体进度条
                if parent_advances is not None:
                    for pid, amt in parent_advances:
                        progress.advance(pid, advance=amt)
                
    epoch_loss = running_loss / len(dataloader)
    score = metric_fn(all_labels, all_preds)
    
    return epoch_loss, score, all_labels, all_preds, all_probs
