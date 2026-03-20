import torch
from tqdm import tqdm

def train_one_epoch(model, dataloader, criterion, optimizer, device, accumulation_steps=4):
    """
    包含 AMP (自动混合精度) 和梯度累加的完整单轮训练函数 (已修复 PyTorch 2.x API 警告)
    """
    model.train()
    running_loss = 0.0
    
    scaler = torch.amp.GradScaler('cuda')
    optimizer.zero_grad()
    
    pbar = tqdm(dataloader, desc="Training")
    for i, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)
        
        with torch.amp.autocast('cuda'):
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss = loss / accumulation_steps
        
        scaler.scale(loss).backward()
        
        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(dataloader):
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
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating")
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
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
