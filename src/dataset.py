import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import StratifiedKFold

class APTOSDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None, is_train=True):
        self.data = dataframe.reset_index(drop=True)
        self.img_dir = img_dir 
        self.transform = transform
        self.is_train = is_train

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_id = self.data.iloc[idx]['id_code']
        img_path = os.path.join(self.img_dir, f"{img_id}.png")
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        if self.is_train:
            label = int(self.data.iloc[idx]['diagnosis'])
            return image, torch.tensor(label, dtype=torch.long)
        else:
            return image, img_id

def get_dataloaders(csv_path, img_dir, batch_size=16, num_workers=8, n_splits=5, fold_idx=0):
    """
    获取数据加载器
    """
    df = pd.read_csv(csv_path)
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    folds = list(skf.split(df['id_code'], df['diagnosis']))
    train_idx, val_idx = folds[fold_idx]
    
    train_df = df.iloc[train_idx].copy()
    val_df = df.iloc[val_idx].copy()
    
    # 训练集
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 验证集
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = APTOSDataset(train_df, img_dir, transform=train_transform, is_train=True)
    val_dataset = APTOSDataset(val_df, img_dir, transform=val_transform, is_train=True)
    
    # 类别不平衡处理
    class_counts = train_df['diagnosis'].value_counts().sort_index().values
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[label] for label in train_df['diagnosis'].values]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    
    # 加入 persistent_workers=True 防止每个 epoch 重新创建进程造成卡顿
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        sampler=sampler, 
        num_workers=num_workers, 
        drop_last=True, 
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    return train_loader, val_loader
