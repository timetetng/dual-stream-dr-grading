import os
import cv2
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold, train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2

class APTOSDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None, is_train=True):
        self.data = dataframe.reset_index(drop=True)
        self.img_dir = img_dir 
        self.transform = transform
        self.is_train = is_train

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_id = str(self.data.iloc[idx]['id_code'])
        img_path = os.path.join(self.img_dir, f"{img_id}.png")
        
        # 统一使用 cv2 读取，配合前置预处理管道，并转为 RGB
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"找不到图像文件: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 使用 albumentations 进行在线数据增强
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
            
        if self.is_train:
            label = int(self.data.iloc[idx]['diagnosis'])
            return image, torch.tensor(label, dtype=torch.long)
        else:
            return image, img_id

def get_transforms(phase='train'):
    if phase == 'train':
        return A.Compose([
            # 边缘填充模式：border_mode，填充颜色：fill=0 (纯黑)
            A.Affine(
                scale=(0.9, 1.1), 
                translate_percent=(-0.1, 0.1), 
                rotate=(-90, 90), 
                border_mode=cv2.BORDER_CONSTANT, 
                fill=0, 
                p=0.7
            ),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

def get_dataloaders(csv_path, img_dir, batch_size=16, num_workers=4, n_splits=5, fold_idx=0, test_size=0.15):
    df = pd.read_csv(csv_path)
    
    # 1. 优先切分出绝对不可见的独立测试集
    train_val_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df['diagnosis'], random_state=42
    )
    train_val_df = train_val_df.reset_index(drop=True)
    
    # 2. 在剩余数据上进行 K-Fold 划分
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    folds = list(skf.split(train_val_df['id_code'], train_val_df['diagnosis']))
    train_idx, val_idx = folds[fold_idx]
    
    train_df = train_val_df.iloc[train_idx].copy()
    val_df = train_val_df.iloc[val_idx].copy()
    
    train_dataset = APTOSDataset(train_df, img_dir, transform=get_transforms('train'), is_train=True)
    val_dataset = APTOSDataset(val_df, img_dir, transform=get_transforms('val'), is_train=True)
    test_dataset = APTOSDataset(test_df, img_dir, transform=get_transforms('val'), is_train=True)
    
    # 3. 绝对反比加权采样，以牺牲少量整体一致性为代价，最大化重症(PDR)的召回率
    class_counts = train_df['diagnosis'].value_counts().sort_index().values
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[label] for label in train_df['diagnosis'].values]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    
    # 4. 构建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, drop_last=True, pin_memory=True, persistent_workers=True if num_workers > 0 else False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=True if num_workers > 0 else False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=True if num_workers > 0 else False)
    
    return train_loader, val_loader, test_loader
