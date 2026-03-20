import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

class APTOSDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, is_train=True):
        """
        初始化数据集
        :param csv_file: 包含 id_code 和 diagnosis 的 DataFrame
        :param img_dir: 图像所在的文件夹路径
        :param transform: torchvision.transforms 数据增强管道
        :param is_train: 是否为训练模式
        """
        self.data = csv_file
        self.img_dir = img_dir
        self.transform = transform
        self.is_train = is_train

    def __len__(self):
        return len(self.data)

    def preprocess_image(self, img_path):
        """
        执行基础的图像清洗：读取、裁剪黑边、转换为 RGB
        """
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # TODO: 在此处加入裁剪黑边和 CLAHE 增强的具体逻辑
        return Image.fromarray(image)

    def __getitem__(self, idx):
        img_id = self.data.iloc[idx]['id_code']
        img_path = os.path.join(self.img_dir, f"{img_id}.png")
        
        image = self.preprocess_image(img_path)
        
        if self.transform:
            image = self.transform(image)
            
        if self.is_train:
            label = int(self.data.iloc[idx]['diagnosis'])
            return image, torch.tensor(label, dtype=torch.long)
        else:
            return image, img_id
