# configs/config.py
import os

class BaseConfig:
    """全局基础超参数配置"""
    # 随机种子与训练参数
    SEED = 42
    NUM_EPOCHS = 2
    BATCH_SIZE = 8
    NUM_WORKERS = 4
    ACCUMULATION_STEPS = 4
    PATIENCE = 12
    
    # 路径配置
    CSV_PATH = 'data/raw/train.csv'
    IMG_DIR = 'data/processed/train_images'
    OUTPUT_DIR = 'outputs'
    
    # 优化器与学习率
    LR_PRETRAINED = 4.57e-5
    LR_NEW = 1.92e-5
    WEIGHT_DECAY = 2.96e-5
    
    # 损失函数超参
    ORDINAL_ALPHA = 0.4448

# 消融实验矩阵配置
# 你可以通过修改 "enabled" 的 True/False 来自由控制是否运行该组实验
EXPERIMENTS = {
    "Baseline (Spatial Only)": {
        "enabled": True,
        "use_freq": False, 
        "fusion_type": "concat", 
        "use_ordinal": False
    },
    "+ Freq (Concat Fusion)": {
        "enabled": True,
        "use_freq": True,  
        "fusion_type": "concat", 
        "use_ordinal": False
    },
    "+ Freq (Gated Fusion)": {
        "enabled": True,
        "use_freq": True,  
        "fusion_type": "gated",  
        "use_ordinal": False
    },
    "+ Freq (Gated) + Ordinal": {
        "enabled": True, # 如果不想跑这组，改成 False 即可
        "use_freq": True,  
        "fusion_type": "gated",  
        "use_ordinal": True
    }
}
