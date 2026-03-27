# configs/config.py

class BaseConfig:
    """全局基础超参数配置"""
    SEED = 42
    NUM_EPOCHS = 20 # 总训练轮次
    NUM_WORKERS = 4 # CPU读取线程数
    BATCH_SIZE = 8 # 单次样本数
    ACCUMULATION_STEPS = 4 # 梯度累加步数，等效批次 = 单次样本数 * 梯度累加步数
    PATIENCE = 12 # 早停阈值
    
    CSV_PATH = 'data/raw/train.csv'
    IMG_DIR = 'data/processed/train_images'
    OUTPUT_DIR = 'outputs'
    
    LR_PRETRAINED = 4.29e-5 # 迁移模型学习率
    LR_NEW = 2.3e-4 # 新层学习率
    WEIGHT_DECAY = 8.43e-4 # 遗忘惩罚
    ORDINAL_ALPHA = 0.361 # 序数回归 SmoothL1 正则化惩罚系数

# 消融实验矩阵
EXPERIMENTS = {
    # 对照组 1：只有传统的空间图（基线）
    "Baseline (Spatial Only)": {
        "enabled": True,
        "use_freq": False, 
        "fusion_type": "concat", 
        "use_ordinal": False,
        "freq_type": "magnitude"
    },
    # 对照组 2：加入普通幅度谱分支
    "+ Freq (Old: Magnitude)": {
        "enabled": True,
        "use_freq": True,  
        "fusion_type": "gated",  
        "use_ordinal": False,
        "freq_type": "magnitude" # 调用旧的 FrequencyBranch
    },
    # 实验组 1：换用数学先验病灶感知分支
    "+ Freq (New: Math Prior)": {
        "enabled": True,
        "use_freq": True,  
        "fusion_type": "gated",  
        "use_ordinal": False,
        "freq_type": "math_prior" # 调用新的 HighFreqLesionBranch
    },
    # 对照组 3 ：原来的序数回归
    "+ Freq (Magnitude + Ordinal loss)": {
        "enabled": True,
        "use_freq": True,  
        "fusion_type": "gated",  
        "use_ordinal": True,
        "freq_type": "magnitude" # 调用旧的 FrequencyBranch
    },
    
    # 实验组 2：完全体（数学先验 + 序数回归联合损失）
    "Final (Math Prior + Ordinal Loss)": {
        "enabled": True, 
        "use_freq": True,  
        "fusion_type": "gated",  
        "use_ordinal": True,
        "freq_type": "math_prior"
    }
}
