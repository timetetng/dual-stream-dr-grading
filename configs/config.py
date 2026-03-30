# configs/config.py

class BaseConfig:
    """全局基础超参数配置"""
    SEED = 42
    NUM_EPOCHS = 50
    NUM_WORKERS = 4 
    BATCH_SIZE = 8 
    ACCUMULATION_STEPS = 4 
    PATIENCE = 12 
    
    CSV_PATH = 'data/raw/train.csv'
    IMG_DIR = 'data/processed/train_images'
    OUTPUT_DIR = 'outputs'
    # 
    # LR_PRETRAINED = 4.29e-5 
    # LR_NEW = 2.3e-4 
    # WEIGHT_DECAY = 8.43e-4 
    # ORDINAL_ALPHA = 0.361 
    # INIT_RADIUS = 40.0 
    LR_PRETRAINED = 4.6285931526012585e-05
    LR_NEW =9.381271720378282e-05
    WEIGHT_DECAY = 0.00025610936319446113
    ORDINAL_ALPHA = 0.46040511768600134
    INIT_RADIUS = 90.0 


# 消融实验矩阵 (保持原样即可)
EXPERIMENTS = {
    "Baseline (Spatial Only)": {
        "enabled": True,
        "use_freq": False, 
        "fusion_type": "concat", 
        "use_ordinal": False,
        "freq_type": "magnitude"
    },
    "+ Freq (Old: Magnitude)": {
        "enabled":True,
        "use_freq": True,  
        "fusion_type": "gated",  
        "use_ordinal": False,
        "freq_type": "magnitude" 
    },
    "+ Freq (New: Math Prior)": {
        "enabled":True,
        "use_freq": True,  
        "fusion_type": "gated",  
        "use_ordinal": False,
        "freq_type": "math_prior" 
    },
    "+ Freq (Magnitude + Ordinal loss)": {
        "enabled":True,
        "use_freq": True,  
        "fusion_type": "gated",  
        "use_ordinal": True,
        "freq_type": "magnitude" 
    },
    "Final (Math Prior + Ordinal Loss)": {
        "enabled": True, 
        "use_freq": True,  
        "fusion_type": "gated",  
        "use_ordinal": True,
        "freq_type": "math_prior"
    }
}
