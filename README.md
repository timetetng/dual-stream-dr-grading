├── data/                      # 统一存放数据
│   ├── raw/                   # 原始解压数据 
│   │   ├── train_images/
│   │   ├── test_images/
│   │   ├── train.csv
│   │   └── test.csv
│   └── processed/             # 存放预处理后的图像，加速读取
├── src/                       # 核心代码源码目录
│   ├── dataset.py             # 数据加载与预处理模块
│   ├── models/                # 模型定义模块
│   │   ├── baseline.py        # 单流基线网络 (ResNet/EfficientNet)
│   │   ├── branches.py        # 空域分支与频域分支
│   │   ├── fusion.py          # 门控融合模块
│   │   └── loss.py            # 序数回归损失与多分类损失
│   ├── engine.py              # 训练与验证逻辑 (标准 train_one_epoch, evaluate_one_epoch)
│   └── utils.py               # 工具函数 (QWK计算、日志记录、混淆矩阵可视化)
├── configs/                   # 配置文件目录
│   └── config.yaml    # 存放超参数 (尺寸、学习率、Batch Size、随机种子)
├── outputs/                   # 实验输出目录
│   ├── logs/                  # 日志或文本日志
│   ├── weights/               # 保存的 .pth 模型权重
│   └── reports/               # 输出结果报告 (混淆矩阵图、可视化图等)
├── main.py                    # 主入口文件，解析配置并启动训练/测试
├── requirements.txt           # 依赖包列表
└── README.md
