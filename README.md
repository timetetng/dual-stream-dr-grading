# 基于双流网络和序数回归的糖尿病视网膜分级研究

本项目旨在对糖尿病视网膜病变（Diabetic Retinopathy, DR）进行 0-4 级的自动化分级。结合了空间域与频率域（FFT）的双流网络架构，并引入了改进的序数回归联合损失函数（Joint Ordinal Loss），以提升分类的准确率并严厉惩罚跨级误判。

## 📁 项目目录结构

```text
.
├── configs/                   # 配置文件目录
│   └── config.py              # 全局基础超参数与消融实验矩阵配置
├── data/                      # 数据集统一存放路径
│   ├── aptos2019-blindness-detection.zip # 原始 Kaggle 数据集压缩包
│   ├── raw/                   # 原始解压数据 (包含 train.csv 与原始图像)
│   └── processed/             # 存放离线预处理后的图像，加速训练读取
├── src/                       # 核心模块源码目录
│   ├── models/                # 模型网络定义模块
│   │   ├── baseline.py        # 单流基线网络 (基于 ResNet50)
│   │   ├── branches.py        # 特征提取分支 (空域分支与基于 FFT 的频域分支)
│   │   ├── fusion.py          # 多模态特征融合模块
│   │   └── loss.py            # 改进的序数回归联合分类损失函数 (JointOrdinalLoss)
│   ├── dataset.py             # 数据加载、划分与在线数据增强模块 (Albumentations)
│   ├── engine.py              # 底层训练与验证逻辑 (支持 AMP 自动混合精度与梯度累加)
│   ├── trainer.py             # 训练器封装
│   ├── ui.py                  # 终端界面与进度条组件 (基于 Rich)
│   └── utils.py               # 指标计算 (QWK, 敏感度, 特异性等) 与可视化工具函数
├── outputs/                   # 实验结果输出目录
│   ├── logs/                  # 训练过程文本日志
│   ├── reports/               # 实验分析报告 (消融实验结果 CSV、混淆矩阵、训练曲线图等)
│   └── weights/               # 训练保存的最佳模型权重 (.pth)
├── main.py                    # 核心训练主入口，负责读取配置并启动一系列消融实验
├── preprocess.py              # 离线数据预处理脚本 (执行正方形裁剪、圆形 ROI 提取与 CLAHE 对比度增强)
├── tune.py                    # 超参数搜索与自动调优脚本
├── visualize.py               # 模型特征图分析与可视化脚本 (如 Grad-CAM 等)
└── README.md                  # 项目说明文档
```

## 🚀 跨平台环境复现指南

本项目采用 **Conda + Pip 混合依赖管理**，以确保在不同操作系统（Linux/Windows/macOS）上均能完美处理包含 CUDA 在内的底层硬件加速依赖。

### 1. 准备环境配置文件
请确保项目根目录下存在 `environment.yml` 文件。如果没有，请创建并写入以下内容：

```yaml
name: dr_project
channels:
  - pytorch
  - nvidia
  - conda-forge
  - defaults
dependencies:
  # 1. 核心 Python 环境
  - python=3.10
  
  # 2. 深度学习框架
  - pytorch=2.6.0
  - torchvision=0.21.0
  - pytorch-cuda=12.4
  
  # 3. 基础科学计算与工具库
  - numpy=2.4.3
  - pandas=3.0.1
  - scikit-learn=1.8.0
  - matplotlib=3.10.8
  - seaborn=0.13.2
  - pillow=12.1.1
  - optuna=4.8.0
  - rich=14.3.3
  - tqdm=4.67.3
  
  # 4. Pip 专属扩展包
  - pip:
    - albumentations==2.0.8
    - grad-cam==1.5.5
    #- opencv-python-headless==4.13.0.92  # 服务器/无头环境推荐使用 headless，避免 GUI 库冲突
    - opencv-python==4.13.0.92
```

### 2. 创建并激活虚拟环境
在终端中执行以下命令，Conda 会自动配置好底层依赖并调用 Pip 安装扩展包：

```bash
# 创建虚拟环境
conda env create -f environment.yml

# 激活环境
conda activate dr_project
```
*(注：如果后续 `environment.yml` 发生变更，可使用 `conda env update -f environment.yml --prune` 更新环境。)*

## 🏃‍♂️ 运行流程

### 1. 数据准备与预处理
1. 将下载的 APTOS 2019 数据集解压，确保 `train.csv` 位于 `data/raw/` 目录下，原始图像位于 `data/raw/train_images/` 目录下。
```bash
kaggle competitions download -c aptos2019-blindness-detection

unzip aptos2019-blindness-detection.zip

mv *.csv data/raw/
mv train_images data/raw/
mv test_imgaes data/raw/
```

2. 运行离线预处理脚本。该脚本将执行正方形裁剪、圆形 ROI 提取与 CLAHE 对比度增强，并将处理后的图像保存至 `data/processed/train_images/` 目录：
   ```bash
   python preprocess.py
   ```

### 2. 配置实验矩阵
项目的超参数和消融实验开关统一由 `configs/config.py` 管理。
* **基础参数**：你可以在 `BaseConfig` 类中修改 `BATCH_SIZE`、`NUM_EPOCHS`、各类学习率等。
* **实验控制**：在 `EXPERIMENTS` 字典中，通过修改 `"enabled": True/False` 可以自由控制本次运行需要跑哪些网络变体（如：单流基线、拼接融合双流、门控融合双流、加入序数回归等）。

### 3. 一键启动训练与评估
运行主程序，代码将自动解析配置、初始化数据加载器，并按照启用的消融实验矩阵依次进行训练与独立测试集的评估：
```bash
python main.py
```

## 📊 实验结果查阅

每次运行结束后，训练器会自动汇总数据并生成报告：
1. **终端输出**：利用 Rich 库渲染格式化的消融实验对比表格，包含 `Val QWK`、`Test QWK`、`Recall`、`Specificity`、`F1 Score`、`AUC` 等核心多分类医学指标。
2. **可视化图表**：Loss 曲线与 QWK 得分趋势图、测试集的混淆矩阵图将自动保存在 `outputs/reports/` 目录下。
3. **模型权重**：各变体在验证集上取得最高 QWK 得分的模型权重（`.pth` 文件）将保存在 `outputs/weights/` 目录下。

