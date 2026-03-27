from diagrams import Diagram, Cluster
from diagrams.custom import Custom
from diagrams.programming.framework import Fastapi # 仅用作占位图标示例

with Diagram("双流网络架构图 (Dual Stream Net)", show=False, direction="TB"):
    
    input_img = Custom("输入眼底图像", "./retina_icon.png") # 可以指定本地图标
    
    with Cluster("空域分支 (Spatial Branch)"):
        resnet = Custom("ResNet50 Backbone", "./cnn_icon.png")
        pool_s = Custom("AdaptiveConcatPool2d", "./pool_icon.png")
        proj_s = Custom("Projector", "./linear_icon.png")
        feat_s = Custom("空间特征 feat_spatial", "./tensor_icon.png")
        
        resnet >> pool_s >> proj_s >> feat_s
        
    with Cluster("频域分支 (HighFreq Lesion Branch)"):
        math_prior = Custom("apply_math_filter", "./fft_icon.png")
        cnn_f = Custom("Lightweight CNN", "./cnn_icon.png")
        pool_f = Custom("AdaptiveAvgPool2d", "./pool_icon.png")
        feat_f = Custom("频域特征 feat_freq", "./tensor_icon.png")
        
        math_prior >> cnn_f >> pool_f >> feat_f
        
    with Cluster("特征融合模块 (Dynamic Gated Fusion)"):
        concat = Custom("特征拼接 (Cat)", "./cat_icon.png")
        weighted_sum = Custom("加权求和", "./sum_icon.png")
        fused_feat = Custom("融合特征 fused_feat", "./tensor_icon.png")
        
        [feat_s, feat_f] >> concat
        concat >> weighted_sum
        [feat_s, feat_f] >> weighted_sum >> fused_feat

    classifier = Custom("线性分类器", "./linear_icon.png")
    output = Custom("输出 DR 分级", "./output_icon.png")
    
    input_img >> resnet
    input_img >> math_prior
    
    fused_feat >> classifier >> output
