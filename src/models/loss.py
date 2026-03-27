import torch
import torch.nn as nn

# class JointOrdinalLoss(nn.Module):
#     """
#     序数回归与多分类联合损失函数
#     结合交叉熵 (CE) 与均方误差 (MSE)，在保证分类精度的同时，严厉惩罚跨级误判。
#     """
#     def __init__(self, alpha=0.5):
#         """
#         :param alpha: 控制序数惩罚 (MSE) 权重的超参数
#         """
#         super(JointOrdinalLoss, self).__init__()
#         self.alpha = alpha
#         self.ce_loss = nn.CrossEntropyLoss()
#         self.mse_loss = nn.MSELoss()
#
#     def forward(self, outputs, targets):
#         # 1. 标准分类损失
#         ce = self.ce_loss(outputs, targets)
#         
#         # 2. 序数惩罚损失 (计算连续的期望预测值)
#         # 将 logits 转换为概率分布
#         probs = torch.softmax(outputs, dim=1)
#         
#         # 创建类别索引 [0, 1, 2, 3, 4]
#         classes = torch.arange(outputs.size(1), device=outputs.device, dtype=torch.float32)
#         
#         # 计算期望预测值 (Expected value) -> \sum (p_i * i)
#         expected_preds = torch.sum(probs * classes, dim=1)
#         
#         # 计算预测期望值与真实标签之间的 MSE
#         mse = self.mse_loss(expected_preds, targets.float())
#         
#         # 3. 联合损失
#         total_loss = ce + self.alpha * mse
#         
#         return total_loss
class JointOrdinalLoss(nn.Module):
    """
    改进的序数回归与多分类联合损失函数
    使用 SmoothL1Loss 替换 MSE，提高训练稳定性并防止梯度爆炸
    """
    def __init__(self, alpha=0.1):
        super(JointOrdinalLoss, self).__init__()
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.SmoothL1Loss()  # 使用平滑 L1 损失

    def forward(self, outputs, targets):
        # 1. 标准分类损失
        ce = self.ce_loss(outputs, targets)
        
        # 2. 序数惩罚损失
        probs = torch.softmax(outputs, dim=1)
        classes = torch.arange(outputs.size(1), device=outputs.device, dtype=torch.float32)
        
        # 计算期望预测值 (Expected value) -> \sum (p_i * i)
        expected_preds = torch.sum(probs * classes, dim=1)
        
        # 使用 Smooth L1 计算预期值与真实标签的差异
        mse = self.mse_loss(expected_preds, targets.float())
        
        # 3. 联合损失
        total_loss = ce + self.alpha * mse
        
        return total_loss
