import torch
import torch.nn as nn
import torch.nn.functional as F


class TextMatchSupInbatchLoss(nn.Module):
    """监督模式In-Batch损失"""
    def __init__(self):
        super(TextMatchSupInbatchLoss, self).__init__()
        self.temperature = 0.05
    
    def forward(self, source_pred, target_pred, y_true):
        sim = F.cosine_similarity(source_pred.unsqueeze(1), target_pred.unsqueeze(0), dim=-1)
        sim = sim / self.temperature
        loss = F.cross_entropy(sim, y_true)
        return torch.mean(loss)


# def simcse_unsup_loss(y_pred, device, temperature=0.1):
#     """SimCSE无监督模式的损失函数
#     Args:
#         y_pred (tensor): bert的输出, [batch_size * 2, 768]
#     """
#     # 得到y_pred对应的label, [1, 0, 3, 2, ..., batch_size-1, batch_size-2]
#     y_true = torch.arange(y_pred.shape[0], device=device)
#     y_true = (y_true - y_true % 2 * 2) + 1
#     # batch内两两计算相似度, 得到相似度矩阵(对角矩阵)
#     sim = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=-1)
#     # 将相似度矩阵对角线置为很小的值, 消除自身的影响
#     sim = sim - torch.eye(y_pred.shape[0], device=device) * 1e12
#     # 相似度矩阵除以温度系数
#     sim = sim / temperature
#     # 计算相似度矩阵与y_true的交叉熵损失
#     # 计算交叉熵，每个case都会计算与其他case的相似度得分，得到一个得分向量，目的是使得该得分向量中正样本的得分最高，负样本的得分最低
#     loss = F.cross_entropy(sim, y_true)
#     return torch.mean(loss)