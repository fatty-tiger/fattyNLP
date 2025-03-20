import torch
import torch.nn as nn


def compute_kl_loss(p, q, pad_mask=None):
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')

    if pad_mask is not None:
        p_loss.masked_fill_(pad_mask, 0.)
        q_loss.masked_fill_(pad_mask, 0.)

    loss = (p_loss.mean() + q_loss.mean()) / 2
    return loss


class MulLabelCategoricalCE(nn.Module):
    def __init__(self, rdrop_switch=False, rdrop_alpha=1.0):
        super(MulLabelCategoricalCE, self).__init__()
        self.rdrop_switch = rdrop_switch
        self.rdrop_alpha = rdrop_alpha

    def forward(self, y_pred, y_true, **kwargs):
        """多标签分类的交叉熵
        说明：y_true和y_pred的shape一致，y_true的元素非0即1，
            1表示对应的类为目标类，0表示对应的类为非目标类。
        警告：请保证y_pred的值域是全体实数，换言之一般情况下y_pred
            不用加激活函数，尤其是不能加sigmoid或者softmax！预测
            阶段则输出y_pred大于0的类。如有疑问，请仔细阅读并理解
            本文。
        y_true.shape: (batch_size, label_size)
        y_pred.shape: (batch_size, label_size)
        """
        # print("y_true.shape", y_true.shape)
        # print("y_pred.shape", y_pred.shape)
        # y_true = y_true.reshape(y_true.shape[0], -1)
        # y_pred = y_pred.reshape(y_pred.shape[0], -1)

        y_pred = (1 - 2 * y_true) * y_pred                       # -1 -> pos classes, 1 -> neg classes
        y_pred_neg = y_pred - y_true * 1e12                      # mask the pred outputs of pos classes
        y_pred_pos = y_pred - (1 - y_true) * 1e12                # mask the pred outputs of neg classes
        zeros = torch.zeros_like(y_pred[..., :1])
        y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
        y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
        ce_loss = (neg_loss + pos_loss).mean()
        
        if self.rdrop_switch:
            kl_loss = compute_kl_loss(y_pred[::2], y_pred[1::2]) + \
                      compute_kl_loss(y_pred[1::2], y_pred[::2])
            kl_loss = self.rdrop_alpha * kl_loss
            loss = ce_loss + kl_loss
            return loss
        else:
            return ce_loss