import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.autograd import Variable

def compute_kl_loss(p, q, pad_mask=None):
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')

    if pad_mask is not None:
        p_loss.masked_fill_(pad_mask, 0.)
        q_loss.masked_fill_(pad_mask, 0.)

    loss = (p_loss.mean() + q_loss.mean()) / 2
    return loss


class CategoricalCE(nn.Module):
    def __init__(self, rdrop_switch=False, rdrop_alpha=1.0):
        super(CategoricalCE, self).__init__()
        self.ce_loss_func = CrossEntropyLoss(reduction='mean')
        self.rdrop_switch = rdrop_switch
        self.rdrop_alpha = rdrop_alpha

    def forward(self, y_pred, y_true):
        ce_loss = self.ce_loss_func(y_pred, y_true)       
        if self.rdrop_switch:
            kl_loss = compute_kl_loss(y_pred[::2], y_pred[1::2]) + \
                      compute_kl_loss(y_pred[1::2], y_pred[::2])
            kl_loss = self.rdrop_alpha * kl_loss
            loss = ce_loss + kl_loss
            return {
                "loss": loss,
                "ce_loss": ce_loss,
                "kl_loss": kl_loss,
            }
        else:
            return {
                "loss": ce_loss,
                "ce_loss": ce_loss,
            }


class MulLabelCategoricalCE(nn.Module):
    def __init__(self, rdrop_switch=False, rdrop_alpha=1.0):
        super(MulLabelCategoricalCE, self).__init__()
        self.rdrop_switch = rdrop_switch
        self.rdrop_alpha = rdrop_alpha

    def forward(self, y_true, y_pred):
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
            return {
                "loss": loss,
                "ce_loss": ce_loss,
                "kl_loss": kl_loss,
            }
        else:
            return {
                "loss": ce_loss,
                "ce_loss": ce_loss,
            }


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, reduction=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.reduction = reduction

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input, dim=-1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt

        if self.reduction == 'sum':
            return loss.sum()
        else:
            return loss.mean()
    
if __name__ == "__main__":
    loss_func = FocalLoss(gamma=2)
    input = torch.randn(3, 5, requires_grad=True)
    target = torch.randint(5, (3,), dtype=torch.int64)
    print("input:\n", input)
    print("target:\n", target)
    loss = loss_func(input, target)
    print(loss.data)
