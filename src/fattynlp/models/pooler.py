import torch
import torch.nn as nn

class BertPooler(nn.Module):
    def __init__(self, pooling):
        super(BertPooler, self).__init__()
        self.pooling = pooling

    def forward(self, out):
        if self.pooling == 'cls':
            out = out.last_hidden_state[:, 0]
        if self.pooling == 'last-avg':
            last = out.last_hidden_state.transpose(1, 2)  # [batch, hidden_size, seqlen]
            out = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, hidden_size]
        if self.pooling == 'first-last-avg':
            first = out.hidden_states[1].transpose(1, 2)  # [batch, hidden_size, seqlen]
            last = out.hidden_states[-1].transpose(1, 2)  # [batch, hidden_size, seqlen]
            first_avg = torch.avg_pool1d(first, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, hidden_size]
            last_avg = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, hidden_size]
            avg = torch.cat((first_avg.unsqueeze(1), last_avg.unsqueeze(1)), dim=1)  # [batch, 2, hidden_size]
            out = torch.avg_pool1d(avg.transpose(1, 2), kernel_size=2).squeeze(-1)  # [batch, hidden_size]
        return out
