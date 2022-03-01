import sys
sys.path.append('.')

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from transformers import BertModel, BertConfig
from transformers import BertPreTrainedModel
from transformers.activations import ACT2FN
from seq2seq import dataset


class BertMtchPairwise(nn.Module):
    def __init__(self, config, margin=0.2):
        super().__init__()
        self.bert = BertModel.from_pretrained(
                "/home/work/pretrained/huggingface/bert-base-chinese",
                add_pooling_layer=True)
        self.loss = nn.MarginRankingLoss(margin=margin, reduction='none')

    def forward(self, feed_list):
        bert_out1 = self.bert(input_ids=feed_list[0])
        bert_out2 = self.bert(input_ids=feed_list[1])
        emb_out1 = bert_out1.pooler_output
        emb_out2 = bert_out2.pooler_output
        cosine_mat_score = torch.matmul(emb_out1, torch.transpose(emb_out2, 0, 1))

        batch_size = feed_list[0].size()[0]
        neg_masks = torch.eye(batch_size)  # 负样本遮蔽
        pos_masks = torch.ones(batch_size, batch_size) - neg_masks  # 正样本遮蔽
        neg_masks, pos_masks = neg_masks.cuda(), pos_masks.cuda()
        labels = torch.ones(batch_size).cuda()

        pos_score = torch.max(torch.mul(cosine_mat_score, neg_masks), 1, keepdim=True).values
        hardest_neg_score = torch.max(torch.mul(cosine_mat_score, pos_masks), 1, keepdim=True).values
        loss = self.loss(pos_score, hardest_neg_score, labels)
        preds = torch.gt(pos_score, hardest_neg_score).squeeze(1)
        # tmp is not on GPU
        # tmp = preds.type(torch.LongTensor)
        # print("tmp", tmp.is_cuda)
        return_dict = {
            'loss': loss,
            'labels': labels,
            'preds': preds,
        }
        return return_dict
