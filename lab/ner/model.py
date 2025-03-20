#!/usr/bin/env python
# encoding: utf8
################################################################################
#
# Copyright (c) 2023 zkh.com, Inc. All Rights Reserved
#
################################################################################
import torch
import torch.nn as nn
from transformers import AutoConfig
from transformers import AutoModel


class BertGlobalPointer(nn.Module):
    def __init__(self, pretrained_model, entity_num, inner_dim, rope):
        super(BertGlobalPointer, self).__init__()
        config = AutoConfig.from_pretrained(pretrained_model)
        self.bert = AutoModel.from_pretrained(pretrained_model, config=config, add_pooling_layer=False)
        self.entity_num = entity_num
        self.inner_dim = inner_dim
        self.hidden_size = config.hidden_size
        self.dense = nn.Linear(self.hidden_size, self.entity_num * self.inner_dim * 2)
        self.RoPE = rope

    def sinusoidal_position_embedding(self, batch_size, seq_len, output_dim, device):
        position_ids = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(-1)
        indices = torch.arange(0, output_dim // 2, dtype=torch.float)
        indices = torch.pow(10000, -2 * indices / output_dim)
        embeddings = position_ids * indices
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = embeddings.repeat((batch_size, *([1]*len(embeddings.shape))))
        embeddings = torch.reshape(embeddings, (batch_size, seq_len, output_dim))
        embeddings = embeddings.to(device)
        return embeddings
        
    def forward(self, input_ids, attention_mask, token_type_ids):
        # (bsz, 1, maxlen) => (bsz, maxlen)
        input_ids = input_ids.squeeze(1)
        token_type_ids = token_type_ids.squeeze(1)
        attention_mask = attention_mask.squeeze(1)

        context_outputs = self.bert(input_ids.squeeze(1), attention_mask.squeeze(1), token_type_ids.squeeze(1))
        # last_hidden_state:(batch_size, seq_len, hidden_size)
        last_hidden_state = context_outputs[0]

        batch_size = last_hidden_state.size()[0]
        seq_len = last_hidden_state.size()[1]

        # outputs:(batch_size, seq_len, entity_num*inner_dim*2)
        outputs = self.dense(last_hidden_state)
        # tuple of size entity_num, each element: (batch_size, seq_len, inner_dim*2)
        outputs = torch.split(outputs, self.inner_dim * 2, dim=-1)
        # outputs:(batch_size, seq_len, entity_num, inner_dim*2)
        outputs = torch.stack(outputs, dim=-2)
        # qw,kw:(batch_size, seq_len, entity_num, inner_dim)
        # (batch_size, entity_num, seq_len, inner_dim) * (batch_size, entity_num, inner_dim, seq_len)
        ## 再劈成两半，分别表示qw和kw
        qw, kw = outputs[...,:self.inner_dim], outputs[...,self.inner_dim:]
        if self.RoPE:
            # pos_emb:(batch_size, seq_len, inner_dim)
            pos_emb = self.sinusoidal_position_embedding(batch_size, seq_len, self.inner_dim, input_ids.device)
            # cos_pos,sin_pos: (batch_size, seq_len, 1, inner_dim)
            cos_pos = pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)
            sin_pos = pos_emb[..., None, ::2].repeat_interleave(2, dim=-1)
            qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], -1)
            qw2 = qw2.reshape(qw.shape)
            qw = qw * cos_pos + qw2 * sin_pos
            kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], -1)
            kw2 = kw2.reshape(kw.shape)
            kw = kw * cos_pos + kw2 * sin_pos
        #print("qw:", qw.shape)
        #print("kw:", kw.shape)
        #return qw
        
        # # logits:(batch_size, entity_num, seq_len, seq_len)
        logits = torch.einsum('bmhd,bnhd->bhmn', qw, kw)
        #print("logits:", logits.shape)
        #print(logits[0][0][0])

        qw = qw.transpose(1, 2).reshape((batch_size * self.entity_num, seq_len, self.inner_dim))
        kw = kw.transpose(1, 2).transpose(2, 3).reshape((batch_size * self.entity_num, self.inner_dim, seq_len))
        #print("qw:", qw.shape)
        #print("kw:", kw.shape)
        logits = torch.bmm(qw, kw).reshape((batch_size, self.entity_num, seq_len, seq_len))
        #print("logits:", logits.shape)
        #print(logits[0][0][0])

        #return logits

        # padding mask
        pad_mask = attention_mask.unsqueeze(1).unsqueeze(1).expand(batch_size, self.entity_num, seq_len, seq_len)
        logits = logits * pad_mask - (1 - pad_mask) * 1e12

        # 排除下三角
        mask = torch.tril(torch.ones_like(logits), -1)
        logits = logits - mask * 1e12
        logits = logits / self.inner_dim ** 0.5
        logits = logits.view(batch_size, -1)

        return logits

    # def inference(self, texts):
    #     logits = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
    #     y_true = input_labels.data.cpu().numpy()
    #     y_pred = logits.data.cpu().numpy()

    #     tmp_true_dict = defaultdict(list)
    #     tmp_pred_dict = defaultdict(list)

    #     for b, l, start, end in zip(*np.where(y_true > 0)):
    #         tmp_true_dict[b].append((id2entity[l], start - 1, end - 1)) 
    #         #y_true_list.append((idx + b, id2entity[l], start, end))

    #     for b, l, start, end in zip(*np.where(y_pred > 0)):
    #         tmp_pred_dict[b].append((id2entity[l], start - 1, end - 1)) 
    #         #y_pred_list.append((idx + b, id2entity[l], start, end))
            
    #     for i in range(batch_size):
    #         y_true_list.append(tmp_true_dict[i][:])
    #         y_pred_list.append(tmp_pred_dict[i][:])
            

