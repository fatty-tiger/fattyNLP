import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel
from transformers import BertForSequenceClassification
from transformers import BertForMaskedLM


class BertPooler(nn.Module):
    def __init__(self, pooling):
        super(BertPooler, self).__init__()
        self.pooling = pooling

    def forward(self, out):
        if self.pooling == 'cls':
            out = out.last_hidden_state[:, 0]
        if self.pooling == 'last-avg':
            last = out.last_hidden_state.transpose(1, 2)  # [batch, 768, seqlen]
            out = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
        if self.pooling == 'first-last-avg':
            first = out.hidden_states[1].transpose(1, 2)  # [batch, 768, seqlen]
            last = out.hidden_states[-1].transpose(1, 2)  # [batch, 768, seqlen]
            first_avg = torch.avg_pool1d(first, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
            last_avg = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
            avg = torch.cat((first_avg.unsqueeze(1), last_avg.unsqueeze(1)), dim=1)  # [batch, 2, 768]
            out = torch.avg_pool1d(avg.transpose(1, 2), kernel_size=2).squeeze(-1)  # [batch, 768]
        return out


class BertDualEncoder(nn.Module):
    def __init__(self, pretrained_model, pooling, proj_method, proj_dim):
        super(BertDualEncoder, self).__init__()

        config = AutoConfig.from_pretrained(pretrained_model)
        self.bert = AutoModel.from_pretrained(pretrained_model, config=config, add_pooling_layer=False)
        # self.bert.gradient_checkpointing_enable()  # 开启梯度检查点
        self.bert.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
        self.hidden_size = config.hidden_size
        self.pooler = BertPooler(pooling)
        self.proj_method = proj_method
        if self.proj_method == 'fc':
            self.dense = nn.Linear(self.hidden_size, proj_dim)
            self.activation = nn.Tanh()

    def forward(self, input_ids, target_input_ids, attention_mask=None, token_type_ids=None,
                target_attention_mask=None, target_token_type_ids=None):
        source_pred = self.bert(input_ids, attention_mask, token_type_ids,
                                output_hidden_states=True, return_dict=True)
        source_pred = self.pooler(source_pred)
        
        target_pred = self.bert(target_input_ids, target_attention_mask, target_token_type_ids,
                                output_hidden_states=True, return_dict=True)
        target_pred = self.pooler(target_pred)

        if self.proj_method == 'fc':
            source_pred = self.activation(self.dense(source_pred))
            target_pred = self.activation(self.dense(target_pred))

        source_pred = F.normalize(source_pred)
        target_pred = F.normalize(target_pred)
        return source_pred, target_pred
    

# class BertDualEncoder(nn.Module):
#     def __init__(self, model_params):
#         super(BertDualEncoder, self).__init__()

#         config = AutoConfig.from_pretrained(model_params.pretrained_model)
#         if hasattr(model_params, "dropout"):
#             config.hidden_dropout_prob = model_params.dropout

#         self.bert = AutoModel.from_pretrained(model_params.pretrained_model,
#                                               config=config, add_pooling_layer=False)
#         self.hidden_size = config.hidden_size
#         self.pooler = BertPooler(model_params.pooling)
#         self.proj = model_params.proj
#         if self.proj == 'fc':
#             self.dense = nn.Linear(self.hidden_size, model_params.output_dim)
#             self.activation = nn.Tanh()

#     def forward(self, input_ids, target_input_ids, attention_mask=None, token_type_ids=None,
#                 target_attention_mask=None, target_token_type_ids=None):
#         source_pred = self.bert(input_ids, attention_mask, token_type_ids,
#                                 output_hidden_states=True, return_dict=True)
#         source_pred = self.pooler(source_pred)
        
#         target_pred = self.bert(target_input_ids, target_attention_mask, target_token_type_ids,
#                                 output_hidden_states=True, return_dict=True)
#         target_pred = self.pooler(target_pred)

#         if self.proj == 'fc':
#             source_pred = self.activation(self.dense(source_pred))
#             target_pred = self.activation(self.dense(target_pred))

#         source_pred = F.normalize(source_pred)
#         target_pred = F.normalize(target_pred)
#         return source_pred, target_pred
