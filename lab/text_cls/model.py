import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoConfig, AutoModel


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


class BertClassificationModel(nn.Module):
    def __init__(self, pretrained_model, pooling, classifier_dropout, num_labels):
        super(BertClassificationModel, self).__init__()
        config = AutoConfig.from_pretrained(pretrained_model)
        self.bert = AutoModel.from_pretrained(pretrained_model,
                                              config=config, add_pooling_layer=False)
        self.hidden_size = config.hidden_size
        self.pooler = BertPooler(pooling)
        self.classifier = nn.Sequential(
            nn.Dropout(classifier_dropout),
            nn.Linear(self.hidden_size, num_labels)
        )

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        out = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                        output_hidden_states=True, return_dict=True)
        out = self.pooler(out)
        out = self.classifier(out)
        return out


class InferenceModel(nn.Module):
    def __init__(self, pretrained_model, pooling, classifier_dropout, size_by_level):
        super(InferenceModel, self).__init__()
        self.size_by_level = size_by_level
        self.num_labels = sum(size_by_level)
        self.topK = 5

        config = AutoConfig.from_pretrained(pretrained_model)
        self.bert = AutoModel.from_pretrained(pretrained_model,
                                              config=config, add_pooling_layer=False)
        self.hidden_size = config.hidden_size
        self.pooler = BertPooler(pooling)
        self.classifier = nn.Sequential(
            nn.Dropout(classifier_dropout),
            nn.Linear(self.hidden_size, self.num_labels)
        )

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        logits = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                        output_hidden_states=True, return_dict=True)
        logits = self.pooler(logits)
        logits = self.classifier(logits)
        start = 0
        top_indice_list = []
        top_proba_list = []
        for _, size in enumerate(self.size_by_level):
            topK = min(self.topK, size)
            end = start + size
            matrix = logits[:, start: end]
            softmax_matrix = F.softmax(matrix, dim=1)

            topk_res = torch.topk(softmax_matrix, topK)
            top_proba_list.append(topk_res.values)
            topk_indices = topk_res.indices + start
            top_indice_list.append(topk_indices)
            start = end
        top_proba = torch.concat(top_proba_list, dim=1)
        top_indice = torch.concat(top_indice_list, dim=1)
        return top_proba, top_indice
