import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoConfig, AutoModel

from fattynlp.common.register  import RegisterSet
from fattynlp.models.pooler import BertPooler


@RegisterSet.models.register
class BertEncoder(nn.Module):
    def __init__(self, model_params):
        super(BertEncoder, self).__init__()

        print(model_params.pretrained_model)
        config = AutoConfig.from_pretrained(model_params.pretrained_model)
        if hasattr(model_params, "dropout"):
            config.hidden_dropout_prob = model_params.dropout

        self.bert = AutoModel.from_pretrained(model_params.pretrained_model,
                                              config=config, add_pooling_layer=False)
        self.hidden_size = config.hidden_size
        self.pooler = BertPooler(model_params.pooling)
        self.proj = model_params.proj
        if self.proj == 'fc':
            self.dense = nn.Linear(self.hidden_size, model_params.output_dim)
            self.activation = nn.Tanh()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        out = self.bert(input_ids, attention_mask, token_type_ids,
                        output_hidden_states=True, return_dict=True)
        out = self.pooler(out)

        if self.proj == 'fc':
            out = self.dense(out)
            out = self.activation(out)

        out = F.normalize(out)
        return out