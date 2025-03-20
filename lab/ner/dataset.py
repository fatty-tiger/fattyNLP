import json
import torch
from dataclasses import dataclass
from typing import Any, Dict, List
from torch.utils.data import Dataset

class BertGlobalPointerDataset(Dataset):
    def __init__(self, train_fpath, tokenizer, max_len, label2id):
        self.datas = []
        with open(train_fpath) as f:
            for line in f:
                d = json.loads(line.strip())
                label = torch.zeros((len(label2id), max_len, max_len), dtype=torch.int64)
                for ent in d['entities']:
                    # +1 tokenize前面会添加[CLS]
                    start = ent['start'] + 1
                    end = ent['end'] + 1
                    if start > max_len - 2:
                        continue
                    if end > max_len - 2:
                        end = max_len - 2
                    index = label2id[ent['label']]
                    label[index, start, end] = 1
                label = label.view(-1)
                feature = tokenizer(d['text'], max_length=max_len, add_special_tokens=True,
                                         padding='max_length', return_tensors='pt', truncation=True,
                                         return_attention_mask=True, return_token_type_ids=True)
                feature["labels"] = label
                self.datas.append(feature)
    
    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        return self.datas[idx]


@dataclass
class BertGlobalPointerDataCollator:
    tokenizer: Any
    max_len: int
    num_label: int
    label2id: Dict[str, int]
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_len = self.max_len
        
        text_list = []
        label_list = []
        print(features[0])
        print(features[1])
        print(features[2])

        for x in features:
            print("##############here#############")
            print(x)
            text_list.append(x['text'])
            label = torch.zeros((self.num_label, max_len, max_len), dtype=torch.int64)
            for ent in x['entities']:
                # +1 tokenize前面会添加[CLS]
                start = ent['start'] + 1
                end = ent['end'] + 1
                if start > max_len - 2:
                    continue
                if end > max_len - 2:
                    end = max_len - 2
                index = self.label2id[ent['label']]
                label[index, start, end] = 1
            label_list.append(label.unsqueeze(0))
        labels = torch.cat(label_list, dim=0)

        assert len(features) == len(text_list) == len(label_list)

        feed_dict = self.tokenizer(text_list, max_length=self.max_len, add_special_tokens=True,
                                   padding='max_length', return_tensors='pt', truncation=True,
                                   return_attention_mask=True, return_token_type_ids=True)
        feed_dict["labels"] = labels
        return feed_dict
