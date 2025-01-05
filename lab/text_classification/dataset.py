import torch
import json
import random
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from utils import file_util


def get_label2id(label_fpath):
    label2id = {}
    with open(label_fpath) as f:
        tmp_d = json.load(f)
        for level, d in tmp_d.items():
            level = int(level)
            label2id[level] = {k: v for k, v in d.items()}
    return label2id


class JsonlHierClassifyDataset(Dataset):
    """
    基于多标签模式的层次分类数据(Jsonl格式)
    """
    def __init__(self, data_fpath, label2id, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.datas = [d for _, d in file_util.reader(data_fpath)]
        self.label2id = label2id
        self.num_labels = sum([len(v) for v in label2id.values()])

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        return self.datas[idx]

    def collate_fn(self, batch):
        batch_size = len(batch)
        texts = [x["text_a"] for x in batch]
        feed_dict = self.tokenizer(texts, max_length=self.max_len, add_special_tokens=True,
                                   padding='max_length', return_tensors='pt', truncation=True,
                                   return_attention_mask=True, return_token_type_ids=False)
        input_labels = torch.zeros((batch_size, self.num_labels))
        for idx, x in enumerate(batch):
            for level, label in enumerate(x["labels"]):
                input_labels[idx, self.label2id[level][label]] = 1
        feed_dict["input_labels"] = input_labels
        return feed_dict


class JsonlHierClassifyDatasetV2(Dataset):
    """
    基于多标签模式的层次分类数据(Jsonl格式)
    """
    def __init__(self, data_fpath, label2id, tokenizer, max_len, rdrop_switch=False):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.datas = [d for _, d in file_util.reader(data_fpath)]
        self.label2id = label2id
        self.num_labels = sum([len(v) for v in label2id.values()])
        self.rdrop_switch = rdrop_switch

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        return self.datas[idx]

    def collate_fn(self, batch):
        batch_size = len(batch)

        texts = []
        for x in batch:
            texts.append(x['text_a'])
            if self.rdrop_switch:
                texts.append(x['text_b'])
                #texts.append(x['text_a'])
        
        feed_dict = self.tokenizer(texts, max_length=self.max_len, add_special_tokens=True,
                                   padding='max_length', return_tensors='pt', truncation=True,
                                   return_attention_mask=True, return_token_type_ids=False)
        label_size = batch_size * 2 if self.rdrop_switch else batch_size
        input_labels = torch.zeros((label_size, self.num_labels))

        # 0=>0,1 1=>2,3 2=>4,5
        for idx, x in enumerate(batch):
            idx_2 = idx * 2 if self.rdrop_switch else idx
            for level, label in enumerate(x["labels"]):
                input_labels[idx_2, self.label2id[level][label]] = 1
                if self.rdrop_switch:
                    input_labels[idx_2 + 1, self.label2id[level][label]] = 1
        
        # bug代码！
        # for idx, x in enumerate(batch):
        #     for level, label in enumerate(x["labels"]):
        #         input_labels[idx, self.label2id[level][label]] = 1
        #     if self.rdrop_switch:
        #         for level, label in enumerate(x["labels"]):
        #             input_labels[idx + 1, self.label2id[level][label]] = 1
        feed_dict["input_labels"] = input_labels
        return feed_dict