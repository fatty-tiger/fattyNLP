import torch
from torch.utils.data import Dataset
import utils


class SingleLabelTextClassifierDataset(Dataset):
    def __init__(self, data_fpath, tokenizer, max_len, max_rows=-1, rdrop_switch=False):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.datas = [d for idx, d in utils.reader(data_fpath) if max_rows <= 0 or idx + 1 < max_rows]
        self.rdrop_switch = rdrop_switch

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        return self.datas[idx]

    def collate_fn(self, batch):
        bsz = len(batch)

        texts = []
        for x in batch:
            texts.append(x['text_a'])
            if self.rdrop_switch:
                texts.append(x['text_a'])
        
        feed_dict = self.tokenizer(texts, max_length=self.max_len, add_special_tokens=True,
                                   padding='max_length', return_tensors='pt', truncation=True,
                                   return_attention_mask=True, return_token_type_ids=False)
        batch_size = bsz * 2 if self.rdrop_switch else bsz
        input_labels = torch.zeros((batch_size,), dtype=torch.long)

        for idx, x in enumerate(batch):
            idx_2 = idx * 2 if self.rdrop_switch else idx
            label_id = x['label_id']
            input_labels[idx_2] = label_id
            if self.rdrop_switch:
                input_labels[idx_2 + 1] = label_id
        feed_dict["input_labels"] = input_labels
        return feed_dict
