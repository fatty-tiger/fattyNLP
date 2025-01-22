import sys
import logging
import json
import torch.nn.functional as F

from tqdm import tqdm
#from .base_dataset  import BaseDataset


class DualEncoderDataset(BaseDataset):
    def __init__(self, args):
        super(DualEncoderDataset, self).__init__(args)

    def load_features(self, datas, tokenizer, max_len, mode='train',
                      limits=-1, disable_tqdm=True, **kwargs):
        limits = len(datas) + 1 if limits <= 0 else limits
        logging.info("Start to load data...")
        features = []
        for idx, d in enumerate(tqdm(datas, disable=disable_tqdm)):
            if idx >= limits:
                break
            if mode == 'train':
                feed_dict_a = tokenizer(d["text_a"], max_length=max_len, add_special_tokens=True, 
                                        padding='max_length', return_tensors='pt', truncation=True,
                                        return_attention_mask=True, return_token_type_ids=False)
                feed_dict_b = tokenizer(d["text_b"], max_length=max_len, add_special_tokens=True, 
                                        padding='max_length', return_tensors='pt', truncation=True,
                                        return_attention_mask=True, return_token_type_ids=False)
                features.append((feed_dict_a, feed_dict_b))
            else:
                feed_dict_a = tokenizer(d["text_a"], max_length=max_len, add_special_tokens=True, 
                                        padding='max_length', return_tensors='pt', truncation=True,
                                        return_attention_mask=True, return_token_type_ids=False)
                features.append(feed_dict_a)
        logging.info(f"Data loaded success, {len(features)} lines total")
        return features


class JsonlDualEncoderDataset(DualEncoderDataset):
    def __init__(self, args):
        super(JsonlDualEncoderDataset, self).__init__(args)

    def line_processor(self, input_line, **kwargs):
        d = json.loads(input_line)
        return d