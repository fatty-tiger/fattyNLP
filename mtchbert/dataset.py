import numpy as np
import torch
import torch.nn as nn

import transformers
transformers.logging.set_verbosity_error()

from torch.utils.data import IterableDataset
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import BertTokenizer


class SimPairDataSet(IterableDataset):
    """ SimPairDataSet """
    def __init__(self, input_file, batch_size, max_len):
        self.input_file = input_file
        self.batch_size = batch_size
        self.max_len = max_len
        self.tokenizer = BertTokenizer.from_pretrained(
            "/home/work/pretrained/huggingface/bert-base-chinese")
        vocab = self.tokenizer.vocab
        self.pad_index = vocab.get(self.tokenizer.pad_token,
                                   vocab.get(self.tokenizer.unk_token))

    def __iter__(self):
        with open(self.input_file) as f:
            batch_lst1, batch_lst2 = [], []
            for line_idx, line in enumerate(f):
                sent1, sent2 = line.strip().split('\t')
                batch_lst1.append(sent1)
                batch_lst2.append(sent2)
                if len(batch_lst1) == self.batch_size:
                    yield self._batch_encode(batch_lst1, batch_lst2)
                    batch_lst1, batch_lst2 = [], []
            if batch_lst1:
                yield self._batch_encode(batch_lst1, batch_lst2)

    def _batch_encode(self, batch_lst1, batch_lst2):
        results = []
        for batch_lst in [batch_lst1, batch_lst2]:
            encoded = self.tokenizer(batch_lst, padding='longest',
                                     truncation='longest_first', max_length=self.max_len)
            paddings = [self.pad_index] * (self.max_len - len(encoded['input_ids'][0]))
            input_ids = torch.tensor([x + paddings for x in encoded['input_ids']])
            results.append(input_ids)
        return results


def test():
    max_len = 32
    batch_size = 4
    train_dataset = SimPairDataSet('./data/mtchbert/sample.txt', batch_size, max_len)
    train_dataloader = DataLoader(train_dataset, None, num_workers=1)
    for idx, batch in enumerate(train_dataloader):
        for item in batch:
            print(item.shape)
        break


if __name__ == '__main__':
    test()
