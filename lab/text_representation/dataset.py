from torch.utils.data import Dataset
from transformers import BertTokenizerFast

from lab.utils import utils

class DualEncoderDataset(Dataset):
    def __init__(self, data_fpath, tokenizer, max_len, is_train=True, max_rows=-1):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.datas = [d for idx, d in utils.reader(data_fpath) if max_rows <= 0 or idx + 1 < max_rows]

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        return self.datas[idx]

    def collate_fn(self, batch):
        texts_a = []
        texts_b = []
        for x in batch:
            texts_a.append(x['text_a'])
            texts_b.append(x['text_b'])
        
        feed_dict_a = self.tokenizer(texts_a, max_length=self.max_len, add_special_tokens=True,
                                     padding='max_length', return_tensors='pt', truncation=True,
                                     return_attention_mask=True, return_token_type_ids=False)
        feed_dict_b = self.tokenizer(texts_b, max_length=self.max_len, add_special_tokens=True,
                                     padding='max_length', return_tensors='pt', truncation=True,
                                     return_attention_mask=True, return_token_type_ids=False)
        
        return feed_dict_a, feed_dict_b


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    train_data_fpath = ""
    tokenizer = BertTokenizerFast.from_pretrained("")
    max_len = 64
    batch_size = 4

    dataset = DualEncoderDataset(train_data_fpath, tokenizer, max_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate_fn)
    for batch_idx, batch_data in enumerate(tqdm(dataloader)):
        print(batch_data)
        break
