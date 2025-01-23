import os
import logging

from transformers import AutoTokenizer

from datetime import datetime
from model import BertDualEncoder
from dataset import DualEncoderDataset

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.distributed.nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from tqdm import tqdm
from lab.utils import log_util
from text_representation.loss import TextMatchInbatchNegLoss


data_fpath = "/home/jiangjie/github/fattyNLP/lab/data/sample.jsonl"

pretrained_model = "/home/work/pretrained_models/ernie-3.0-nano-zh"
pooling = "cls"
proj_method = "fc"
proj_dim = 8
max_len = 64

batch_size = 2
lr = 5e-5
temperature = 0.05


def train():
    dist.init_process_group("nccl", init_method='env://')
    # 当前进程在全局进程中的排名
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    # 是当前进程在其所在节点上的排名
    local_rank = int(os.environ['LOCAL_RANK'])
    logging.info(f"world_size: {world_size}, rank: {rank}, local_rank: {local_rank}")

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    train_dataset = DualEncoderDataset(data_fpath, tokenizer, max_len)
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, collate_fn=train_dataset.collate_fn)
    model = BertDualEncoder(pretrained_model, pooling, proj_method, proj_dim).to(local_rank)
    model = DistributedDataParallel(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_func = TextMatchInbatchNegLoss()

    model.train()
    
    for batch_idx, batch_data in enumerate(tqdm(train_dataloader)):
        # source        [batch, 1, seq_len] -> [batch, seq_len]
        source = batch_data[0]
        source_input_ids = source.get('input_ids').squeeze(1).to(local_rank)
        source_attention_mask = source.get('attention_mask').squeeze(1).to(local_rank)
        #source_token_type_ids = source.get('token_type_ids').squeeze(1).to(local_rank)

        # target        [batch, 1, seq_len] -> [batch, seq_len]
        target = batch_data[1]
        target_input_ids = target.get('input_ids').squeeze(1).to(local_rank)
        target_attention_mask = target.get('attention_mask').squeeze(1).to(local_rank)
        #target_token_type_ids = target.get('token_type_ids').squeeze(1).to(local_rank)

        # bsz * dim
        source_logits, target_logits = model(source_input_ids, target_input_ids,
                                             attention_mask=source_attention_mask,
                                             target_attention_mask=target_attention_mask)
        
        source_logits_gathered = torch.distributed.nn.all_gather(source_logits)
        source_logits_concated = torch.cat(source_logits_gathered, dim=0)

        # (bsz * gpu_num) * dim
        target_logits_gathered = torch.distributed.nn.all_gather(target_logits)
        target_logits_concated = torch.cat(target_logits_gathered, dim=0)
        
        logging.info(f"rank: {rank}, local_rank: {local_rank}, source_logits: \n{source_logits}, {source_logits.shape}")
        logging.info(f"rank: {rank}, local_rank: {local_rank}, source_logits_concated:\n {source_logits_concated}, {source_logits_concated.shape}")

        logging.info(f"rank: {rank}, local_rank: {local_rank}, target_logits: \n{target_logits}, {target_logits.shape}")
        logging.info(f"rank: {rank}, local_rank: {local_rank}, target_logits_concated:\n {target_logits_concated}, {target_logits_concated.shape}")
        
        similarity = F.cosine_similarity(source_logits_concated.unsqueeze(1), target_logits_concated.unsqueeze(0), dim=-1)
        logging.info(f"rank: {rank}, local_rank: {local_rank}, similarity: {similarity}, {similarity.shape}")

        targets = torch.arange(target_logits_concated.shape[0]).to(local_rank)
        logging.info(f"rank: {rank}, local_rank: {local_rank}, targets: {targets}, {targets.shape}")

        break

    dist.destroy_process_group()


if __name__ == '__main__':
    log_util.simple_init()
    train()