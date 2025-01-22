import os
import logging


from datetime import datetime
from model import BertDualEncoder
from dataset import DualEncoderDataset
from loss import TextMatchSupInbatchLoss

import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

import torch.distributed.nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from tqdm import tqdm

# log_util.simple_init()


def train():
    dist.init_process_group("nccl", init_method='env://')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ['LOCAL_RANK'])
    logging.info(f"world_size: {world_size}, rank: {rank}, local_rank: {local_rank}")

    if rank == 0:
        now = datetime.now()
        ftime = now.strftime("%Y%m%d_%H%M")
        ckp_dir = f"cross_batch/runtime_{ftime}/ckp"
        if not os.path.exists(ckp_dir):
            os.makedirs(ckp_dir)


    tokenizer = build_tokenizer(args)
    dataset = JsonlDualEncoderDataset(args)
    train_dataset = dataset.build_dataset("train", tokenizer, args.model.max_len, mode="train", disable_tqdm=False)
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_dataloader = DataLoader(train_dataset, batch_size=args.train.batch_size,
                                  num_workers=args.train.num_workers, sampler=train_sampler)
    model = BertDualEncoder(args.model).to(local_rank)
    model = DistributedDataParallel(model, device_ids=[local_rank])

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.train.lr)
    loss_func = CrossEntropyLoss()

    model.train()
    best = 0
    step = 0
    for epoch in range(1, args.train.epochs + 1):
        epoch_losses = []
        for batch_idx, data in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()

            # source        [batch, 1, seq_len] -> [batch, seq_len]
            source = data[0]
            source_input_ids = source.get('input_ids').squeeze(1).to(local_rank)
            source_attention_mask = source.get('attention_mask').squeeze(1).to(local_rank)
            #source_token_type_ids = source.get('token_type_ids').squeeze(1).to(local_rank)

            # target        [batch, 1, seq_len] -> [batch, seq_len]
            target = data[1]
            target_input_ids = target.get('input_ids').squeeze(1).to(local_rank)
            target_attention_mask = target.get('attention_mask').squeeze(1).to(local_rank)
            #target_token_type_ids = target.get('token_type_ids').squeeze(1).to(local_rank)

            # bsz * dim
            source_logits, target_logits = model(source_input_ids, target_input_ids,
                                                 attention_mask=source_attention_mask,
                                                 target_attention_mask=target_attention_mask)
            
            batch_size = source_logits.shape[0]
            offset = batch_size * rank
            
            # (bsz * gpu_num) * dim
            target_logits_gathered = torch.distributed.nn.all_gather(target_logits)
            target_logits_new = torch.cat(target_logits_gathered, dim=0)

            true_labels = (torch.arange(source_logits.shape[0]) + offset).to(local_rank)
            sim_new = F.cosine_similarity(source_logits.unsqueeze(1), target_logits_new.unsqueeze(0), dim=-1)
            sim_new = sim_new / args.train.temperature

            loss = loss_func(sim_new, true_labels)

            loss.backward()
            optimizer.step()
            step += 1

            # 打印损失
            if step % args.train.log_step == 0:
                logging.info('loss:{}, in step {} epoch {} rank {}'.format(loss, step, epoch, rank))
            epoch_losses.append(loss)
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        logging.info('avg_loss:{}, in epoch {} rank {}'.format(avg_loss, epoch, rank))
        if rank == 0 and epoch % args.train.save_epoch == 0:
            ckp_fpath = f"{ckp_dir}/ckp-bsz-{args.train.batch_size}-epoch-{epoch}.pt"
            torch.save(model.module.state_dict(), ckp_fpath)
            logging.info(f"saving checkpoints to {ckp_fpath}")
    dist.destroy_process_group()


if __name__ == '__main__':
    train()