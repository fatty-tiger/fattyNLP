import sys
import os
sys.path.append('.')

import logging
import torch

from transformers import BertModel, BertConfig
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from mtchbert import BertMtchMatrixWise
from mtchbert import SimPairDataSet


LOG_DATE_FMT = '%Y‐%m‐%d %H:%M:%S'
LOG_FMT = '%(levelname)s: %(asctime)s: %(filename)s:%(lineno)d * %(thread)d %(message)s'
logging.basicConfig(level=logging.DEBUG,
                    stream=sys.stderr,
                    datefmt=LOG_DATE_FMT,
                    format=LOG_FMT)


def main():
    with_cuda = True
    max_len = 512
    batch_size = 8
    epochs = 1

    log_per_step = 10
    eval_train_step = 10

    device = torch.device("cuda:0" if with_cuda and torch.cuda.is_available() else "cpu")
    logging.info(device)
    logging.info(os.environ['CUDA_VISIBLE_DEVICES'])

    logging.info("building model")
    config = BertConfig.from_pretrained(
                "/home/work/pretrained/huggingface/bert-base-chinese")
    model = BertMtchMatrixWise(config)
    model.to(device)

    if with_cuda and torch.cuda.device_count() > 1:
        device_ids = list(range(torch.cuda.device_count()))
        logging.info(device_ids)
        model = nn.DataParallel(model, device_ids=device_ids)

    logging.info("building dataset")
    train_dataset = SimPairDataSet('./data/mtchbert/sample.txt', batch_size, max_len)
    train_dataloader = DataLoader(train_dataset, None, num_workers=1)

    logging.info("building optimizer")
    optimizer = Adam(model.parameters(), lr=1e-4)

    logging.info("training...")
    for i in range(epochs):
        total_loss = 0.0
        total_corrects = 0
        total_samples = 0
        model.train()
        for steps, batch in enumerate(train_dataloader):
            feed_list = [x.to(device) for x in batch]
            return_dict = model(feed_list)
            optimizer.zero_grad()
            loss = return_dict['loss']
            loss.mean().backward()
            optimizer.step()
            batch_loss = loss.mean().item()
            total_loss += batch_loss
            if steps % log_per_step == 0:
                avg_loss = total_loss / (steps + 1)
                logging.info("=====LOSS=====epoch={}, step={}, batch_loss={}, avg_loss={}".format(i, steps, batch_loss, avg_loss))

            if steps % eval_train_step == 0:
                preds = return_dict['preds'].cpu().type(torch.LongTensor)
                labels = return_dict['labels'].cpu().type(torch.LongTensor)
                corrects = torch.sum(torch.eq(preds, labels)).item()
                samples = preds.size()[0]
                total_corrects += corrects
                total_samples += samples
                batch_acc = corrects * 1.0 / samples
                avg_acc = total_corrects * 1.0 / total_samples
                logging.info("=====METRICS=====epoch={}, step={}, batch_acc={}, avg_acc={}".format(i, steps, batch_acc, avg_acc))


if __name__ == '__main__':
    main()

