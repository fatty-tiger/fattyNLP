import os
import logging
import torch

from argument import BertGlobalPointerModelArguments
from argument import BertGlobalPointerTrainingArguments
from char_tokenizer import CharTokenizer
from model import BertGlobalPointer
from dataset import BertGlobalPointerDataset
from loss import MulLabelCategoricalCE

from transformers import Trainer
from transformers import HfArgumentParser


def train():
    parser = HfArgumentParser((BertGlobalPointerModelArguments, BertGlobalPointerTrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses()
    model_args: BertGlobalPointerModelArguments
    training_args: BertGlobalPointerTrainingArguments

    max_len = 128
    tokenizer = CharTokenizer.from_pretrained(model_args.model_name_or_path)
    model = BertGlobalPointer(model_args.model_name_or_path, model_args.entity_num, model_args.inner_dim, model_args.rope)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_args.learning_rate)
    loss_func = MulLabelCategoricalCE()
    label2id = {
        "position": 0,
        "name": 1,
        "book": 2,
        "company": 3,
        "government": 4,
        "movie": 5,
        "scene": 6,
        "address": 7,
        "game": 8,
        "organization": 9
    }

    # train data
    train_dataset = BertGlobalPointerDataset(training_args.train_data, tokenizer, max_len, label2id)
    trainer = Trainer(model, training_args,
                      train_dataset=train_dataset,
                      optimizers=(optimizer, None),
                      compute_loss_func=loss_func)
    trainer.train()

if __name__ == "__main__":
    train()
