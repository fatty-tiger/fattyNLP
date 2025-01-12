import os
import logging
import torch
import time
import json
from datetime import datetime
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR
from transformers import BertTokenizerFast

from model import BertClassificationModel
from dataset import SingleLabelTextClassifierDataset
from loss import CategoricalCE

from sklearn.metrics import classification_report
from prettytable import PrettyTable, MARKDOWN

import utils

train_data_fname = "train.jsonl"
train_max_rows = -1
dev_data_fname_list = [
    "dev.jsonl",
]
dev_max_rows = -1
label_fname = "label2id.json"

device_name = "cuda:0"
device_name = "cuda:1"
device = torch.device(device_name)

pretrained_model = "/home/work/pretrained_models/ernie-3.0-medium-zh"
pretrained_model_name = pretrained_model.split('/')[-1]
pooling = "cls"
classifier_dropout = 0.1
max_len = 512

train_epochs = 5
lr = 1e-4
warmup_steps = 1000
rdrop_switch = True
rdrop_alpha = 4.0
batch_size = 32
dev_batch_size = 256

log_step = 100
eval_step = 2000
hparam_epoch = 1
save_epoch = 1
min_save_epoch = 5

hparam_dict = {
    "pretrained_model": pretrained_model_name,
    "pooling": pooling,
    "classifier_dropout": classifier_dropout,
    "max_len": max_len,
    "train_data_fname": train_data_fname,
    "learning_rate": lr,
    "batch_size": batch_size*2 if rdrop_switch else batch_size,
    "warmup_steps": warmup_steps,
    "rdrop_switch": rdrop_switch,
    "rdrop_alpha": rdrop_alpha
}


def lr_lambda(current_step: int):
    global lr, warmup_steps
    if current_step < warmup_steps:
        # 线性增长，从 0 增长到 1
        return float(current_step) / float(warmup_steps)
    # warmup 后可以切换到其他策略，例如余弦退火
    return 1.0


def to_markdown(d):
    table = PrettyTable()
    table.set_style(MARKDOWN)
    table.field_names = ["Key", "Value"]
    # 将字典中的键值对添加到表格中
    for key, value in d.items():
        table.add_row([key, value])
    return table.get_string()


class SingleLabelTextClassifierTrainer(object):
    """
    Trainer for Single Label Multi Class Text Classifier
    """
    def __init__(self, data_dir):
        self.data_dir = data_dir
        label_fpath = os.path.join(data_dir, label_fname)
        self.label2id = {}
        with open(label_fpath) as f:
            self.label2id = json.load(f)
        self.num_labels = len(self.label2id)
        self.id2label = {v: k for k, v in self.label2id.items()}
    
    def step(self, model, step_data):
        input_ids = step_data['input_ids'].to(device)
        attention_mask = step_data['attention_mask'].to(device)
        logits = model(input_ids, attention_mask)
        
        targets = None
        if 'input_labels' in step_data:
            targets = step_data['input_labels'].to(device)
        
        return logits, targets

    def evaluate(self, model, dev_dataloader, name, epoch, step):
        model.eval()
        y_true_list = []
        y_pred_list = []
        eval_losses = []
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(tqdm(dev_dataloader)):
                logits, targets = self.step(model, batch_data)
                loss_dict = self.loss_func(logits, targets)
                eval_losses.append(loss_dict['loss'].cpu().item())
                y_true_list.extend(targets.cpu().numpy().tolist())
                y_pred_list.extend(logits.argmax(dim=1).cpu().numpy().tolist())
        avg_loss = sum(eval_losses) / len(eval_losses)
        classify_metric = classification_report(y_true_list, y_pred_list, output_dict=True, zero_division=0.0)          
        
        metric_dict = {}
        metric_dict['accuracy'] = classify_metric['accuracy']
        metric_dict['macro_recall'] = classify_metric['macro avg']['recall']
        metric_dict['macro_precision'] = classify_metric['macro avg']['precision']
        metric_dict['macro_f1_score'] = classify_metric['macro avg']['f1-score']
        
        self.logger.info(f"Metric of {name}, epoch-{epoch}, step-{step}, accuracy: {metric_dict['accuracy']:.4f}, f1_score: {metric_dict['macro_f1_score']:.4f}")

        writer.add_scalar(f"Loss/{name}", avg_loss, step)
        scalar_dict = {}
        scalar_dict['accuracy'] = metric_dict['accuracy']
        scalar_dict['macro_f1_score'] = metric_dict['macro_f1_score']
        writer.add_scalars(f"Metric/{name}", scalar_dict, step)

        model.train()

        return metric_dict

    def train(self, output_dir):
        self.logger = logger = utils.init_logger(output_dir, "train", level=logging.DEBUG)
        logger.info("Hyper Parameters:\n" + json.dumps(hparam_dict, ensure_ascii=False, indent=4))
        
        data_dir = self.data_dir
        
        self.tokenizer = tokenizer = BertTokenizerFast.from_pretrained(pretrained_model)
        
        train_data_fpath = os.path.join(data_dir, train_data_fname)
        train_dataset = SingleLabelTextClassifierDataset(train_data_fpath, tokenizer, max_len, rdrop_switch=rdrop_switch, max_rows=train_max_rows)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)
        logger.info("train_dataloader built success")
        
        dev_dataloader_dict = {}
        for dev_data_fname in dev_data_fname_list:
            name = dev_data_fname.split(".")[0]
            dev_data_fpath = os.path.join(data_dir, dev_data_fname)
            dev_dataset = SingleLabelTextClassifierDataset(dev_data_fpath, tokenizer, max_len, rdrop_switch=False, max_rows=dev_max_rows)
            dev_dataloader = DataLoader(dev_dataset, batch_size=dev_batch_size, shuffle=False, collate_fn=dev_dataset.collate_fn)
            dev_dataloader_dict[name] = dev_dataloader
        logger.info("dev_dataloader built success")

        self.model = model = BertClassificationModel(pretrained_model, pooling, classifier_dropout, self.num_labels)
        self.model.to(device)
        self.optimizer = optimizer = torch.optim.AdamW(model.parameters(), lr=lr)        
        self.scheduler = scheduler = LambdaLR(optimizer, lr_lambda)
        self.loss_func = loss_func = CategoricalCE()

        step = 1
        best_metric = 0
        best_ckp_fpath = None
        for epoch in range(1, train_epochs + 1):
            start = int(time.time())
            epoch_losses = []
            for batch_idx, batch_data in enumerate(tqdm(train_dataloader)):
                logits, targets = self.step(model, batch_data)
                
                loss_dict = self.loss_func(logits, targets)
                loss = loss_dict['loss']
                ce_loss = loss_dict['ce_loss']if 'ce_loss' in loss_dict else "nan"
                kl_loss = loss_dict['kl_loss'] if 'kl_loss' in loss_dict else "nan"

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                # 打印损失
                if step % log_step == 0:
                    logger.info(f"Epoch-{epoch}, step-{step}: lr: {scheduler.get_last_lr()[0]:.6f}, loss: {loss}, ce_loss: {ce_loss}, kl_loss: {kl_loss}")
                epoch_losses.append(loss)
                # 记录训练损失
                writer.add_scalar('Loss/train', loss.cpu().item(), step)
                
                # evaluate on dev
                if eval_step > 0 and step % eval_step == 0:
                    eval_metric = None
                    metric_dict = {}
                    for dev_name, dev_dataloader in dev_dataloader_dict.items():
                        tmp_metric_dict = self.evaluate(model, dev_dataloader, dev_name, epoch, step)
                        for k, v in tmp_metric_dict.items():
                            metric_dict[f"{dev_name}_{k}"] = v
                    
                    eval_metric = metric_dict["dev_macro_f1_score"]
                    if eval_metric and eval_metric > best_metric:
                        logger.info(f"Best metric {eval_metric} in epoch-{epoch}, step-{step}")
                        best_metric = eval_metric
                        # 记录超参数对应指标
                        writer.add_hparams(hparam_dict, metric_dict, run_name="best")

                        if epoch >= min_save_epoch:
                            ckp_fname = 'ckp-bsz-{}-lr-{}-epoch-{}.pt'.format(batch_size, lr, epoch)
                            ckp_fpath = os.path.join(output_dir, ckp_fname)
                            best_ckp_fpath = ckp_fpath
                            torch.save(model.state_dict(), ckp_fpath)
                            logger.info(f"saving checkpoints to {ckp_fpath}")
                step += 1
            
            avg_loss = sum(epoch_losses) / len(epoch_losses) 
            seconds = int(time.time()) - start
            logger.info(f"Epoch {epoch} finished, avg_loss: {avg_loss}, cost {seconds} seconds.")
            
            if epoch >= min_save_epoch and save_epoch > 0 and epoch % save_epoch == 0:
                ckp_fname = 'ckp-bsz-{}-lr-{}-epoch-{}.pt'.format(batch_size, lr, epoch)
                ckp_fpath = os.path.join(output_dir, ckp_fname)
                torch.save(model.state_dict(), ckp_fpath)
                logger.info(f"saving checkpoints to {ckp_fpath}")

        logger.info(f"Training finished, best_metric: {best_metric}, best_ckp_fpath: {best_ckp_fpath}")
        train_result_dict = {
            "output_dir": output_dir,
            "best_ckp_fpath": best_ckp_fpath
        }
        with open(os.path.join(output_dir, "train_result.json"), "w") as wr:
            wr.write(json.dumps(train_result_dict, ensure_ascii=False, indent=4))
        logger.info("Train finished, output info: \n" + json.dumps(train_result_dict, ensure_ascii=False, indent=4))
        return train_result_dict


if __name__ == "__main__":
    data_dir = "datasets/text_classify/thucnews"
    
    run_dt = datetime.now().strftime('%Y%m%d_%H%M')
    output_dir = f"output/thucnews/runtime_{run_dt}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    writer = SummaryWriter(log_dir=f"tensorboard/thucnews_runtime_{run_dt}")
    writer.add_text("hparams", to_markdown(hparam_dict))

    # 训练
    trainer = SingleLabelTextClassifierTrainer(data_dir)
    train_result_dict = trainer.train(output_dir)

