import os
import logging
import torch
import numpy as np
import time
from datetime import datetime
import json
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR
from transformers import BertTokenizerFast

from model import BertClassificationModel, InferenceModel
from dataset import get_label2id
from dataset import JsonlHierClassifyDataset
from loss import MulLabelCategoricalCE
from metric import HierarchicalClassifyMetric

#pretrained_model = "/home/work/pretrained_models/ernie-3.0-nano-zh"
#pretrained_model = "/home/work/pretrained_models/ernie-3.0-medium-zh"
#pretrained_model = "/home/work/pretrained_models/bert-small"
#pretrained_model = "/home/jiangjie/nogit/ailab-category-service/training/rebuild_tokenizer/output/bert-small-new"
pretrained_model = "/home/jiangjie/nogit/ailab-category-service/training/rebuild_tokenizer/output/ernie-3.0-nano-zh-new"
pooling = "cls"
classifier_dropout = 0.1
max_len = 64
device_name = "cuda:0"

lr = 1e-4
batch_size = 512
train_epochs = 40
log_step = 50
eval_step = 1000
save_epoch = 10
min_save_epoch = 40
# warmup_steps = 500
warmup_steps = 2000

train_data_fname = "train.jsonl"
qts_dev_fname = "qts_dev_natural.jsonl"
qts_dev_fname2 = "qts_dev_average.jsonl"

# test
# lr = 5e-5
# batch_size = 128
# train_epochs = 10
# log_step = 100
# eval_step = 50
# save_epoch = 100
# min_save_epoch = 100


def lr_lambda(current_step: int):
    global lr, warmup_steps
    if current_step < warmup_steps:
        # 线性增长，从 0 增长到 1
        return float(current_step) / float(warmup_steps)
    # warmup 后可以切换到其他策略，例如余弦退火
    return 1.0


class Trainer(object):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.label_fpath = data_dir + "/label2id.json"
        self.label2id = get_label2id(self.label_fpath)
        # 总层数
        self.num_levels = len(self.label2id)
        # 每一层标签数量
        self.level_num_labels = [len(self.label2id[k]) for k in range(self.num_levels)]
        # 所有标签数量
        self.num_labels = sum(self.level_num_labels)
        self.id2label = {}
        for level, level_label2id in self.label2id.items():
            self.id2label[level] = {v: k for k, v in level_label2id.items()}

    def train(self, output_dir):
        def evaluate(model, dev_dataloader):
            model.eval()
            with torch.no_grad():
                y_true_list = []
                y_pred_list = []
                eval_losses = []
                for batch_idx, batch_data in enumerate(tqdm(dev_dataloader)):
                    input_ids = batch_data['input_ids'].to(device)
                    attention_mask = batch_data['attention_mask'].to(device)
                    input_labels = batch_data['input_labels'].to(device)
                    bsz = input_ids.shape[0]
                    logits = model(input_ids, attention_mask)
                    loss = loss_func(input_labels, logits)
                    eval_losses.append(loss.cpu().item())

                    y_true = torch.nonzero(input_labels) 
                    y_true = y_true[:, 1].reshape((bsz, -1)).cpu().numpy().tolist()
                    y_true_list.extend(y_true)
                    
                    start_idx = 0
                    y_pred = []
                    for i in range(num_levels):
                        level_y_pred = logits[:, start_idx: start_idx + level_num_labels[i]].argmax(dim=1)
                        level_y_pred = level_y_pred + start_idx
                        level_y_pred = level_y_pred.reshape((bsz, -1)).cpu().numpy()
                        y_pred.append(level_y_pred)
                        start_idx += level_num_labels[i]
                    # batch_size * level_num
                    y_pred = np.concatenate(y_pred, axis=1)
                    y_pred_list.extend(y_pred.tolist())

                metric = HierarchicalClassifyMetric(y_true_list, y_pred_list, id2label)
                metric_dict = metric.get_metric_dict()
                acc_lst = [metric_dict[f"level_{k}"]["accuracy"] for k in range(1, 5)]
                avg_acc = sum(acc_lst) / len(acc_lst)
                metric_dict["avg_accuracy"] = avg_acc
                avg_loss = sum(eval_losses) / len(eval_losses)
                metric_dict["avg_loss"] = avg_loss
            return metric_dict

        def get_eval_metric(metric_dict):
            return metric_dict["level_4"]["macro avg"]["f1-score"]

        data_dir = self.data_dir
        device = torch.device(device_name)
        tokenizer = BertTokenizerFast.from_pretrained(pretrained_model)
        train_data_fpath = os.path.join(data_dir, train_data_fname)
        qts_dev_fpath = os.path.join(data_dir, qts_dev_fname)
        qts_dev_fpath2 = os.path.join(data_dir, qts_dev_fname2)
        
        train_dataset = JsonlHierClassifyDataset(train_data_fpath, self.label2id, tokenizer, max_len)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)
        logging.info("train_dataloader built success")

        dev_dataset = JsonlHierClassifyDataset(qts_dev_fpath, self.label2id, tokenizer, max_len)
        dev_dataloader1 = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=train_dataset.collate_fn)
        logging.info("dev_dataloader1 built success")

        dev_dataset2 = JsonlHierClassifyDataset(qts_dev_fpath2, self.label2id, tokenizer, max_len)
        dev_dataloader2 = DataLoader(dev_dataset2, batch_size=batch_size, shuffle=False, collate_fn=train_dataset.collate_fn)
        logging.info("dev_dataloader2 built success")

        num_levels = self.num_levels
        level_num_labels = self.level_num_labels
        id2label = self.id2label

        model = BertClassificationModel(pretrained_model, pooling, classifier_dropout, self.num_labels)
        model = model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)        
        scheduler = LambdaLR(optimizer, lr_lambda)
        loss_func = MulLabelCategoricalCE()

        step = 0
        best_metric = 0
        best_ckp_fpath = None
        for epoch in range(1, train_epochs + 1):
            start = int(time.time())
            epoch_losses = []
            for batch_idx, batch_data in enumerate(tqdm(train_dataloader)):
                input_ids = batch_data['input_ids'].to(device)
                attention_mask = batch_data['attention_mask'].to(device)
                input_labels = batch_data['input_labels'].to(device)
                logits = model(input_ids, attention_mask)
                loss = loss_func(input_labels, logits)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                # 打印损失
                if step % log_step == 0:
                    logging.info(f"loss: {loss}, in step-{step + 1} epoch-{epoch}")
                    logging.info(f"Step {step}: Learning rate {scheduler.get_last_lr()[0]:.6f}")
                epoch_losses.append(loss)
                # 记录训练损失
                writer.add_scalar('Loss/train', loss.cpu().item(), step)
                
                # evaluate on dev
                if eval_step > 0 and step % eval_step == 0:
                    metric_dict = evaluate(model, dev_dataloader1)
                    logging.info(f"################## EVAL RESULT FOR NATURAL QTS_DATA  EPOCH-{epoch}, STEP-{step} ###################")
                    logging.info("\n" + json.dumps(metric_dict, ensure_ascii=False, indent=4))
                    writer.add_scalar("Loss/eval", metric_dict["avg_loss"], step)
                    scalar_dict = {}
                    for i in range(1, 5):
                        scalar_dict[f'level{i}_acc'] = metric_dict[f"level_{i}"]["accuracy"]
                        scalar_dict[f'level{i}_f1'] = metric_dict[f"level_{i}"]["macro avg"]["f1-score"]
                    writer.add_scalars("Metric/QTS_Natural", scalar_dict, step)
                    
                    metric_dict = evaluate(model, dev_dataloader2)
                    logging.info(f"################## EVAL RESULT FOR AVERAGED QTS_DATA EPOCH-{epoch}, STEP-{step} ###################")
                    logging.info("\n" + json.dumps(metric_dict, ensure_ascii=False, indent=4))
                    writer.add_scalar("Loss/eval", metric_dict["avg_loss"], step)
                    scalar_dict = {}
                    for i in range(1, 5):
                        scalar_dict[f'level{i}_acc'] = metric_dict[f"level_{i}"]["accuracy"]
                        scalar_dict[f'level{i}_f1'] = metric_dict[f"level_{i}"]["macro avg"]["f1-score"]
                    writer.add_scalars("Metric/QTS_Average", scalar_dict, step)

                    eval_metric = get_eval_metric(metric_dict)
                    if eval_metric > best_metric:
                        logging.info(f"best metric {eval_metric} in epoch {epoch}")
                        best_metric = eval_metric
                        if epoch >= min_save_epoch:
                            ckp_fname = 'ckp-bsz-{}-lr-{}-epoch-{}.pt'.format(batch_size, lr, epoch)
                            ckp_fpath = os.path.join(output_dir, ckp_fname)
                            best_ckp_fpath = ckp_fpath
                            torch.save(model.state_dict(), ckp_fpath)
                            logging.info(f"saving checkpoints to {ckp_fpath}")
                step += 1

            avg_loss = sum(epoch_losses) / len(epoch_losses) 
            seconds = int(time.time()) - start
            logging.info(f"Epoch {epoch} finished, avg_loss: {avg_loss}, cost {seconds} seconds.")
            
            if epoch >= min_save_epoch and save_epoch > 0 and epoch % save_epoch == 0:
                ckp_fname = 'ckp-bsz-{}-lr-{}-epoch-{}.pt'.format(batch_size, lr, epoch)
                ckp_fpath = os.path.join(output_dir, ckp_fname)
                torch.save(model.state_dict(), ckp_fpath)
                logging.info(f"saving checkpoints to {ckp_fpath}")
            
        logging.info(f"Training finished, best_metric: {best_metric}, best_ckp_fpath: {best_ckp_fpath}")
        train_result_dict = {
            "output_dir": output_dir,
            "best_ckp_fpath": best_ckp_fpath,
        }
        return train_result_dict

    def evaluate_from_ckp(self, checkpoint_fpath, test_data_fpath):
        device = torch.device(device_name)
        tokenizer = BertTokenizerFast.from_pretrained(pretrained_model)

        test_dataset = JsonlHierClassifyDataset(test_data_fpath, self.label2id, tokenizer, max_len)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=test_dataset.collate_fn)
        logging.info("test_dataloader built success")

        num_levels = self.num_levels
        level_num_labels = self.level_num_labels
        id2label = self.id2label

        model = BertClassificationModel(pretrained_model, pooling, classifier_dropout, self.num_labels)
        model = model.to(device)
        model.load_state_dict(torch.load(checkpoint_fpath, map_location=device), strict=True)
        
        model.eval()
        y_true_list = []
        y_pred_list = []
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(tqdm(test_dataloader)):
                input_ids = batch_data['input_ids'].to(device)
                attention_mask = batch_data['attention_mask'].to(device)
                input_labels = batch_data['input_labels'].to(device)
                bsz = input_ids.shape[0]
                logits = model(input_ids, attention_mask)
                y_true = torch.nonzero(input_labels) 
                y_true = y_true[:, 1].reshape((bsz, -1)).cpu().numpy().tolist()
                y_true_list.extend(y_true)
                
                start_idx = 0
                y_pred = []
                for i in range(num_levels):
                    level_y_pred = logits[:, start_idx: start_idx + level_num_labels[i]].argmax(dim=1)
                    level_y_pred = level_y_pred + start_idx
                    level_y_pred = level_y_pred.reshape((bsz, -1)).cpu().numpy()
                    y_pred.append(level_y_pred)
                    start_idx += level_num_labels[i]
                # batch_size * level_num
                y_pred = np.concatenate(y_pred, axis=1)
                y_pred_list.extend(y_pred.tolist())
            
        metric = HierarchicalClassifyMetric(y_true_list, y_pred_list, id2label)
        metric_dict = metric.get_metric_dict()
        acc_lst = [metric_dict[f"level_{k}"]["accuracy"] for k in range(1, 5)]
        avg_acc = sum(acc_lst) / len(acc_lst)
        metric_dict["avg_accuracy"] = avg_acc
        logging.info("##################EVAL RESULT###################")
        logging.info("\n" + json.dumps(metric_dict, ensure_ascii=False, indent=4))
    
    def export(self, best_ckp_fpath, best_onnx_fpath):
        if os.path.exists(best_onnx_fpath):
            logging.info(f"Export failed, {best_onnx_fpath} already exists!")
            return
            
        device = torch.device("cpu")
        model = InferenceModel(pretrained_model, pooling, classifier_dropout, self.level_num_labels)
        model = model.to(device)
        logging.info(f"load checkpoints from {best_ckp_fpath}")
        model.load_state_dict(torch.load(best_ckp_fpath, map_location=device), strict=True)                                                            
        model.eval()
                                                                                                     
        query = "橡胶软连接 dn40 16p"  
        tokenizer = BertTokenizerFast.from_pretrained(pretrained_model)
        feed_dict = tokenizer(query, max_length=max_len, add_special_tokens=True,
                              padding='max_length', return_tensors='pt', truncation=True,
                              return_attention_mask=True, return_token_type_ids=True)
        
        input_names = ['input_ids', 'attention_mask', 'token_type_ids']
        dynamic_axes = {                                                                                
            'input_ids': {0: 'batch_size', 1: 'seq_len'},
            'attention_mask': {0: 'batch_size', 1: 'seq_len'},
            'token_type_ids': {0: 'batch_size', 1: 'seq_len'},
            'top_proba': {0: 'batch_size', 1: 'seq_len'},
            'top_indice': {0: 'batch_size', 1: 'seq_len'},
        }                                                                                               
        model_inputs = [feed_dict[k] for k in input_names]
        torch.onnx.export(                                                                          
            model,                                                                                  
            tuple(model_inputs),                                                                    
            f=best_onnx_fpath,                                                                
            input_names=input_names,                                                                
            output_names=['top_proba', 'top_indice'],
            dynamic_axes=dynamic_axes,                                                           
            do_constant_folding=True,                                                  
            opset_version=12                                                                     
        )
        logging.info(f"Export finished, onnx_fpath: {best_onnx_fpath}")


if __name__ == "__main__":
    from zkhutil.utils import log_util
    log_util.simple_init()

    data_dir = "jobs/v20241012_01"
    
    run_dt = datetime.now().strftime('%Y%m%d_%H%M')
    # outputdir
    output_dir = data_dir + f"/runtime_{run_dt}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # tensorboard writer
    writer = SummaryWriter(log_dir=f"logs/tensorboard/multilabel_{run_dt}")

    # 训练
    trainer = Trainer(data_dir)
    train_result_dict = trainer.train(output_dir)