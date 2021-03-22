import glob
import logging
import os
import random
import argparse
from tqdm import tqdm

import numpy as np

import torch
from torchvision.transforms import transforms
from torch.utils.data import Dataset,DataLoader,DistributedSampler,TensorDataset,RandomSampler
from torch import nn
from torch import distributed as dist
import torch.nn.functional as F

from transformers import (
    WEIGHTS_NAME,
    BertConfig,
    BertForMultipleChoice,
    BertJapaneseTokenizer,
    PreTrainedTokenizer,
    AdamW,
    get_linear_schedule_with_warmup,
)

from apex import amp

def synchronize():
    if not dist.is_available():
        return

    if not dist.is_initialized():
        return

    world_size = dist.get_world_size()

    if world_size == 1:
        return

    dist.barrier()


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
if __name__ == "__main__":
    seed_everything(42)

    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    
    rank = args.local_rank
    device = f"cuda:{rank}"
    torch.cuda.set_device(rank)
    
    
    lr = 5e-5
    epochs = 5
    batch_size = 1
    weight_decay = 1e-2
    multi_flag = False
    warmup_step = 0
    seq_len = 768
    drop_rate = 0.1
    gradient_accumurate = 8
    model_arch = "bert-base-v2"
    
    
    if model_arch == "bert-base-v1":
        path_name = "cl-tohoku/bert-base-japanese-whole-word-masking"
    elif model_arch == "bert-base-v2":
        path_name = "cl-tohoku/bert-base-japanese-v2"
    elif model_arch == "bert-large":
        path_name = "cl-tohoku/bert-large-japanese"
        
    args.local_rank = 0
    path = f"./params-mix-top1{rank}/"
    os.makedirs(path,exist_ok=True)
    
    if multi_flag:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()
        
    train_all = torch.load("../data/basev2-train_features-seq768-sorted_title-bm25_search-search_ver3.pt")
    dev1_all = torch.load("../data/basev2-dev1_features-seq768-sorted_title-bm25_search-search_ver3.pt")
    dev2_all = torch.load("../data/basev2-dev2_features-seq768-sorted_title-bm25_search-search_ver3.pt")        
    
    train_input_ids,train_input_mask,train_segment_ids,train_label_ids = train_all["train_input_ids"],train_all["train_input_mask"],train_all["train_segment_ids"],train_all["train_label_ids"]
    dev1_input_ids,dev1_input_mask,dev1_segment_ids,dev1_label_ids = dev1_all["dev1_input_ids"],dev1_all["dev1_input_mask"],dev1_all["dev1_segment_ids"],dev1_all["dev1_label_ids"]
    dev2_input_ids,dev2_input_mask,dev2_segment_ids,dev2_label_ids = dev2_all["dev2_input_ids"],dev2_all["dev2_input_mask"],dev2_all["dev2_segment_ids"],dev2_all["dev2_label_ids"]
    
    ##訓練データ+開発データ1+開発データ2で学習する場合
#     train_input_ids   = torch.cat((train_input_ids,dev1_input_ids,dev2_input_ids))
#     train_input_mask  = torch.cat((train_input_mask,dev1_input_mask,dev2_input_mask))
#     train_segment_ids = torch.cat((train_segment_ids,dev1_segment_ids,dev2_segment_ids))
#     train_label_ids   = torch.cat((train_label_ids,dev1_label_ids,dev2_label_ids))
    
    label_list = [f"{i}" for i in range(20)]
    num_labels = len(label_list)
    task_name = "jaqket"
    MODEL_CLASSES = {"bert": (BertConfig, BertForMultipleChoice, BertJapaneseTokenizer)}
    
    config_class, model_class, tokenizer_class = MODEL_CLASSES["bert"]
    bert_config = config_class.from_pretrained(path_name,num_labels=num_labels,finetuning_task=task_name,)
    tokenizer = tokenizer_class.from_pretrained(path_name)
    model = model_class.from_pretrained(path_name,config=bert_config)
    model.dropout = nn.Dropout(p=drop_rate)

    param = model.bert.embeddings.position_embeddings.weight.data
    param2 = F.interpolate(param.view(1,1,512,768),size=(768,768),mode='bicubic',align_corners=True)[0,0]
    model.bert.embeddings.position_embeddings.weight = nn.Parameter(param2)
    
    train_datasets = TensorDataset(train_input_ids,train_input_mask,train_segment_ids,train_label_ids)
    if multi_flag:
        train_sampler = DistributedSampler(train_datasets,shuffle=True)
    else:
        train_sampler = RandomSampler(train_datasets)
    train_dataloader = DataLoader(train_datasets, sampler=train_sampler, batch_size=batch_size,drop_last=True)

    dev1_datasets = TensorDataset(dev1_input_ids,dev1_input_mask,dev1_segment_ids,dev1_label_ids)
    dev1_dataloader = DataLoader(dev1_datasets, batch_size=batch_size)

    dev2_datasets = TensorDataset(dev2_input_ids,dev2_input_mask,dev2_segment_ids,dev2_label_ids)
    dev2_dataloader = DataLoader(dev2_datasets, batch_size=batch_size)
    

    one_iters = len(train_dataloader)
    one_iters = one_iters//gradient_accumurate
    num_iters = one_iters*epochs
    
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        }]

    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=num_iters)
    
    model = model.to(device,non_blocking=True)

    model, optimizer = amp.initialize(model, optimizer, opt_level="O2",verbosity=0)
    if multi_flag:
        model = nn.parallel.DistributedDataParallel(            
                model,
                device_ids=[rank],
                output_device=rank,
                # broadcast_buffers=False,
                # find_unused_parameters=True,
            )
        
    # scaler = torch.cuda.amp.GradScaler()
    step=1
    train_accs = []
    train_losses = []
    position_ids = torch.LongTensor([i for i in range(768)]).to(device)
    for step in range(num_iters):
        optimizer.zero_grad()
        model.zero_grad()
        for i in range(gradient_accumurate):
            batch = iter(train_dataloader).next()
            batch = tuple([b.to(device) for b in batch])
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "position_ids":position_ids,
                "labels": batch[3],
            }  
            true_label = batch[3]
            outputs = model(**inputs)
            loss = outputs["loss"]
            loss /= gradient_accumurate
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()

#         torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 1.0)
        optimizer.step()
        scheduler.step()
        
        temp = (true_label == outputs['logits'].argmax(dim=-1)).detach().cpu().numpy().tolist()
        train_accs.extend(temp)
        train_losses.append(loss.item())
        
        if (step%(one_iters//4)) == 0:
            if args.local_rank==0:
                with torch.no_grad():
                    accs1 = []
                    losses1 = []
                    model.eval()
                    for steps, batch in enumerate(dev1_dataloader):
                        batch = tuple(t.to(device) for t in batch)
                        inputs = {
                                    "input_ids": batch[0],
                                    "attention_mask": batch[1],
                                    "token_type_ids": batch[2],
                                    "position_ids":position_ids,
                                    "labels": batch[3],
                                }
                        outputs = model(**inputs)

                        loss = outputs["loss"]
                        losses1.append(loss.item())
                        
                        temp = (batch[3] == outputs['logits'].argmax(dim=-1)).detach().cpu().numpy().tolist()
                        accs1.extend(temp)
                    accs2 = []
                    losses2 = []
                    for steps, batch in enumerate(dev2_dataloader):
#                         model.eval()
                        batch = tuple(t.to(device) for t in batch)
                        inputs = {
                                    "input_ids": batch[0],
                                    "attention_mask": batch[1],
                                    "token_type_ids": batch[2],
                                    "position_ids":position_ids,
                                    "labels": batch[3],
                                }
                        outputs = model(**inputs)

                        loss = outputs["loss"]
                        losses2.append(loss.item())
                        
                        temp = (batch[3] == outputs['logits'].argmax(dim=-1)).detach().cpu().numpy().tolist()
                        accs2.extend(temp)
                train_accs = []
                train_losses = []
                model.train()
            if args.local_rank==0:
                torch.save(model.state_dict(),path+f"model.pt")