#%%writefile pretrain.py
import os, math, random, time, sys, gc,  sys, json, psutil
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import logging
from imp import reload
reload(logging)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler(f"train_{time.strftime('%m%d_%H%M', time.localtime())}.log"),
        logging.StreamHandler()
    ]
)

import numpy as np
import pandas as pd

from config.data_cfg import *
from config.model_cfg import *
from config.pretrain_cfg import *
from data.record_trans import record_transform
from data.qq_dataset import QQDataset
from qqmodel.qq_uni_model import QQUniModel
from optim.create_optimizer import create_optimizer
from utils.eval_spearman import evaluate_emb_spearman
from utils.utils import set_random_seed

from tfrecord.torch.dataset import MultiTFRecordDataset, TFRecordDataset

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, ChainDataset
from transformers import AutoConfig
from transformers import get_cosine_schedule_with_warmup

gc.enable()
    
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
               
set_random_seed(SEED)

def get_pred_and_loss(model, item, task=None):
    video_feature = item['frame_features'].to(DEVICE)
    input_ids = item['id'].to(DEVICE)
    attention_mask = item['mask'].to(DEVICE)
    video_mask = item['frame_mask'].to(DEVICE)
    
    target = None
    if 'target' in item:
        target = item['target'].to(DEVICE)
    
    pred, emb, loss = model(video_feature, video_mask, input_ids, attention_mask, target, task)
    return pred, emb, loss

def eval(model, data_loader, get_pred_and_loss, compute_loss=True, eval_max_num=99999):
    """Evaluates the |model| on |data_loader|"""
    model.eval()
    loss_l, emb_l, vid_l = [], [], []

    with torch.no_grad():
        for batch_num, item in enumerate(data_loader):
            pred, emb, loss = get_pred_and_loss(model, item, task='tag')
            
            if loss is not None:
                loss_l.append(loss.to("cpu"))
                
            emb_l += emb.to("cpu").tolist()
            
            vid_l.append(item['vid'][0].numpy())
            
            if (batch_num + 1) * emb.shape[0] >= eval_max_num:
                break
            
    return np.mean(loss_l), np.array(emb_l), np.concatenate(vid_l)

def train(model, model_path, 
          train_loader, val_loader, 
          optimizer, get_pred_and_loss, scheduler=None, 
          num_epochs=5):
    best_val_loss, best_epoch, step = None, 0, 0
    start = time.time()

    for epoch in range(num_epochs):
        for batch_num, item in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()
            pred, emb, loss = get_pred_and_loss(model, item)
            loss.backward()

            optimizer.step()
            if scheduler:
                scheduler.step()

            if step == 20 or (step % 500 == 0 and step > 0):
                elapsed_seconds = time.time() - start# Evaluate the model on val_loader.

                val_loss, emb, vid_l = eval(model, val_loader, get_pred_and_loss=get_pred_and_loss, eval_max_num=10000)

                improve_str = ''
                if not best_val_loss or val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), model_path)
                    improve_str = f"|New best_val_loss={best_val_loss:6.4}"

                logging.info(f"Epoch={epoch + 1}/{num_epochs}|step={step:3}|val_loss={val_loss:6.4}|time={elapsed_seconds:0.3}s" + improve_str)

                start = time.time()
            step += 1
        #model.load_state_dict(torch.load(model_path)) #Load best model
        torch.save(model.state_dict(), model_path)
        val_loss, emb, vid_l = eval(model, val_loader, get_pred_and_loss=get_pred_and_loss, eval_max_num=99999)
        label, spear = evaluate_emb_spearman(emb, vid_l, label_path=f"{DATA_PATH}/pairwise/label.tsv")
        logging.info(f"val_loss={val_loss} val_spearman={spear}")

    return best_val_loss

# Show config
logging.info("Start")
for fname in ['pretrain', 'model', 'data']:
    logging.info('=' * 66)
    with open(f'config/{fname}_cfg.py') as f:
        logging.info(f"Config - {fname}:" + '\n' + f.read().strip())
    
list_val_loss = []
logging.info(f"Model_type = {MODEL_TYPE}")
trans = record_transform(model_path=BERT_PATH, 
                         tag_file=f'{DATA_PATH}/tag_list.txt', 
                         get_tagid=True)

for fold in range(NUM_FOLDS):
    logging.info('=' * 66)
    model_path = f"model_pretrain_{fold + 1}.pth"
    logging.info(f"Fold={fold + 1}/{NUM_FOLDS} seed={SEED+fold}")
    
    set_random_seed(SEED + fold)

    # Load dataset
    logging.info("Load data into memory")
    m0 = psutil.Process(os.getpid()).memory_info()[0] / 2. ** 30
    train_dataset_list = [f"{DATA_PATH}/pointwise/pretrain_{ix}.tfrecords" for ix in range(PRETRAIN_FILE_NUM)] +\
                         [f"{DATA_PATH}/pairwise/pairwise.tfrecords"] #+ \
                         #[f"{DATA_PATH}/test_{ix}/test_{ix}.tfrecords" for ix in ['a', 'b']]

    train_dataset = QQDataset(train_dataset_list, trans, desc=DESC)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, drop_last=True, shuffle=True, num_workers=4)

    train_dataset_list2 = [f"{DATA_PATH}/pairwise/pairwise.tfrecords"]# + \
                          #[f"{DATA_PATH}/test_{ix}/test_{ix}.tfrecords" for ix in ['b']]

    train_dataset2 = QQDataset(train_dataset_list2, trans, desc=DESC)
    train_loader2 = DataLoader(train_dataset2, batch_size=BATCH_SIZE, drop_last=True, shuffle=True, num_workers=4)        

    val_dataset = QQDataset([f"{DATA_PATH}/pairwise/pairwise.tfrecords"], trans, desc=DESC)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, drop_last=False, num_workers=4)

    delta_mem = psutil.Process(os.getpid()).memory_info()[0] / 2. ** 30 - m0
    logging.info(f"Dataset used memory = {delta_mem:.1f}GB")

    FUTHER_EPOCHS = 5
    total_steps1 = NUM_EPOCHS * len(train_dataset) // BATCH_SIZE
    total_steps2 = FUTHER_EPOCHS * len(train_dataset2) // BATCH_SIZE
    
    warmup_steps1 = int(WARMUP_RATIO * total_steps1)
    warmup_steps2 = int(WARMUP_RATIO * total_steps2)
    logging.info(f'Total train steps1={total_steps1}, warmup steps1={warmup_steps1}')
    logging.info(f'Total train steps2={total_steps2}, warmup steps2={warmup_steps2}')

    # model
    model = QQUniModel(MODEL_CONFIG, bert_cfg_dict=BERT_CFG_DICT, model_path=BERT_PATH, task=PRETRAIN_TASK)
    model.to(DEVICE)

    # optimizer
    optimizer = create_optimizer(model, model_lr=LR, layerwise_learning_rate_decay=LR_LAYER_DECAY)

    # schedueler
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_training_steps=total_steps1, num_warmup_steps=warmup_steps1)
    val_loss = train(model, model_path, train_loader, val_loader, optimizer, 
                     get_pred_and_loss=get_pred_and_loss,
                     scheduler=scheduler, num_epochs=NUM_EPOCHS)
    
    scheduler2 = get_cosine_schedule_with_warmup(optimizer, num_training_steps=total_steps2, num_warmup_steps=warmup_steps2)
    val_loss = train(model, model_path, train_loader2, val_loader, optimizer, 
                     get_pred_and_loss=get_pred_and_loss,
                     scheduler=scheduler2, num_epochs=FUTHER_EPOCHS)
    list_val_loss.append(val_loss)
    
    del train_dataset, val_dataset
    gc.collect()

    logging.info(f"Fold{fold} val_loss_list=" + str([round(kk, 6) for kk in list_val_loss]))

logging.info(f"Val Cv={np.mean(list_val_loss):6.4} +- {np.std(list_val_loss):6.4}")
logging.info("Train finish")
