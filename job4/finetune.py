#%%writefile finetune.py
import os, math, random, time, sys, gc,  sys, json
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
import scipy
import scipy.stats

from config.data_cfg import *
from config.model_cfg import *
from config.finetune_cfg import *
from data.record_trans import record_transform
from data.qq_dataset import QQDataset
from qqmodel.qq_uni_model import QQUniModel
from optim.create_optimizer import create_optimizer
from utils.eval_spearman import evaluate_emb_spearman
from utils.utils import set_random_seed

from tfrecord import tfrecord_loader
from tfrecord.torch.dataset import MultiTFRecordDataset, TFRecordDataset

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, ChainDataset
from transformers import AutoConfig
from transformers import get_cosine_schedule_with_warmup

from zipfile import ZIP_DEFLATED, ZipFile

gc.enable()
    
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"   
set_random_seed(SEED)

def get_pred_and_loss(model, item, compute_loss=True, pairwise=True):
    video_feature = item['frame_features'].to(DEVICE)
    input_ids = item['id'].to(DEVICE)
    attention_mask = item['mask'].to(DEVICE)
    video_mask = item['frame_mask'].to(DEVICE)

    pred1, emb1, _ = model(video_feature, video_mask, input_ids, attention_mask, target=None, task=[])
 
    cos_sim = None
    loss = None
    target = None
    
    if pairwise:
        video_feature = item['frame_features2'].to(DEVICE)
        input_ids = item['id2'].to(DEVICE)
        attention_mask = item['mask2'].to(DEVICE)
        video_mask = item['frame_mask2'].to(DEVICE)

        pred2, emb2, _ = model(video_feature, video_mask, input_ids, attention_mask, target=None, task=[])  
        cos_sim = torch.cosine_similarity(emb1, emb2)  
        if compute_loss:
            target = item['target'].to(DEVICE)
            loss = nn.MSELoss()(cos_sim.view(-1), target.view(-1))        
    return cos_sim, emb1, loss, target

def eval(model, data_loader, get_pred_and_loss, pairwise=True, 
         compute_loss=True, eval_max_num=99999):
    """Evaluates the |model| on |data_loader|"""
    model.eval()
    loss_l, emb_l, vid_l, pred_l, target_l = [], [], [], [], []

    with torch.no_grad():
        for batch_num, item in enumerate(data_loader):
            pred, emb, loss, target = get_pred_and_loss(model, item, compute_loss=compute_loss, pairwise=pairwise)
            
            if loss is not None:
                loss_l.append(loss.to("cpu"))
                
            if pred is not None:
                pred_l.append(pred.to("cpu").numpy())

            if target is not None:
                target_l.append(target.to("cpu").numpy())
                
            vid_l.append(item['vid'][0].numpy())
                
            emb_l += emb.to("cpu").tolist()
            
            if (batch_num + 1) * emb.shape[0] >= eval_max_num:
                break
            
    if len(vid_l) != 0:
        vid_l = np.concatenate(vid_l)
    if len(pred_l) != 0:
        pred_l = np.concatenate(pred_l)
    if len(target_l) != 0:
        target_l = np.concatenate(target_l)
        
    return np.mean(loss_l), np.array(emb_l), vid_l, pred_l, target_l

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
            pred, emb, loss, target = get_pred_and_loss(model, item, compute_loss=True)
            loss.backward()

            optimizer.step()
            if scheduler:
                scheduler.step()

            if step == 20 or (step % 500 == 0 and step > 0):
                elapsed_seconds = time.time() - start # Evaluate the model on val_loader.

                val_loss, emb, vidl, pred, target = eval(model, val_loader, get_pred_and_loss=get_pred_and_loss, eval_max_num=10000)
                val_spearman = scipy.stats.spearmanr(target, pred).correlation
                
                improve_str = ''
                if not best_val_loss or val_loss <= best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), model_path)
                    improve_str = f"|New best_val_loss={best_val_loss:6.4}"

                logging.info(f"Epoch={epoch + 1}/{num_epochs}|step={step:3}|val_spearman={val_spearman:6.4}|val_mse={val_loss:6.4}|time={elapsed_seconds:0.3}s" + improve_str)

                start = time.time()
            step += 1

    return best_val_loss
    
#########################################
# Train pairwise model
#########################################
logging.info("Start")
for fname in ['finetune', 'model', 'data']:
    logging.info('=' * 66)
    with open(f'config/{fname}_cfg.py') as f:
        logging.info(f"Config - {fname}:" + '\n' + f.read().strip())

logging.info(f"Model_type = {MODEL_TYPE}")

list_val_loss = []
trans = record_transform(model_path=BERT_PATH, 
                         tag_file=f'{DATA_PATH}/tag_list.txt')
label = pd.read_csv(f"{DATA_PATH}/pairwise/label.tsv", sep='\t', header=None, dtype={0:str, 1:str})

# normalize label
target = label[2]
target = scipy.stats.rankdata(target, 'average')
label[2] = (target - target.min()) / (target.max() - target.min())

pair_dataset = QQDataset([f"{DATA_PATH}/pairwise/pairwise.tfrecords"], 
                        record_transform=trans, desc=DESC_NOTAG, label_df=label, pairwise=True)
logging.info("Load pair dataset finish")

test_dataset = QQDataset([f"{DATA_PATH}/test_b/test_b.tfrecords"], trans, desc=DESC_NOTAG)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers = 4)
logging.info("Load test dataset finish")
    
for fold in range(1):
    logging.info('=' * 66)
    model_path = f"model_finetune_{fold + 1}.pth"
    
    pair_dataset.pairwise = True
    pair_dataset.label_df = label

    set_random_seed(SEED + fold)

    train_ix = label[(label[0].astype(int) % NUM_FOLDS != fold) | (label[1].astype(int) % NUM_FOLDS != fold)].index
    val_ix = label[(label[0].astype(int) % NUM_FOLDS == fold) & (label[1].astype(int) % NUM_FOLDS == fold)].index
    train_dataset = torch.utils.data.Subset(pair_dataset, train_ix)
    val_dataset = torch.utils.data.Subset(pair_dataset, val_ix)
    
    total_steps=NUM_EPOCHS * len(train_dataset) // BATCH_SIZE
    warmup_steps=int(WARMUP_RATIO * total_steps)
    logging.info(f"Fold={fold + 1}/{NUM_FOLDS} seed={SEED+fold} train_len={len(train_dataset)} val_len={len(val_dataset)}")
    logging.info(f'Total train steps={total_steps}, warmup steps={warmup_steps}')

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, drop_last=True, num_workers=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, drop_last=False, num_workers=4)

    # model
    model = QQUniModel(MODEL_CONFIG, bert_cfg_dict=BERT_CFG_DICT, model_path=BERT_PATH, task=['tag'])
    model.load_state_dict(torch.load(PRETRAIN_PATH), strict=False)
    model.to(DEVICE)

    # optimizer
    optimizer = create_optimizer(model, model_lr=LR, layerwise_learning_rate_decay=LR_LAYER_DECAY)

    # scheduler
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_training_steps=total_steps, num_warmup_steps=warmup_steps)
    
    # train
    val_loss = train(model, model_path, train_loader, val_loader, optimizer, 
                     get_pred_and_loss=get_pred_and_loss,
                     scheduler=scheduler, num_epochs=NUM_EPOCHS)
    model.load_state_dict(torch.load(model_path)) # load best model
    list_val_loss.append(val_loss)

    gc.collect()

    logging.info(f"Fold{fold} val_loss_list=" + str([round(kk, 6) for kk in list_val_loss]))

    logging.info(f"Val Cv={np.mean(list_val_loss):6.4} +- {np.std(list_val_loss):6.4}")
    logging.info(f"Train-{fold} finish")
    
    #########################################
    # Inference test dataset
    #########################################
    logging.info('Start test predict')
    _, emb_l, vid_l, _, _ = eval(model, test_loader , get_pred_and_loss, pairwise=False, compute_loss=False, eval_max_num=99999)
    logging.info('Predict test finish')

    vid_embedding={}
    for vid, embedding in zip(vid_l.astype(str), emb_l.astype(np.float16)):
        vid_embedding[vid] = embedding.tolist()

    # Save result json
    logging.info('Save test results')
    with open(f'result_{fold}.json', 'w') as f:
        json.dump(vid_embedding, f)

    # Zip
    logging.info('Start test zip')
    with ZipFile(f'result_{fold}.zip', 'w', compression=ZIP_DEFLATED) as zip_file:
        zip_file.write(f'result_{fold}.json')

    #########################################
    # Inference validate dataset
    #########################################
    logging.info('Start valid predict')
    pair_dataset.pairwise = False
    label_valid = label[(label[0].astype(int) % NUM_FOLDS == fold) & (label[1].astype(int) % NUM_FOLDS == fold)]
    label_valid = pd.DataFrame({0: list(set(list(label_valid[0]) + list(label_valid[1])))})
    pair_dataset.label_df = label_valid
    logging.info(f'Valid predict shape={label_valid.shape}')

    val_loader = torch.utils.data.DataLoader(pair_dataset, batch_size=BATCH_SIZE, num_workers = 4)

    _, emb_l, vid_l, _, _ = eval(model, val_loader, get_pred_and_loss, pairwise=False, compute_loss=False, eval_max_num=99999)
    logging.info('Predict valid finish')

    vid_embedding={}
    for vid, embedding in zip(vid_l.astype(str), emb_l.astype(np.float16)):
        vid_embedding[vid] = embedding.tolist()

    # Save result json
    logging.info('Save valid results')
    with open(f'result_valid_{fold}.json', 'w') as f:
        json.dump(vid_embedding, f)
        
    logging.info(f'fold-{fold} finish')

logging.info('Finish')
