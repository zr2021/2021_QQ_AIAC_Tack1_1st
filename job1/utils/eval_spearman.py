import numpy as np
import pandas as pd
import torch
import scipy

def evaluate_emb_spearman(emb, vid_l, label_path):
    """Evaluate val dataset spearman"""
    vid_df = pd.DataFrame({'id': vid_l})
    vid_df['id'] = vid_df['id'].astype(str)
    vid_df['index'] = vid_df.index

    # get label
    label = pd.read_csv(label_path, sep='\t', header=None, dtype={0: str, 1: str})
    label = pd.merge(label, vid_df, left_on=[0], right_on=['id'], how='left')
    label = pd.merge(label, vid_df, left_on=[1], right_on=['id'], how='left')
    label['emb_sim'] = calc_emb_sim(emb, list(label['index_x']), list(label['index_y']))
    spear = scipy.stats.spearmanr(label[2], label['emb_sim']).correlation
    return label, spear

def calc_emb_sim(query, query_index, candidate_index, bs = 1024):
    """Calc pair emb-sim on gpu"""
    n = len(query_index)
    n_batch = n // bs + 1

    query = torch.tensor(query).cuda()

    emb_sim_list = []
    for i in range(n_batch):
        left = bs * i
        right = bs * (i+1)
        if i == n_batch - 1:
            right = n

        emb_left = query[query_index[left:right]].cuda()
        emb_right = query[candidate_index[left:right]].cuda()

        emb_sim = torch.mul(emb_left, emb_right).sum(axis=1)
        emb_sim_list.append(emb_sim.cpu().numpy())
    return np.concatenate(emb_sim_list)
