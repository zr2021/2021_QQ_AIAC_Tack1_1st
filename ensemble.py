import pandas as pd
import numpy as np
import scipy.stats
import scipy.spatial
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import TruncatedSVD, PCA
import json
from zipfile import ZIP_DEFLATED, ZipFile
import torch
import torch.nn

output_path = 'submit'
    
print('Ensemble start')

# Define ensemble list
print('=' * 55)
print('Ensemble list:')
jobdict_tuple = [
    (f'job1', 1/6, 'tag|mlm|mfm', 'Uni24L|addcls', '(point+pair shuffle)*40epoch', 0.886031, 0.028813),#job1
    (f'job2', 1/6, 'tag|mlm|mfm', 'Uni24L|nocls', '(point+pair shuffle)*40epoch', 0.884257, 0.029493), #job2
    (f'job3', 1/6, 'tag|mlm', 'Uni24L|nocls', '(point+pair shuffle)*40epoch', 0.883843, 0.029248),#job3
    (f'job4', 1/6, 'tag|mlm|mfm', 'Uni24L|addcls', '(point+pair shuffle)*40->pair*5', 0.885397, 0.029059),#job4
    (f'job5', 1/6, 'tag|mlm|mfm', 'Uni24L|addcls', '(point_shuf->pair_shuf->test_shuf)*32', 0.885795, 0.028866),#job5
    (f'job6', 1/6, 'tag|mlm0.25|mfm', 'Uni24L|addcls', '(point+pair shuffle)*40epoch', 0.886289, 0.029039),#job6
]
job_desc_df = pd.DataFrame(jobdict_tuple)
job_desc_df.columns = ['job_id', 'ensemble_weight', 'pretrain task', 'model', 'pretrain data', 'val_spearman', 'val_mse']
print(job_desc_df)

for stage in ['val', 'test']:
    print('=' * 55)
    if stage == 'val':
        print('Eval ensemble for val dataset')
        filename = 'result_valid'
    else:
        print('Ensemble for test dataset')
        filename = 'result'

    # Get job path
    jobdict = {}
    for item in jobdict_tuple:
        jobid, weight, fold_number = item[0], item[1], 1
        for ix in range(fold_number):
            w = 1 / len(jobdict_tuple)
            jobdict[f"{jobid}/{filename}_{ix}.json"] = w
    jobidlist = list(jobdict)
        
    # Read vec
    vid_df = None
    for jobid in jobidlist:
        with open(jobid) as f:
            content = f.read()
        vec_dict = json.loads(content)
        print(f"{jobid} | len={len(vec_dict)} | weight = 1/{int(1/jobdict[jobid])}")

        if vid_df is None:
            vid_list, vec_list = [], []
            for vid in vec_dict:
                vid_list.append(vid)
                vec_list.append([k for k in vec_dict[vid]])
                
            vid_df = pd.DataFrame({'vid':vid_list, jobid:vec_list})
        else:
            vid_df[jobid] = vid_df['vid'].apply(lambda x: [k for k in vec_dict[x]])

    # Concat vec and svd
    concat_list = []
    for jobid in jobidlist:
        weight = jobdict[jobid]
        
        norm_svd_vec = torch.nn.functional.normalize(torch.tensor(np.stack(vid_df[jobid])), p=2, dim=1)
        norm_svd_vec = np.array(norm_svd_vec)

        norm_svd_vec = np.sqrt(weight) * norm_svd_vec
        
        concat_list.append(norm_svd_vec)
        
    concat_vec = np.concatenate(concat_list, axis=1)
    print(f"Concat_vec shape={concat_vec.shape}")
    print(f"Start svd...")

    # Svd
    svd = TruncatedSVD(n_components=256, n_iter=50, random_state=66)
    svd.fit(concat_vec)
    svd_vec = svd.transform(concat_vec)

    print(f"Svd_vec shape = {svd_vec.shape}")
    print(f"Svd_explaine = {sum(svd.explained_variance_ratio_)}")

    if stage == 'test':
        print('Start save')
        # Save result
        vid_l = np.array(vid_df['vid']).astype(str)
        vid_embedding={}
        for vid, embedding in zip(vid_l.astype(str), svd_vec.astype(np.float16)):
            vid_embedding[vid] = embedding.tolist()

        # Save result json
        print('Save test results')
        with open(f"{output_path}/{filename}.json", 'w') as f:
            json.dump(vid_embedding, f)

        # Zip
        print('Start test zip')
        with ZipFile(f"{output_path}/submit.zip", 'w', compression=ZIP_DEFLATED) as zip_file:
            zip_file.write(f"{output_path}/{filename}.json")
        print(f'Save result => {output_path}/submit.zip')
    else:
	# Eval on local val dataset
        df = pd.read_csv('./input/data/pairwise/label.tsv', sep='\t', header=None, dtype={0:str, 1:str})
        idlist_set = set(list(df[0]) + list(df[1]))

        # Normliaze label
        target = df[2]
        target = scipy.stats.rankdata(target, 'average')
        df[2] = (target - target.min()) / (target.max() - target.min())
        df_bak = df.copy()
        val_ix = 0
        df_score = df_bak[(df_bak[0].astype(int) %5 == val_ix)&(df_bak[1].astype(int) % 5 == val_ix)].reset_index(drop=True)

        df_list = []
        for jobid in jobidlist + ['svd-blending', 'concat']:
            if jobid == 'svd-blending':
                vid_df['vec'] = list(svd_vec)
            elif jobid == 'concat':
                vid_df['vec'] = list(concat_vec)
            else:
                vid_df['vec'] = vid_df[jobid]

            df_score = pd.merge(df_score, vid_df[['vid', 'vec']], left_on=[0], right_on=['vid'], how='left')
            df_score = pd.merge(df_score, vid_df[['vid', 'vec']], left_on=[1], right_on=['vid'], how='left')
	    
            name = jobid#.split('/')[0]
	    
            df_score[name] = df_score.apply(lambda x: 1 - scipy.spatial.distance.cosine(x['vec_x'], x['vec_y']), axis = 1)

            spearman = scipy.stats.spearmanr(df_score[2], df_score[name]).correlation
            mse = mean_squared_error(df_score[name], df_score[2])

            df_list.append([jobid, spearman, mse])
            
            del df_score['vid_x']
            del df_score['vid_y']
            del df_score['vec_x']
            del df_score['vec_y']
        print(pd.DataFrame(df_list, columns=['jobid', 'val_spearman', 'val_mse']))
print('Finish')
