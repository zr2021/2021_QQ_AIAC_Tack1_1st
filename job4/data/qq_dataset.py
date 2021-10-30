#%%writefile data/qq_dataset.py
import pandas as pd
import torch
from tfrecord import tfrecord_loader

class QQDataset(torch.utils.data.Dataset):
    def __init__(self, tfrecord_path_list, record_transform, desc, label_df=None, pairwise=False, black_id_list=set([])): 
        self.output_dict = {}
        self.label_df = label_df

        self.record_transform = record_transform
        self.pairwise = pairwise
             
        for tfrecord_path in tfrecord_path_list:
            tfdesc = desc
            
            if 'test' in tfrecord_path: # test 中不包含 tag_id 和 category_id
                tfdesc = {k:desc[k] for k in desc if k not in ['tag_id', 'category_id']}
                
            for record in tfrecord_loader(tfrecord_path, None, tfdesc):
                features = self.record_transform.parse_tfrecord(record)
                vid = features['id']
                
                if vid in black_id_list:
                    continue
                    
                self.output_dict[vid] = features
                
                #if len(self.output_dict) > 30000:
                    #break
                
        if self.label_df is None:
            self.label_df = pd.DataFrame({0: list(self.output_dict)})
 
    def __len__(self):
        return self.label_df.shape[0]
    
    def __getitem__(self, index):
        row = self.label_df.iloc[index]
        id_1 = row[0]
        o  = self.record_transform.transform(self.output_dict[id_1], parse=False)
        
        if self.pairwise:
            id_2, label = row[1], row[2]
            o2 = self.record_transform.transform(self.output_dict[id_2], parse=False)
            o['frame_features2'] = o2['frame_features']
            o['id2'] = o2['id']
            o['mask2'] = o2['mask']
            o['frame_mask2'] = o2['frame_mask']
            o['target'] = torch.tensor(label, dtype=torch.float32)
        return o
