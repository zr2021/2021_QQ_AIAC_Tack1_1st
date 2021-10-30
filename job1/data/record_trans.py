#%%writefile data/record_trans.py
import numpy as np
from transformers import AutoTokenizer
import torch
from sklearn.preprocessing import MultiLabelBinarizer

class record_transform():
    def __init__(self, model_path, tag_file, 
                 get_title=True, get_frame=True,
                 get_tagid=False, get_vid=True, 
                 get_asr=False, get_category=False,
                 text_maxlen=36, frame_maxlen=32):
        self.get_title = get_title
        self.get_frame = get_frame
        self.get_tagid = get_tagid
        self.get_vid = get_vid
        self.get_asr = get_asr
        self.get_category = get_category
        self.text_maxlen = text_maxlen
        self.frame_maxlen = frame_maxlen
        
        # video padding frame
        self.zero_frame = np.zeros(1536).astype(dtype=np.float32)
        
        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # label encoder
        self.mlb = MultiLabelBinarizer()
        self.tag_list = set()
        with open(tag_file, encoding='utf-8') as fh:
            for line in fh:
                fields = line.strip().split('\t')
                self.tag_list.add(int(fields[0]))
        self.mlb.fit([self.tag_list])
        
    def transform(self, features, parse=True):
        """Prepare data"""
        if parse:
            features = self.parse_tfrecord(features)
            
        o = {}
        if self.get_title:
            o['id'] = torch.tensor(features['text_id'], dtype=torch.long)
            o['mask'] = torch.tensor(features['text_mask'], dtype=torch.long)
            
        if self.get_frame:
            # add frame features and mask
            o['frame_mask'] = torch.ones(self.frame_maxlen, dtype=torch.long)
            if len(features['frame_features']) != self.frame_maxlen:
                o['frame_mask'][len(features['frame_features']) - self.frame_maxlen:] = 0 

            frame_features_padding = self.padding_frames(features['frame_features'])
            o['frame_features'] = torch.tensor(frame_features_padding, dtype=torch.float32)
        
        if self.get_tagid:
            tags = features['tag_id']
            multi_hot = self.mlb.transform([tags])[0]
            o['target'] = torch.tensor(multi_hot, dtype=torch.float)
            
        if self.get_vid:
            o['vid'] = torch.tensor(int(features['id']), dtype=torch.long),
            
        return o  
    
    def parse_tfrecord(self, features):
        """Parse tfrecord"""
        o = {}  
        if self.get_title:
            titles = bytes(np.frombuffer(features['title'], dtype=np.uint8)).decode()
            title_id_mask = self.tokenizer.encode_plus(
                                ['', titles],
                                add_special_tokens=True,
                                max_length=self.text_maxlen,
                                padding='max_length',
                                truncation = True)
            o['text_id'] = title_id_mask['input_ids']# [CLS][SEP] Text [SEP]
            o['text_mask'] = title_id_mask['attention_mask']
            
        if self.get_frame:
            o['frame_features'] = [np.frombuffer(bytes(x), dtype=np.float16) for x in features['frame_feature']]
        
        if self.get_asr:
            o['asr_text'] = bytes(np.frombuffer(features['asr_text'], dtype=np.uint8)).decode()
            
        if self.get_tagid:
            o['tag_id'] = np.array(features['tag_id'], dtype=np.int32)
            o['tag_id'] = [t for t in o['tag_id'] if t in self.tag_list]
            
        if self.get_category:
            o['category_id'] = np.array(features['category_id'], dtype=np.int32)
            
        if self.get_vid:
            o['id'] = bytes(np.frombuffer(features['id'], dtype=np.uint8)).decode() 

        return o
    
    def padding_frames(self, frame_feature):
        """padding fram features"""
        num_frames = len(frame_feature)
        frame_gap = (num_frames - 1) / self.frame_maxlen
        if frame_gap <= 1:
            res = frame_feature + [self.zero_frame] * (self.frame_maxlen - num_frames)
        else:
            res = [frame_feature[round((i + 0.5) * frame_gap)] for i in range(self.frame_maxlen)]
        return np.c_[res]
