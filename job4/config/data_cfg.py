DATA_PATH = '../input/data'
BERT_PATH = '../input/pretrain-model/roberta-wwm-large'

DESC = {
    'tag_id':"int",
    'id': 'byte',
    'category_id': 'int',
    'title': 'byte',
    'asr_text': 'byte',
    'frame_feature': 'bytes'
}

DESC_NOTAG = {
    'id': 'byte',
    'title': 'byte',
    'asr_text': 'byte',
    'frame_feature': 'bytes'
}
