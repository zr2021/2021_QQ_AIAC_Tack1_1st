MODEL_TYPE = 'uni'#'all', 'cross', 'frame', 'bi', 'uni'

MODEL_CONFIG = {
    'INPUT_SIZE': 1792,
    'HIDDEN_SIZE': 256,
    'NUM_CLASSES': 10000,
    'FEATURE_SIZE': 1536,
    'OUTPUT_SIZE': 1024,
    'EXPANSION_SIZE': 2,
    'CLUSTER_SIZE': 64,
    'NUM_GROUPS': 8,
    'DROPOUT_PROB': 0.2,
}

BERT_CFG_DICT = {}
BERT_CFG_DICT['uni'] = {
    'hidden_size':768,
    'num_hidden_layers':6,
    'num_attention_heads':12,
    'intermediate_size':3072,
    'hidden_dropout_prob':0.0,
    'attention_probs_dropout_prob':0.0
}
