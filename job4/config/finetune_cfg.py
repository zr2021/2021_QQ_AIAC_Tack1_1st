# Finetune config
# Pretrain model path
PRETRAIN_PATH='./model_pretrain_1.pth'
# Training params
NUM_FOLDS = 5
SEED = 2021
BATCH_SIZE = 32
NUM_EPOCHS = 10
WARMUP_RATIO = 0.06
REINIT_LAYER = 0
WEIGHT_DECAY = 0.01
LR = {'others':5e-5, 'nextvlad':5e-5, 'roberta':1e-5}
LR_LAYER_DECAY = 0.975
