from transformers import AdamW
def create_optimizer(model, model_lr={'others':5e-3, 'nextvlad':5e-4, 'roberta':5e-5},
                     weight_decay=0.01, layerwise_learning_rate_decay=0.975,
                     adam_epsilon=1e-6, use_bertadam = False):
    # Set learning_rates for each layers
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = []
    for layer_name in model_lr:
        lr = model_lr[layer_name]
#         #Layer decay learning rate
#         if layer_name == 'roberta':  # Robert 使用 layerwise_decay
#             layers =  [getattr(model, layer_name).embeddings] + list(getattr(model, layer_name).encoder.layer)
#             layers.reverse()
#             for layer in layers:
#                 lr *= layerwise_learning_rate_decay
#                 optimizer_grouped_parameters += [
#                     {
#                         "params": [p for n, p in layer.named_parameters() if not any(nd in n for nd in no_decay)],
#                         "weight_decay": weight_decay,
#                         "lr": lr,
#                     },
#                     {
#                         "params": [p for n, p in layer.named_parameters() if any(nd in n for nd in no_decay)],
#                         "weight_decay": 0.0,
#                         "lr": lr,
#                     },
#                 ]
        if layer_name != 'others':  # 设定了特定 lr 的 layer
             optimizer_grouped_parameters += [
                {
                    "params": [p for n, p in model.named_parameters() if (not any(nd in n for nd in no_decay) 
                                                                          and layer_name in n)],
                    "weight_decay": weight_decay,
                    "lr": lr,
                },
                {
                    "params": [p for n, p in model.named_parameters() if (any(nd in n for nd in no_decay) 
                                                                          and layer_name in n)],
                    "weight_decay": 0.0,
                    "lr": lr,
                },
             ]
        else:  # 其他，默认学习率
            optimizer_grouped_parameters += [
                {
                    "params": [p for n, p in model.named_parameters() if (not any(nd in n for nd in no_decay) 
                                                                          and not any(name in n for name in model_lr))],
                    "weight_decay": weight_decay,
                    "lr": lr,
                },
                {
                    "params": [p for n, p in model.named_parameters() if (any(nd in n for nd in no_decay) 
                                                                          and not any(name in n for name in model_lr))],
                    "weight_decay": 0.0,
                    "lr": lr,
                },
            ]
            
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=model_lr['roberta'],
        eps=adam_epsilon,
        correct_bias=not use_bertadam
    )
    return optimizer
