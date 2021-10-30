# 2021_QQ_AIAC_Tack1_1st
QQ浏览器2021AI算法大赛赛道一 第1名 方案

paper : <br> 

## 环境
python==3.7.10 <br> 
torch==1.7.1 <br> 
transformers==4.5.1 <br> 
pretrain 需要显存>=24GB 内存>=100GB <br> 

## 数据下载
(1) 视频数据集 <br> 
视频数据集在官网下载 https://algo.browser.qq.com/ <br> 
预期主办方会开源数据集，开源后会将地址补上 <br> 
下载后放到 ./input/data 文件夹 <br> 
tag_list 为标签的 top1w，官方 baseline 中提供，放到同一文件夹 <br> 

(2) 预训练模型 <br> 
预训练模型使用了 https://huggingface.co/hfl/chinese-roberta-wwm-ext-large <br> 
请使用 python3 -u download_pretrain_model.py 下载 <br> 

##  步骤代码
(1) 预训练 + finetune <br> 
脚本命令：sh train.sh <br> 
时间算力：单模在 1 张 a100 上大约需要 pretrain(2 day)，finetune(2 hour) <br> 
输出文件：每个单模的 checkpoint 保存在 jobN/model_finetune_1.pth <br> 
备注：各个单模间没有前后依赖关系，每个任务需要一张单卡，有多卡可以并行训练各个单模 <br> 

(2) 代码结构说明 <br> 
download_pretrain_model.py : 下载预训练模型的脚本 <br> 
ensemble.py : 融合的脚本 <br> 
job1-job6 : 六个模型训练任务，其文件结构完全一致，各 job 之间主要差别在预训练设置上 <br> 
注：job1在赛后额外补充了一些代码注释 <br> 
jobN/pretrain.py 预训练脚本 <br> 
jobN/finetune.py finetune脚本 <br> 
jobN/data 数据预处理部分，包含 dataset、mask token 等 <br> 
jobN/config 包含 pretrain 与 finetune 的一些超参配置 <br> 
jobN/qqmodel/qq_uni_model.py 模型定义 <br> 
 

## 简介
简要介绍的 ppt 请参考 Introduction.pdf <br> 

### 模型简介
多模态模型结构与参数量和 Bert-large 一致, <br> 
layer=24, hidden_size=1024, num_attention_heads=16。 <br> 
其输入为[CLS] Video_frame [SEP] Video_title [SEP]。<br> 
frame_feature 通过 fc 降维为 1024 维，与 text 的 emb 拼接。<br> 
Input_emb -> TransformerEncoder * 24 -> Pooling -> Fc -> Video_emb<br> 

### 预训练
预训练采用了 Tag classify, Mask language model, Mask frame model 三个任务<br> 

(1) Video tag classify 任务<br> 
tag 为人工标注的视频标签，pointwise 和 pairwise 数据集合中提供。<br> 
和官方提供的 baseline 一致，我们采用了出现频率前1w 的tag 做多标签分类任务。<br> 
Bert 最后一层的 [CLS] -> fc 得到 tag 的预测标签，与真实标签计算 BCE loss<br> 

(2) Mask language model 任务<br> 
与常见的自然语言处理 mlm 预训练方法相同，对 text 随机 15% 进行 mask，预测 mask 词。<br> 
多模态场景下，结合视频的信息预测 mask 词，可以有效融合多模态信息。<br> 

(3) Mask frame model 任务<br> 
对 frame 的随机 15% 进行 mask，mask 采用了全 0 的向量填充。<br> 
考虑到 frame 为连续的向量，难以类似于 mlm 做分类任务。<br> 
借鉴了对比学习思路，希望 mask 的预测帧在整个 batch 内的所有帧范围内与被 mask 的帧尽可能相似。<br> 
采用了 Nce loss，最大化 mask 帧和预测帧的互信息<br> 

(4) 多任务联合训练<br> 
预训练任务的 loss 采用了上述三个任务 loss 的加权和，<br>
L = L(tag) * 1250 / 3 + L(mlm) / 3.75 + L(mfm) / 9<br>
tag 梯度量级比较小，因此乘以了较大的权重。<br>
注：各任务合适的权重对下游 finetune 的效果影响比较大。<br>

(5) 预训练 Setting <br> 
初始化：bert 初始化权重来自于在中文语料预训练过的开源模型 https://huggingface.co/hfl/chinese-roberta-wwm-ext-large<br>
数据集：预训练使用了 pointwise 和 pairwise 集合，部分融合模型中加上了 test 集合（只有 mlm 和 mfm 任务）<br>
超参：batch_size=128, epoch=40, learning_rate=5e-5, scheduler=warmup_with_cos_decay, warum_ratio=0.06<br>
注：预训练更多的 epoch 对效果提升比较大，从10 epoch 提升至 20 epoch 对下游任务 finetune 效果提升显著。<br>


### Finetune
(1) 下游任务<br> 
视频 pair  分别通过 model 得到 256维 embedding，两个 embedding 的 cos 相似度与人工标注标签计算 mse<br>

(2) Finetune header<br> 
实验中发现相似度任务中，使用 mean_pooling 或者 attention_pooling 聚合最后一层 emb 接 fc 层降维效果较好。<br>

(3) Label normalize<br> 
评估指标为 spearman，考查预测值和实际值 rank 之间的相关性，因此对人工标注 label 做了 rank 归一化。<br>
即 target = scipy.stats.rankdata(target, 'average')<br>

(4) Finetune Setting<br> 
数据集：训练集使用了 pairwise 中 (id1%5!=0) | (id2%5 !=0) 的部分约 6.5w，验证集使用了(id1%5==0) & (id2%5==0) 的部分约 2.5k<br>
超参：batch_size=32, epoch=10, learning_rate=1e-5, scheduler=warmup_with_cos_decay, warum_ratio=0.06<br>


## Ensemble
(1) 融合的方法<br> 
采用了 weighted concat -> svd 降维 方法进行融合。实验中发现这种方法降维效果折损较小。<br>
concat_vec = [np.sqrt(w1) * emb1, np.sqrt(w2) * emb2, np.sqrt(w3) * emb3 ...]<br>
svd_vec = SVD(concat_vec, 256)<br>

(2) 融合的模型<br> 
最终的提交融合了六个模型。
模型都使用了 bert-large 这种结构，均为迭代过程中产出的模型，各模型之间只有微小的 diff，各个模型加权权重均为 1/6。<br>
下面表格中列出了各模型的diff部分，验证集mse，验证集spearman<br>

| jobid | ensemble-weight | detail | val-spearman | val-mse |
| ---- | ---- | ---- | ---- | ---- |
| job1 | 1/6 | base | 0.886031 | 0.028813 |
| job2 | 1/6 | 预训练tag分类任务为mean_pooling+fc | 0.884257 | 0.029493 |
| job3 | 1/6 | 预训练任务无 mfm | 0.883843 | 0.029248 |
| job4 | 1/6 | 预训练数据为 (point + pair)shuf-40epoch => pair-5epoch | 0.885397 | 0.029059 |
| job5 | 1/6 | 预训练数据为 (point-shuf => pair-shuf => test-shuf)-32epoch | 0.885795 | 0.028866 |
| job6 | 1/6 | 预训练 mlm/mfm mask概率调整为25% | 0.886289 | 0.029039 |

(3) 单模型的效果与融合的效果<br> 
单模的测试集成绩约在 0.836<br>
融合两个模型在 0.845<br>
融合三个模型在 0.849<br>
融合五个模型在 0.852<br>
