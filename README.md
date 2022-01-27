# EMMA
## Introduction

 Existing emotion-aware conversational models usually focus on controlling the response contents to align with a specific emotion class, whereas empathy is the ability to understand and concern the feelings and experience of others. Hence, it is critical to learn the causes that evoke the users' emotion for empathetic responding, a.k.a. emotion causes. To gather emotion causes in online environments, we leverage counseling strategies and develop an empathetic chatbot to utilize the causal emotion information. On a real-world online dataset, we verify the effectiveness of the proposed approach by comparing our chatbot with several SOTA methods using automatic metrics, expert-based human judgements as well as user-based online evaluation.   The paper "Towards an Online Empathetic Chatbot with Emotion Causes" has been accepted by The 44th International ACM SIGIR Conference on Research and Development in Information Retrieval. For details,  https://doi.org/10.1145/3404835.3463042

## Dataset

X-EMAC: XiaoAI Empathetic Conversation dataset


## Dataset Overview

In version 1.0, we only released q-a pair. In version 2.0, in addition to expanding the data size, we added additional information such as emotion_label, keywords, reply intent, etc.

## Dependencies

* Python3
* torch == 1.4.0
* pytorch-ignite==0.2.1
* transformers==2.1.1
* tensorboardX==1.8
* tensorflow ==1.14.0

## Quick Start

### TrainData

Query\tResponse\tQueryEmotionLabel\tResponseEmotionLabel


### TestData

prev Query|prev Response|Query+EmotionLabel+EmotionCause，Response




### Train


```shell
CUDA_VISIBLE_DEVICES=0,1,2 python train.py --pretrained --model_checkpoint ./pretrained_models/ --data_path ../data/xiaoai_train.txt --scheduler linear 
```
### Test


```shell
CUDA_VISIBLE_DEVICES=0,1,2 python infer_moban.py \
          --datapath ../data/test_dialog.json \
          --out_path test_dialog \
          --model_checkpoint  model_checkpoint/ \
          --max_length 20 \
          --min_length 1
```

## LICENSE
Licensed under either of
* MIT license
* BSD license

# EMMA
## 介绍

现有的共情对话模型通常聚焦在控制模型生成特定情绪的回复，没有考虑到把情绪原因应用到共情对话生成。 但实际上，理解、关注他人感受，了解用户产生情绪的原因至关重要。我们采取了多种策略获取线上用户真实的情绪原因，并且开发了一个具有共情能力的对话机器人。在真实的场景中，我们与SOTA方法进行了对比，通过自动化指标、人工指标验证了我们提出的方法的有效性。该成果的论文《Towards an Online Empathetic Chatbot with Emotion Causes》已被国际人工智能方向顶级会议 The 44th International ACM SIGIR Conference on Research and Development in Information Retrieval 接收。详细见：https://doi.org/10.1145/3404835.3463042

## 数据集
X-EMAC: 小爱共情对话数据集 

## 数据集概况

在1.0版本中，我们仅开发了q-a pair 信息，在2.0版本除了扩大了数据集的规模外， 还增加了query的情绪类别、情绪原因、回复的意图类别等标注信息。

## 依赖
* Python3
* torch == 1.4.0
* pytorch-ignite==0.2.1
* transformers==2.1.1
* tensorboardX==1.8
* tensorflow ==1.14.0

## 快速开始

### 训练语料格式

Query\tResponse\tQueryEmotionLabel\tResponseEmotionLabel


### 测试语料格式

prev Query|prev Response|Query+EmotionLabel+EmotionCause，Response




### 训练


```shell
CUDA_VISIBLE_DEVICES=0,1,2 python train.py --pretrained --model_checkpoint ./pretrained_models/ --data_path ../data/xiaoai_train.txt --scheduler linear 
```
### 测试


```shell
CUDA_VISIBLE_DEVICES=0,1,2 python infer_moban.py \
          --datapath ../data/test_dialog.json \
          --out_path test_dialog \
          --model_checkpoint  model_checkpoint/ \
          --max_length 20 \
          --min_length 1
```
## LICENSE
Licensed under either of
* MIT license
* BSD license
