# -*- coding: utf-8 -*-
'''
 @File  : call.py
 @Author: ChangSiteng
 @Date  : 2020-01-28
 @Desc  : 向外提供调用接口
 '''

# import-path
import json
import sys
import os
import time

import torch

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
from data_utils.sentence_split import SentenceSplit
from train_04_emojiupdate1.word_and_emoji_embedding import sentence_no_emoji_split_tokenizer

'''
这个base_call_path有可能更改
hottopic存在本地可以
topic下的文件由于需要情感分析，建议爬虫之后上传到Gpu服务器，base_call_path变为gpu服务器的基本地址
gpu读文件、情感分析结果，写入gpu上的文件，后端获取的是return值
'''

with open("../data/nlpcc2014/emotion_label.json", "r") as load_f:
    emotion_labels = json.load(load_f)

def get_data(topic):
    data_path = "../data/calldata/" + topic + "/call.txt"
    analysisPath = "../data/calldata/" + topic + "/call_result.txt"
    countPath = "../data/calldata/" + topic + "/call_result_count.json"
    split = SentenceSplit()
    test_datas = []
    with open(data_path, "r") as f:
        datas = f.readlines()

    for data in datas:
        emoji_split_sentence_num, sentence_no_emoji, emoji_list \
            , emoji_count, sentence_no_emoji_split, emoji_all_count, emoji_all_type_count \
            = split.split(sentence=data, emoji_all_count=0, emoji_all_type_count=0)
        '拿 sentence_no_emoji_split 和 emoji_list'
        example = {}
        example['origin_sentence'] = data
        example['sentence_no_emoji'] = sentence_no_emoji
        example['emoji'] = (emoji_list)
        example['emoji_count'] = (emoji_count)
        example['sentence_no_emoji_split'] = sentence_no_emoji_split_tokenizer(sentence_no_emoji_split.strip())
        test_datas.append(example)
    print(topic,'--call.txt文件中共有数据：',len(test_datas),'条')
    return test_datas

def call_test(model,model_path,test_datas,device,topic):
    '''
       至此test_datas中完成表情符分离、子句分离和分词
       '''
    '''
    词向量获取和结果放入model中
    cd 对应的文件夹，python train.py x x 0 --->获取到对应的modelpath和dataPath
    isCall为0时true时，直接走call_testh函数，直接写入txt文件中，Path从datapath中提取
    '''
    data_path = "../data/calldata/" + topic + "/call.txt"
    analysisPath = "../data/calldata/" + topic + "/call_result.txt"
    countPath = "../data/calldata/" + topic + "/call_result_count.json"
    print('开始调用模型')
    model.load_state_dict(torch.load(model_path))
    print(f'\t-----calling-------')
    model.eval()
    '''
    to device 一定要写，否则用的是CPU
    '''
    model.to(device)
    count = {}
    new_dict = {v: k for k, v in emotion_labels.items()}
    with torch.no_grad():
        for example in test_datas:
            sentence = example['sentence_no_emoji_split']
            if len(sentence) == 0: continue  # 这里是因为切出的句子，有的没有汉字，只有表情，当前没有加表情，使用此方法过滤一下
            emoji = example['emoji']
            outputs = model(sentences=sentence, all_emojis=emoji, device=device)
            predictions = torch.argmax(outputs, dim=1)[0].item()
            if predictions is None:
                print(f'{sentence}|{emoji}')
            else:
                if new_dict[predictions] in count:
                    count[new_dict[predictions]] = count[new_dict[predictions]] + 1
                else:
                    count[new_dict[predictions]] = 1
            with open(analysisPath,'a') as f:
                f.write(example['origin_sentence']+"\t"+new_dict[predictions])
                f.write('\n')

    with open(countPath, 'w+') as f:
        encode_json = json.dumps(count)
        print(encode_json, file=f)


