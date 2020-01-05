# -*- coding: utf-8 -*-
'''
 @File  : train.py.py
 @Author: ChangSiteng
 @Date  : 2020-01-05
 @Desc  : 
 '''

# import-path
import json
import sys
import os

import torch

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import jieba
import torch

from train_05_newword.new_word_3.utils import get_stopwords

from train_05_newword.new_word_3.run import run

'''
读取文件
因为后面情感分析用到 emoji\emotion\sentence_no_split，把这三个都读出来
'''
def get_data(isTrain,findtoken = None):

    # 读取test文本 sentences一维数组[（句子，情感，表情），（句子，情感，表情）...]
    sentences = []

    word_folder = "../../data/nlpcc2014/all_data/"
    if isTrain:
        path = word_folder + "train_data.json"
        data_for_token = []  # 只记录句子，为了token
    else:
        path = word_folder + "test_data.json"

    with open(path, 'r') as load_f:
        for line in load_f:
            dict = json.loads(line)
            json_data = {}
            json_data['sentence_no_emoji'] = dict['sentence_no_emoji']
            json_data['emotions'] = dict['emotions']
            json_data['emoji'] = dict['emoji']
            sentences.append(json_data)
            if isTrain:
                sentence = dict['sentence_no_emoji']
                word_list = [x for x in jieba.cut(sentence, cut_all=False) if x not in stopwords]
                data_for_token.append(word_list)

    print(data_for_token)
    if isTrain:
        result, add_word = run(data_for_token, 8)
    data = []
    for sentence in sentences:
        emoji = sentence['emoji']
        emotions = sentence['emotions']
        sentence = sentence['sentence_no_emoji'].strip()
        if len(sentence) > 0:
            characters = []
            for character in sentence:
                if character != ' ' and character != '':
                    characters.append(character)
            words = findtoken.cut_sentence(sentence)
            tags = []
            for word in words:
                # 判断 single —— s  Begin -b End-e  Medim-m
                length = len(word)
                if length <= 0:
                    print("异常word", word, "---", words)
                elif length == 1:
                    tags.append("s")
                elif length == 2:
                    tags.append("b")
                    tags.append("e")
                else:
                    tags.append("b")
                    for i in range(length - 2):
                        tags.append("m")
                    tags.append("e")
            json_data = {}
            json_data['sentence_no_emoji'] = sentence
            json_data['emotions'] = emotions
            json_data['emoji'] = emoji
            json_data['origin_split'] = words
            json_data['char_no_emoji'] = characters
            json_data['tags'] = tags
            data.append(json_data)
    return sentences, data, findtoken

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    stopwords = get_stopwords()
    # 读取文本
    train_sentences,training_data, findtoken = get_data(isTrain=True)
    # test_sentences,test_data,findtoken = get_data(False,findtoken)
