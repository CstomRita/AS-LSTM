# -*- coding: utf-8 -*-
# file: data_util_def.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from data_utils.tokenizer import Tokenizer
from data_utils.nlpcc_parse import *


# 创建非bert的[word,index]映射
def build_tokenizer(fnames, max_seq_len, dat_fname):
    # 如果存在分词模型，加载
    if os.path.exists(dat_fname):
        print('loading tokenizer:', dat_fname)
        tokenizer = pickle.load(open(dat_fname, 'rb'))
    # 否则重新分词，并将切分的结果和出现次数存储到一个dat_fname文件中
    else:
        text = ''
        parse = Parse()
        # fnames：是一个存储数据集地址的数组
        for fname in fnames:
            sentences = parse.parse(fname)['sentence']
            for sentence in sentences:
                # 这里是获取的每一句话，需要做分词的工作，我们再在这里做将表情符号切分的工作
                print(sentence)
        tokenizer = Tokenizer(max_seq_len)
        tokenizer.fit_on_text(text)

        # 把结果保存到一个文件中
        pickle.dump(tokenizer, open(dat_fname, 'wb'))
    return tokenizer

# 创建词嵌入
def build_embedding_matrix(word2idx, embed_dim, dat_fname):
    if os.path.exists(dat_fname):
        print('loading embedding_matrix:', dat_fname)
        embedding_matrix = pickle.load(open(dat_fname, 'rb'))
    else:
        print('loading word vectors...')
        embedding_matrix = np.zeros((len(word2idx) + 2, embed_dim))  # idx 0 and len(word2idx)+1 are all-zeros
        fname = './glove.twitter.27B/glove.twitter.27B.' + str(embed_dim) + 'd.txt' \
            if embed_dim != 300 else './glove.42B.300d.txt'
        word_vec = _load_word_vec(fname, word2idx=word2idx)
        print('building embedding_matrix:', dat_fname)
        for word, i in word2idx.items():
            vec = word_vec.get(word)
            if vec is not None:
                # words not found in embedding index will be all-zeros.
                 embedding_matrix[i] = vec
        pickle.dump(embedding_matrix, open(dat_fname, 'wb'))
    return embedding_matrix

def _load_word_vec(path, word2idx=None):
    fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    word_vec = {}
    for line in fin:
        tokens = line.rstrip().split()
        if word2idx is None or tokens[0] in word2idx.keys():
            word_vec[tokens[0]] = np.asarray(tokens[1:], dtype='float32')
    return word_vec


def pad_and_truncate(sequence, maxlen, dtype='int64', padding='post', truncating='post', value=0):
    x = (np.ones(maxlen) * value).astype(dtype)
    if truncating == 'pre':
        trunc = sequence[-maxlen:]
    else:
        trunc = sequence[:maxlen]
    trunc = np.asarray(trunc, dtype=dtype)
    if padding == 'post':
        x[:len(trunc)] = trunc
    else:
        x[-len(trunc):] = trunc
    return x



