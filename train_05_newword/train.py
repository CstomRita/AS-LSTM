# -*- coding: utf-8 -*-
'''
 @File  : train.py
 @Author: ChangSiteng
 @Date  : 2019-12-27
 @Desc  : 
 '''

# import-path
import json
import sys
import os

import torch
from torch import optim
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)


from train_05_newword.new_word_1.find_new_word_onJieba import FindNewTokenOnJieba
from train_05_newword.new_word_2.crf import BiLSTM_CRF


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


if __name__ == '__main__':

    # 读取train文本 sentences一维数组[句子，句子，句子]
    sentences = []
    train_word_folder = "../data/nlpcc2014/all_data/"
    path = train_word_folder + "train_data.json"
    with open(path, 'r') as load_f:
        for line in load_f:
            dict = json.loads(line)
            sentences.append(dict['sentence'])

    findtoken = FindNewTokenOnJieba(sentences=sentences)

    '''
    CRf部分
    '''
    START_TAG = "<START>"
    STOP_TAG = "<STOP>"
    EMBEDDING_DIM = 5
    HIDDEN_DIM = 4

    # Make up some training data
    # [([词，词],[tag tag])],([]),([])
    # 实现自动标注
    training_data = []
    for sentence in sentences:
        characters = []
        for character in sentence:
            characters.append(character)
        words = findtoken.cut_sentence(sentence)
        tags = []
        for word in words:
            # 判断 single —— s  Begin -b End-e  Medim-m
            length = len(word)
            if length == 1 :
                tags.append("s")
            elif length == 2:
                tags.append("b")
                tags.append("e")
            else :
                tags.append("b")
                for i in range(1,length):
                    tags.append("m")
                tags.append("e")
        training_data.append((characters,tags))
    print(training_data)


    # # 训练字向量
    word_to_ix = {}
    for sentence, tags in training_data:
        for word in sentence:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
    # print(word_to_ix) # word_to_ix记录非重复的词

    tag_to_ix = {"s": 0, "b": 1, "e": 2, "m" : 3,START_TAG: 4, STOP_TAG: 5}  # 标签
    #
    model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

    # Check predictions before training
    with torch.no_grad():
        precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
        precheck_tags = torch.tensor([tag_to_ix[t] for t in training_data[0][1]], dtype=torch.long)
        print(model(precheck_sent))

    # Make sure prepare_sequence from earlier in the LSTM section is loaded
    for epoch in range(300):  # again, normally you would NOT do 300 epochs, it is toy data
        for sentence, tags in training_data:
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            # Step 2. Get our inputs ready for the network, that is,
            # turn them into Tensors of word indices.
            sentence_in = prepare_sequence(sentence, word_to_ix)
            targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)

            # Step 3. Run our forward pass.
            loss = model.neg_log_likelihood(sentence_in, targets)

            # Step 4. Compute the loss, gradients, and update the parameters by
            # calling optimizer.step()
            loss.backward()
            optimizer.step()

    # Check predictions after training
    with torch.no_grad():
        precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
        print(model(precheck_sent))