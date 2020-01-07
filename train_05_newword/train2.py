# -*- coding: utf-8 -*-
'''
 @File  : train2.py.py
 @Author: ChangSiteng
 @Date  : 2020-01-05
 @Desc  : 
 '''

# import-path
import json
import sys
import os
import time

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import jieba
import torch


from train_05_newword.new_word_3.run import run
from train_05_newword.new_word_3.utils import get_stopwords
from torch import optim

from train_05_newword.new_word_2.crf import BiLSTM_CRF
from train_05_newword.train import prepare_sequence, get_result_word


split_symbol = "\/"
stopwords = get_stopwords()

dataFolder = "./data"+time.strftime('%Y-%m-%d',time.localtime())+"/"

def write_to_file(isTrain,folderpath,datas):
    write_time = 0
    if isTrain:
        path = folderpath + 'train.json'
    else :
        path = folderpath + 'test.json'
    with open(path, 'w+') as fw:
        for example_data in datas:
            encode_json = json.dumps(example_data)
            # 一行一行写入，并且采用print到文件的方式
            print(encode_json, file=fw)
            write_time += 1
    print("load data并保存在", path, ",写了", write_time, "次")
    if isTrain:
        # 将分好的词划分出来，拼接到一起，方便glove训练
        with open(folderpath+'words_origin_split.txt', 'w+') as fw:
            for example_data in datas:
                print(example_data['origin_split'], file=fw)
        with open(folderpath+'words_origin_crf_split.txt', 'w+') as fw:
            for example_data in datas:
                print(example_data['crf_split'], file=fw)

def write_jieba_split(folderpath,jiba_split,Trained):
    if Trained:
        with open(folderpath + 'new_jieba_split.txt', 'w+') as fw:
            print("训练添加词典后")
            for example_data in jiba_split:
                print([(x+'/') for x in jieba.cut(example_data, cut_all=False) if x not in stopwords and len(x.strip())>0], file=fw)
        print("分词TXT已经保存在new_jieba_split.txt中")
    else:
        with open(folderpath + 'jieba_split.txt', 'w+') as fw:
            print("训练添加词典前")
            for example_data in jiba_split:
                print([(x + '/') for x in jieba.cut(example_data, cut_all=False) if
                       x not in stopwords and len(x.strip()) > 0], file=fw)
        print("分词TXT已经保存在jieba_split.txt中")

'''
读取文件
因为后面情感分析用到 emoji\emotion\sentence_no_split，把这三个都读出来
'''
def get_data(isTrain,findtoken = None):

    # 读取test文本 sentences一维数组[（句子，情感，表情），（句子，情感，表情）...]
    sentences = []
    jiba_split = []

    word_folder = "../data/nlpcc2014/all_data/"
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
                jiba_split.append(sentence)

    if isTrain:
        write_jieba_split(dataFolder,jiba_split,Trained=False)
        result, add_word = run(data_for_token, topN=8,score=0.1)
        for word in add_word.keys():
            jieba.add_word(word)  #add_word保证添加的词语不会被cut掉
        write_jieba_split(dataFolder,jiba_split,Trained=True)
    data = []
    for sentence in sentences:
        emoji = sentence['emoji']
        emotions = sentence['emotions']
        sentence = sentence['sentence_no_emoji'].strip()
        if len(sentence) > 0:
            characters = []
            for character in sentence:
                if character != ' ' and character != '' and character != '\u3000': # 去除空格'\u3000' 中文空格
                    characters.append(character)
            words = [(x) for x in jieba.cut(sentence, cut_all=False)
                     if len(x.strip()) > 0]
            tags = []
            for word in words:
                # 判断 single —— s  Begin -b End-e  Medim-m
                length = len(word.strip())
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
            if (len(characters) != len(tags)):
                print("长度不相等", words, characters,tags,len(characters),"------",len(tags))
            data.append(json_data)
    return sentences, data, findtoken

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.mkdir(dataFolder)

    stopwords = get_stopwords()
    # 读取文本
    train_sentences,training_data, findtoken = get_data(isTrain=True)
    test_sentences,test_data,findtoken = get_data(False,findtoken)

    '''
       CRf部分
       '''
    START_TAG = "<START>"
    STOP_TAG = "<STOP>"
    EMBEDDING_DIM = 300  # input_size = embeddding_size
    HIDDEN_DIM = 128

    # 字向量
    word_to_ix = {}
    for example in training_data:
        sentence = example['char_no_emoji']
        for word in sentence:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
    # print(word_to_ix) # word_to_ix记录非重复的词

    tag_to_ix = {"s": 0, "b": 1, "e": 2, "m": 3, START_TAG: 4, STOP_TAG: 5}  # 标签

    model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

    # Check predictions before training
    # with torch.no_grad():
    #     precheck_sent = prepare_sequence(training_data[0]['char_no_emoji'], word_to_ix)
    #     precheck_tags = torch.tensor([tag_to_ix[t] for t in training_data[0]['tags']], dtype=torch.long).to(device)
    #     score, tag_seq = model(precheck_sent)
    #     print("原始", get_result_word(training_data[0]['char_no_emoji'], precheck_tags))
    #     print("训练前", tag_seq, "-----", get_result_word(training_data[0]['char_no_emoji'], tag_seq))

    # Make sure prepare_sequence from earlier in the LSTM section is loaded
    for epoch in range(20):  # again, normally you would NOT do 300 epochs, it is toy data
        print("######第",epoch + 1,"次CRF")
        for example in training_data:
            tags = example['tags']
            sentence = example['char_no_emoji']
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            # Step 2. Get our inputs ready for the network, that is,
            # turn them into Tensors of word indices.
            sentence_in = prepare_sequence(sentence, word_to_ix)  # 字向量
            targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long).to(
                device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

            # Step 3. Run our forward pass.
            loss, isException = model.neg_log_likelihood(sentence_in, targets)
            if isException:
                print(sentence, "-------", len(sentence))
                print(tags, "-------", len(tags))
            # Step 4. Compute the loss, gradients, and update the parameters by
            # calling optimizer.step()
            loss.backward()
            optimizer.step()
        print("总共traindata:", len(training_data))

    # 存储一下训练集的结果，对比加入crf后的改变
    for example in training_data:
        with torch.no_grad():
            precheck_sent = prepare_sequence(example['char_no_emoji'], word_to_ix)
            score, tag_seq = model(precheck_sent)
            example['crf_split'] = get_result_word(example['char_no_emoji'], tag_seq)
    write_to_file(True, dataFolder, training_data)

    # 存储模型
    # torch.save(model.state_dict(), "crf")

    # 跑测试集
    for example in test_data:
        with torch.no_grad():
            precheck_sent = prepare_sequence(example['char_no_emoji'], word_to_ix)
            score, tag_seq = model(precheck_sent)
            example['crf_split'] = get_result_word(example['char_no_emoji'], tag_seq)
    write_to_file(False,dataFolder , test_data)