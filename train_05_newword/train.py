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

import jieba
import torch
from torch import optim


curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from train_05_newword.new_word_1.find_new_word_onJieba2 import FindNewTokenOnJieba2

from train_05_newword.new_word_1.find_new_word_onJieba import FindNewTokenOnJieba
from train_05_newword.new_word_2.crf import BiLSTM_CRF

# 将分词结果存储到train.json中的no_emoji_split中
# 训练集还需要words_origin.txt中

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
        with open(folderpath+'words_origin_jieba.txt', 'w+') as fw:
            for example_data in datas:
                print("\/".join(jieba.cut(example_data['sentence_no_emoji'])).split("\/"), file=fw)
        print("分词TXT已经保存在words_origin_*.txt中")

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long).to(device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))


def get_result_word(sentence,result):
    words = []
    word = ""
    for index,character in enumerate(sentence):
        word += character
        if result[index] == 0 or result[index] == 2:
            words.append(word)
            word = ""
        elif result[index] == 4 or result[index] == 5:
            print("出现start-stop标注")
    return words

'''
读取文件
因为后面情感分析用到 emoji\emotion\sentence_no_split，把这三个都读出来
'''
def get_data(isTrain,findtoken = None):

    # 读取test文本 sentences一维数组[（句子，情感，表情），（句子，情感，表情）...]
    sentences = []

    word_folder = "../data/nlpcc2014/all_data/"
    if isTrain:
        path = word_folder + "train_data.json"
        sentences_for_token = []  # 只记录句子，为了token
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
                sentences_for_token.append(dict['sentence_no_emoji'])

    print(sentences_for_token)
    if isTrain:
        findtoken = FindNewTokenOnJieba2(sentences=sentences_for_token)

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

    # 读取文本
    train_sentences,training_data, findtoken = get_data(isTrain=True)
    test_sentences,test_data,findtoken = get_data(False,findtoken)

    '''
    CRf部分
    '''
    START_TAG = "<START>"
    STOP_TAG = "<STOP>"
    EMBEDDING_DIM = 300 # input_size = embeddding_size
    HIDDEN_DIM = 128

    #字向量
    word_to_ix = {}
    for example in training_data:
        sentence = example['char_no_emoji']
        for word in sentence:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
    # print(word_to_ix) # word_to_ix记录非重复的词

    tag_to_ix = {"s": 0, "b": 1, "e": 2, "m" : 3,START_TAG: 4, STOP_TAG: 5}  # 标签


    model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

    # Check predictions before training
    with torch.no_grad():
        precheck_sent = prepare_sequence(training_data[0]['char_no_emoji'], word_to_ix)
        precheck_tags = torch.tensor([tag_to_ix[t] for t in training_data[0]['tags']], dtype=torch.long).to(device)
        score, tag_seq = model(precheck_sent)
        print("原始",get_result_word(training_data[0]['char_no_emoji'],precheck_tags))
        print("训练前",tag_seq,"-----",get_result_word(training_data[0]['char_no_emoji'],tag_seq))

    # Make sure prepare_sequence from earlier in the LSTM section is loaded
    for epoch in range(20):  # again, normally you would NOT do 300 epochs, it is toy data
        for example in training_data:
            tags = example['tags']
            sentence = example['char_no_emoji']
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            # Step 2. Get our inputs ready for the network, that is,
            # turn them into Tensors of word indices.
            sentence_in = prepare_sequence(sentence, word_to_ix) # 字向量
            targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long).to(device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

            # Step 3. Run our forward pass.
            loss,isException = model.neg_log_likelihood(sentence_in, targets)
            if isException:
                print(sentence,"-------",len(sentence))
                print(tags, "-------", len(tags))
            # Step 4. Compute the loss, gradients, and update the parameters by
            # calling optimizer.step()
            loss.backward()
            optimizer.step()
        print("总共traindata:" , len(training_data))

    # 存储一下训练集的结果，对比加入crf后的改变
    for example in training_data:
        with torch.no_grad():
            precheck_sent = prepare_sequence(example['char_no_emoji'], word_to_ix)
            # score, tag_seq = model(precheck_sent)
            example['crf_split'] = get_result_word(example['char_no_emoji'],tag_seq)
    write_to_file(True,"./data/",training_data)

    # 存储模型
    # torch.save(model.state_dict(), "crf")

    # 跑测试集
    for example in test_data:
        with torch.no_grad():
            precheck_sent = prepare_sequence(example['char_no_emoji'], word_to_ix)
            # score, tag_seq = model(precheck_sent)
            example['crf_split'] = get_result_word(example['char_no_emoji'],tag_seq)
    write_to_file(False,"./data/",test_data)

