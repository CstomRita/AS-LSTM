# -*- coding: utf-8 -*-
'''
 @File  : run.py
 @Author: ChangSiteng
 @Date  : 2020-01-05
 @Desc  : 
 '''

# import-path
import sys
import os
import time

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import os
import jieba

from train_05_newword.new_word_3.model import TrieNode
from train_05_newword.new_word_3.utils import get_stopwords, load_dictionary, generate_ngram, save_model, load_model


def load_data(filename, stopwords):
    """
    :param filename:
    :param stopwords:
    :return: 二维数组,[[句子1分词list], [句子2分词list],...,[句子n分词list]]
    """
    data = []
    with open(filename, 'r') as f:
        for line in f:
            word_list = [x for x in jieba.cut(line.strip(), cut_all=False) if x not in stopwords]
            data.append(word_list)
    return data


def load_data_2_root(root,data):
    print('------> 插入节点','共',len(data),'次')
    start = time.clock()
    i = 0
    for word_list in data:
        # tmp 表示每一行自由组合后的结果（n gram）
        # tmp: [['它'], ['是'], ['小'], ['狗'], ['它', '是'], ['是', '小'], ['小', '狗'], ['它', '是', '小'], ['是', '小', '狗']]
        ngrams = generate_ngram(word_list, 3)
        # 把ngram都插入其中，没有用词频限制
        i += 1
        print('------> 插入节点',i,'------有ngram',len(ngrams),'次')
        for d in ngrams:
            root.add(d)
        if i >= 5000:
            break # 跳出循环，先拿5000数据
    end = time.clock()
    print('------> 插入成功,花费',(end-start)/60,'分种')


def run(data,topN):
    root_name =  rootPath+'/new_word_3/data/root.pkl' # 词频树的存放，可以便于计算互信息
    stopwords = get_stopwords()
    if os.path.exists(root_name):
        root = load_model(root_name)
    else:
        dict_name =  rootPath+'/new_word_3/data/dict.txt'
        word_freq = load_dictionary(dict_name)
        root = TrieNode('*', word_freq)
        save_model(root, root_name)


    # 将新的文章插入到Root中
    load_data_2_root(root,data)

    # 定义取TOP5个

    result, add_word = root.find_word(topN)

    return result,add_word


if __name__ == '__main__':
 # 加载数据集
    stopwords = get_stopwords()
    filename = './data/demo.txt'
    data = load_data(filename,stopwords) # 数据集分词后的结果 二维数组,[[句子1分词list], [句子2分词list],...,[句子n分词list]]
    result, add_word = run(data,8)

    print(data)
 # 如果想要调试和选择其他的阈值，可以print result来调整
    # print("\n----\n", result)
    print("\n----\n", '增加了 %d 个新词, 词语和得分分别为: \n' % len(add_word))
    print('#############################')
    for word, score in add_word.items():
        print(word + ' ---->  ', score)
    print('#############################')

    # 前后效果对比
    test_sentence = '蔡英文在昨天应民进党当局的邀请，准备和陈时中一道前往世界卫生大会，和谈有关九二共识问题'
    print('添加前：')
    print("".join([(x + '/ ') for x in jieba.cut(test_sentence, cut_all=False) if x not in stopwords]))

    for word in add_word.keys():
        jieba.add_word(word)
    print("添加后：")
    print("".join([(x + '/ ') for x in jieba.cut(test_sentence, cut_all=False) if x not in stopwords]))