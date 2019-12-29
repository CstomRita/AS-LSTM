# -*- coding: utf-8 -*-
'''
 @File  : find_new_word_onJieba.py
 @Author: ChangSiteng
 @Date  : 2019-11-20
 @Desc  :  在结巴分词上再进行分词的扩展
 texts是结巴分词后的结果 二维数组，[[每句话的结巴分词],[],[]]，存放的所有的训练句子结巴分词后的结果
 append_text_train_again是重新训练，还是会在结巴分词的结果上进行训练，要求格式一个一维数组[句子，句子，句子]的形式
 cut_sentence(一个句子)
 '''

# import-path
import json
import sys
import os

import jieba

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from collections import defaultdict
import numpy as np

split_symbol = " "

class FindNewTokenOnJieba:
    def __init__(self, sentences , min_count=15, token_length=4, min_proba={2: 5, 3: 25, 4: 100}):
        '''
        min_count：出现的频次为多少认为是一个词
        token_length:分词的最大长度
        要和 min_proba 保持一致{2:5,3:25,4:125,.......token_length:xxxx}
        '''
        self.min_count = min_count
        self.token_length = token_length
        self.min_proba = min_proba
        self.texts = []
        for sentence in sentences:
            generator = jieba.cut(sentence)
            words = split_symbol.join(generator).split(split_symbol)
            self.texts.append(words)


        '''
        1：统计1，2，3...ngrams的词频并根据词频设定的阈值过滤小于阈值的部分
        '''
        self.statistic_ngrams()
        '''
        2 根据凝固度过滤掉第一步统计出来的ngrams中小于凝固度的词语
        '''
        self.filter_ngrams()
        '''
        3：根据以上步骤筛选出来的词语对句子进行分词
        self.all_tokens：切分出的所有词{词：个数}
        self.new_word_1:
        self.sentences_tokens[sentence:切分的词组]
        '''
        self.sentences_cut()
        '''
        4：对结果再筛选
        '''
        self.judge_exist()

    '''
    对一些文本再训练
    '''

    def append_text_train_again(self, texts):
        if (type(texts).__name__ == 'str'):
            '''
            要求texts的格式一个一维数组[句子，句子，句子]的形式
            '''
            for sentence in texts:
                generator = jieba.cut(sentence)
                words = split_symbol.join(generator).split(split_symbol)
                self.texts.append(words)
        self.statistic_ngrams()
        self.filter_ngrams()
        self.sentences_cut()
        self.judge_exist()

    '''
    算法步骤1：统计1，2，3...ngrams的词频并根据词频设定的阈值过滤小于阈值的部分
    '''

    def statistic_ngrams(self):  # 粗略统计1，2..ngrams
        print('Starting statistic ngrams!')
        ngrams = defaultdict(int)
        for txt in self.texts:
            for char_id in range(len(txt)):
                for step in range(1, self.token_length + 1):
                    if char_id + step <= len(txt):
                        word = tuple(txt[char_id:char_id + step]) # 转变成元组，元组是不可变的，可以做作为key，list不行
                        ngrams[word] += 1
        self.ngrams = {k: v for k, v in ngrams.items() if v >= self.min_count}
        # print("ngrams:",self.ngrams)

    '''
    算法步骤2：
    根据凝固度过滤掉第一步统计出来的ngrams中小于凝固度的词语

    根据凝固度公式计算2grams以上词语的凝固度，并过滤小于凝固度阈值的词语，
    针对不同长度的词语，凝固度的阈值设置不一样。
    按文中作者给出的经验阈值，以字典的形式给出，长度为2的为5，长度为3的为25，长度为4的为125。
    min_proba={2:5,3:25,4:125}。
    其中，凝固度的计算按照每种切分可能都进行计算，取最小的最为其凝固度。
    '''

    def filter_ngrams(self):  # 过滤凝固度小于设定阈值的词
        self.ngrams_ = set(token for token in self.ngrams if self.calculate_prob(token))
        # print("根据凝固度过滤的:",self.ngrams_)

    def calculate_prob(self, token):  # 计算2grams及以上的凝固度
        self.total = sum([v for k, v in self.ngrams.items() if len(k) == 1])
        if len(token) >= 2:
            # print(token)
            score = min(
                [self.total * self.ngrams[token] / (self.ngrams[token[:i + 1]] * self.ngrams[token[i + 1:]])
                 for i in range(len(token) - 1)]
            )
            if score > self.min_proba[len(token)]:
                return True
        else:
            return False

    '''
    算法步骤3：根据以上步骤筛选出来的词语对句子进行分词
    '''

    def sentences_cut(self):
        self.sentences_tokens = []
        all_tokens = defaultdict(int)
        for txt in self.texts:
            if len(txt) > 2:
                for token in self.cut_sentence_onWords(txt)[1]:
                    all_tokens[token] += 1
                self.sentences_tokens.append(self.cut_sentence_onWords(txt))
        self.all_tokens = {k: v for k, v in all_tokens.items() if v >= self.min_count}

    '''
    算法步骤4：
    上面的分词结果可能会存在一个问题
    比如说"各项目"，是由于"各项","项目"都存在于ngrams，所以"各项目"才会保留下来,
    这时可以扫描"各项目"是否在3grams里面，如果在其中，将其保留，不在就删除。
    其实，不进行这一步的筛选，我认为也是可以的,主要看你的需求,
    例如"大字号打印机"，可以切成{大字号，打印机}，也可以保留成一个。
    '''

    def judge_exist(self):
        self.pairs = []  ##按照 句子-token  进行显示
        for sent, token in self.sentences_tokens:
            real_token = []
            for tok in token:
                if self.is_real(tok) and len(tok) != 1:
                    real_token.append(tok)
            self.pairs.append((sent, real_token))
        self.new_word = {k: v for k, v in self.all_tokens.items() if self.is_real(k)}

    def is_real(self, token):
        if len(token) >= 3:
            for i in range(3, self.token_length + 1):
                for j in range(len(token) - i + 1):
                    if token[j:j + i] not in self.ngrams_:
                        return False
            return True
        else:
            return True

    '''
     工具类:统计发现的新词的个数
    '''

    def statistic_token(self):
        count = defaultdict(int)
        length = list(map(lambda x: len(x), self.new_word.keys()))
        for i in length:
            count[i] += 1
        print("每个词的字符串长度的个数统计：", count)

    def cut_sentence_onWords(self, words):
        mask = np.zeros(len(words) - 1)  # 从第二个字开始标注
        for char_id in range(len(words) - 1):
            for step in range(2, self.token_length + 1):
                if tuple(words[char_id:char_id + step]) in self.ngrams_:
                    mask[char_id:char_id + step - 1] += 1

        sent_token = [words[0]]
        for index in range(1, len(words)):
            if mask[index - 1] > 0:
                sent_token[-1] += words[index]
            else:
                sent_token.append(words[index])

        return (words, sent_token)
    '''
     工具类:切分某个一个句子
    '''

    def cut_sentence(self, sentence):
        # 同样是在jieba分词的结果上标注
        generator = jieba.cut(sentence)
        words = split_symbol.join(generator).split(split_symbol)
        mask = np.zeros(len(words) - 1)  # 从第二个词开始标注
        for char_id in range(len(words) - 1):
            for step in range(2, self.token_length + 1):
                if tuple(words[char_id:char_id + step]) in self.ngrams_:
                    mask[char_id:char_id + step - 1] += 1

        sent_token = []
        # sent_token = [words[0]]
        word = ""
        for index in range(len(words)):
            word += words[index]
            if index == len(words)-1 or mask[index] <= 0:
                sent_token.append(word)
                word = ""
        return sent_token


if __name__ == '__main__':
    # 读取train文本 sentences一维数组[句子，句子，句子]
    sentences = []
    train_word_folder = "../../data/nlpcc2014/all_data/"
    path = train_word_folder + "train_data.json"
    with open(path, 'r') as load_f:
        for line in load_f:
            dict = json.loads(line)
            sentences.append(dict['sentence'])

    findtoken = FindNewTokenOnJieba(sentences=sentences)
    print(findtoken.new_word)
    test = "这家套餐很不错"
    (txt, sent_token) = findtoken.cut_sentence(test)
    # print(",".join(jieba.cut(test)))
    # print(sent_token)
    # print(findtoken.new_word)