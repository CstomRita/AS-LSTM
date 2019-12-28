# -*- coding: utf-8 -*-
# @File  : find_new_word.py
# @Author: ChangSiteng
# @Date  : 2019-06-12
# @Desc  : 来源 https://zhuanlan.zhihu.com/p/39461254
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import re



class FindNewToken(object):
    def __init__(self, texts,min_count=20, token_length=4,min_proba={2:5,3:25,4:125}):

        self.word_frequency = defaultdict(int) # 统计N-Gram中所有的词频
        '''
        min_count：出现的频次为多少认为是一个词
        token_length:分词的最大长度
        要和 min_proba 保持一致{2:5,3:25,4:125,.......token_length:xxxx}
        '''
        self.min_count = min_count
        self.token_length = token_length
        self.min_proba = min_proba
        self.texts = texts
        '''
        1：统计1，2，3...ngrams的词频 并根据词频设定的阈值过滤小于阈值的部分
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
    def append_text_train_again(self,texts):
        if (type(texts).__name__ == 'str'):
            '''
            对str做一个转成list的转变，否则str强制转换之后每个元素是字符
            "123"---->["1","2","3"]
            '''
            texts = [texts]
        self.texts.extend(texts)
        self.statistic_ngrams()
        self.filter_ngrams()
        self.sentences_cut()
        self.judge_exist()

    '''
    算法步骤1：统计1，2，3...ngrams的词频并根据词频设定的阈值过滤小于阈值的部分
    '''
    def statistic_ngrams(self): # 粗略统计一段文本里的1，2..ngrams
        print('Starting statistic ngrams!')
        for txt in self.texts:
            for char_id in range(len(txt)):
                for step in range(1,self.token_length+1):
                    if char_id+step <=len(txt):
                        self.word_frequency[txt[char_id:char_id+step]] += 1
        self.ngrams = {k:v for k,v in self.word_frequency.items() if v>=self.min_count}
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
            score = min(
                [self.total * self.ngrams[token] / (self.ngrams[token[:i + 1]] * self.ngrams[token[i + 1:]]) for i in
                 range(len(token) - 1)])
            if score > self.min_proba[len(token)]:
                return True
        else:
            return False

    '''
    算法步骤3：根据以上步骤筛选出来的词语对句子进行分词
    '''
    def cut_sentence(self, txt):
        mask = np.zeros(len(txt) - 1)  # 从第二个字开始标注
        for char_id in range(len(txt) - 1):
            for step in range(2, self.token_length + 1):
                if txt[char_id:char_id + step] in self.ngrams_:
                    mask[char_id:char_id + step - 1] += 1
        sent_token = [txt[0]]
        for index in range(1, len(txt)):
            if mask[index - 1] > 0:
                sent_token[-1] += txt[index]
            else:
                sent_token.append(txt[index])

        return (txt, sent_token)

    def sentences_cut(self): # 切分此段文本的所有句子
        self.sentences_tokens = []
        all_tokens = defaultdict(int)
        for txt in self.texts:
            if len(txt) > 2:
                for token in self.cut_sentence(txt)[1]:
                    all_tokens[token] += 1
                self.sentences_tokens.append(self.cut_sentence(txt))
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




# if __name__ == '__main__':
#     # train_xml_path = "../../data/nlpcc2014/Training data for Emotion Classification.xml"
#     # file = read_file(train_xml_path)
#     # text = file.sentence_split()
#     # findtoken = FindNewToken(text)
#     # print(findtoken.ngrams_)
    # (txt, sent_token) = findtoken.cut_sentence("真是喜大普奔啊啊")
    # print(findtoken.all_tokens)
    # print(findtoken.new_word)
    # findtoken.append_text_train_again(text)
