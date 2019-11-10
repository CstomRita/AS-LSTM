# -*- coding: utf-8 -*-
# @File  : word_and_emoji_embedding.py
# @Author: ChangSiteng
# @Date  : 2019-07-05
# @Desc  :
import sys
import os

from torchtext.vocab import Vectors

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import re

import torch
from torch import nn
from torchtext import data
from torchtext import datasets
import random

# torchtext是Torch中封装的词向量训练工具
from torchtext.data import BucketIterator

from data_utils.sentence_split import SentenceSplit


def sentence_no_emoji_split_tokenizer(text):
    # 这里的主要目的是为了切词，不管分成了几句话，只要把分的词放在数组返回即可
    # 输入的一个字符串类型的
    words = []
    pattern = SentenceSplit.get_pattern()
    sentences = re.split(pattern,text)
    for sentence in sentences:
        temp = [word for word in sentence.split() if word.strip()]
        words.extend(temp) # extend，将一个微博中多个子句的分词结果合并成一个一维数组返回
        # append返回的是二维数组，表示的是各个分句下的分词结果
    return words


class Tensor:

    EMOTION = ''
    TEXT = ''
    train_data = ''
    test_data = ''
    valid_data = ''
    batch_size = ''
    train_json_name = 'train_data.json'
    test_json_name = 'test_data.json'

    def __init__(self,batch_size,SEED,dataFolder):

        path = os.getcwd()[0:os.getcwd().rfind("/")] + '/data/nlpcc2014/' + dataFolder + '/'
        self.path = path
        self.batch_size = batch_size

        # 1 创建Filed对象# Field对象指定你想要怎么处理某个数据
        #         # 数字时不需要使用词向量use_vocab
        #         # 是否需要tokenizer切分 sequential
        self.EMOTION = data.Field(sequential=False, use_vocab=False)
        self.TEXT = data.Field(sequential=True, tokenize=sentence_no_emoji_split_tokenizer)

        # 2 加载语料库
        self.load_data(SEED)

        # 3 构建语料库的vocabulary     # 同时，加载预训练的 word-embedding
        # 也可以通过 vocab.Vectors 使用自定义的 vectors.
        # 从预训练的 vectors 中，将当前 corpus 词汇表的词向量抽取出来，构成当前 corpus 的 Vocab（词汇表）
        # 指定缓存路径
        cache = path + '.vector_cache'
        word_vectors = Vectors(name='glove.words.300.vectors.txt', cache=cache)
        self.TEXT.build_vocab(self.train_data,vectors=word_vectors)
        # TEXT.build_vocab会指定构建哪个数据集的哪个word-embedding，并赋给TEXT这个对象
        # 对于测试集，不需要构建


    def train_iterator(self):
        # 4 batching 操作：用 torchtext 提供的 API 来创建一个 iterator
        train_iterator = BucketIterator(self.train_data,
                                batch_size=self.batch_size,
                                device = torch.device("cpu")) # cpu by -1, gpu by 0
                                # sort_key=lambda x: len(x.source), # field sorted by len
                                # sort_within_batch=True,
                                # repeat=False)
        return train_iterator

    def test_iterator(self):
        test_iterator = BucketIterator(self.test_data,
                                        batch_size=self.batch_size,
                                        device=torch.device("cpu"))
        return test_iterator

    def valid_iterator(self):
        valid_iterator = BucketIterator(self.valid_data,
                                       batch_size=self.batch_size,
                                       device=torch.device("cpu"))
        return valid_iterator

    def get_TEXT(self):
        return self.TEXT

    def get_EMOTION(self):
        return self.EMOTION


    def get_text_vocab(self):
        # Field的vocab属性保存了wordvector数据，我们可以把这些数据拿出来
        # 然后我们使用 Pytorch的EmbeddingLayer来解决embeddinglookup问题。
        # embed = nn.Embedding(len(vocab), self.emb_dim)
        # embed.weight.data.copy_(vocab.vectors)
        vocab = self.TEXT.vocab
        return vocab

    # 2 加载语料库
    def load_data(self,SEED):
        # torchtext能够读取的json文件，多个json序列中间是以换行符隔开的，而且最外面没有列表
        self.train_data = data.TabularDataset.splits(
            path=self.path,
            train=self.train_json_name,
            format='json',
            fields={
                'sentence_no_emoji_split': ('sentence_no_emoji_split', self.TEXT),
                'emotions': ('emotions', self.EMOTION)
            }
        )[0]  # 这里有个0是因为是split的，虽然split只有一个元素，那是一个数组
        self.test_data = data.TabularDataset.splits(
            path=self.path,
            train=self.train_json_name,
            format='json',
            fields={
                'sentence_no_emoji_split': ('sentence_no_emoji_split', self.TEXT),
                'emotions': ('emotions', self.EMOTION)
            }
        )[0]
        # 将训练集切分成训练集和验证集
        self.train_data, self.valid_data = self.train_data.split(random_state=random.seed(SEED))

        # for i in range(0, len(self.valid_data)):
        #     print(vars(self.valid_data[i]))


if __name__ == "__main__":
    BATCH_SIZE = 64
    SEED = 1234
    tensor = Tensor(BATCH_SIZE, SEED)
    train_iterator = tensor.train_iterator()