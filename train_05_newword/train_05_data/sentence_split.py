# -*- coding: utf-8 -*-
# @File  : sentence_split.py
# @Author: ChangSiteng
# @Date  : 2019-06-26
# @Desc  : 1. 分词的工作
#          2. 表情符号切分的工作
import sys
import os

from train_05_newword.new_word.find_new_word import FindNewToken
from train_05_newword.new_word.read_file import read_file

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import json
import pickle
import re

import jieba

from data_utils.nlpcc_parse import Parse

punctuations = ['，', '。', '？', '~', '！', '、', '……']

class SentenceSplit:

    datas = []
    pattern = ''
    emoji_pattern = ''

    def __init__(self,path):
        parse = Parse("../../data/nlpcc2014/emotion_label.json")
        self.datas = parse.parse(path)
        self.pattern = SentenceSplit.get_pattern()
        self.emoji_pattern = r'\[(\w*)\]'

        file = read_file(path)
        text = file.sentence_split()
        self.findtoken = FindNewToken(text)

    def set_path(self,path):
        parse = Parse("../../data/nlpcc2014/emotion_label.json")
        self.datas = parse.parse(path)
        self.pattern = SentenceSplit.get_pattern()
        self.emoji_pattern = r'\[(\w*)\]'

        file = read_file(path)
        text = file.sentence_split()
        self.findtoken.append_text_train_again(text)

    # 静态方法
    @staticmethod
    def get_pattern():
        pattern = ''
        for punctuation in punctuations:
            pattern += punctuation + "|"
        pattern = pattern[:-1]
        return pattern

    def sentence_split(self, path, iftrain):
        # 这里是获取的每一句话，需要做分词的工作，我们再在这里做将表情符号切分的工作
        for example in self.datas:
            # print(sentence)
            sentence = example['sentence']
            sentence_no_emoji_split = ''
            emoji_list = []
            emoji_count = []
            # 1 首先表情符号的纯文本
            sentence_no_emoji = re.sub(self.emoji_pattern, '', sentence)
            # 2 按照标点符号切分子句
            short_sentences = re.split(self.pattern,sentence)
            punctuations = re.findall(self.pattern,sentence) # 为了保持标点符号的一致
            # 3在每个子句中看是否有表情符号，这是因为子句后的表情符号会对子句产生影响
            for short_sentence in short_sentences:
                if short_sentence.strip() == '':
                    continue
                emojis = re.findall(self.emoji_pattern, short_sentence)
                emoji_list.append(list(emojis))

                # 4根据除去表情符号的子句，再分词
                short_entence_no_emoji = re.sub(r'\[(\w*)\]', '', short_sentence)
                if short_entence_no_emoji.strip() == '':
                    continue
                sentence_no_emoji_split_temp = " ".join(self.word_split(short_entence_no_emoji))
                sentence_no_emoji_split = sentence_no_emoji_split + str(sentence_no_emoji_split_temp)
                index = short_sentences.index(short_sentence)
                if len(punctuations) > index:
                    sentence_no_emoji_split = sentence_no_emoji_split + punctuations[index]

            example['sentence_no_emoji'] = sentence_no_emoji
            example['emoji'] = (emoji_list)
            example['sentence_no_emoji_split'] = sentence_no_emoji_split

        #https://blog.csdn.net/weixin_43896398/article/details/85559172
        # torchtext能够读取的json文件和我们一般意义上的json文件格式是不同的（这也是比较坑的地方），我们需要把上面的数据处理成如下格式：
        #
        # {"source": "10 111 2 3", "target": "1 1 2 2"}
        # {"source": "10 111 2 3", "target": "1 1 2 2"}
        # {"source": "10 111 2 3", "target": "1 1 2 2"}
        # {"source": "10 111 2 3", "target": "1 1 2 2"}
        # {"source": "10 111 2 3", "target": "1 1 2 2"}
        #可以看到，里面的内容和通常的Json并无区别，每个字段采用字典的格式存储。
        # 不同的是，多个json序列中间是以换行符隔开的，而且最外面没有列表。

        with open(path, 'w+') as fw:
            for example_data in self.datas:
                encode_json = json.dumps(example_data)
                # 一行一行写入，并且采用print到文件的方式
                print(encode_json, file=fw)

        # json_data = json.dumps(self.datas)
        # with open(path, 'w+',encoding='utf-8') as f_six: # w+用于读写，覆盖
        #     f_six.write(json_data)
        print("load data并保存在",path)

        if iftrain:
        # 将分好的词划分出来，拼接到一起，方便glove训练
            with open('words_origin.txt','w+') as fw:
                for example_data in self.datas:
                    print(example_data['sentence_no_emoji_split'],file=fw)
            print("分词TXT已经保存在words_origin.txt中")
        # 将表情符单词 供glove词向量
            with open('emojis_origin.txt','w+') as fw:
                for example_data in self.datas:
                    temp = ''
                    emojis_origin = example_data['emoji']
                    for emoji_origin in emojis_origin:
                        if len(emoji_origin) > 0:
                            for emoji_temp in emoji_origin:
                             temp += emoji_temp + ' '
                    if len(temp) > 0 :
                        print(temp,file=fw)
            print("表情分词TXT已经保存在emojis_origin.txt中")

    '''
    这里是调用分词的入口
    '''
    def word_split(self,sentence):
        (txt, sent_token) = self.findtoken.cut_sentence(sentence)
        return sent_token


if __name__=='__main__':
    train_xml_path = "../../data/nlpcc2014/Training data for Emotion Classification.xml"
    split = SentenceSplit(train_xml_path)
    split.sentence_split("../train_05_data/train_data.json",True)

    print(len(split.findtoken.texts))

    test_xml_path = "../../data/nlpcc2014/EmotionClassficationTest.xml"
    split.set_path(test_xml_path)
    split.sentence_split("../train_05_data/test_data.json",False)

    print(len(split.findtoken.texts))