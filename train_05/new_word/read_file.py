# -*- coding: utf-8 -*-
'''
 @File  : read_file.py
 @Author: ChangSiteng
 @Date  : 2019-08-01
 @Desc  : 
 '''

# import-path
import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

# -*- coding: utf-8 -*-
# @File  : sentence_split.py
# @Author: ChangSiteng
# @Date  : 2019-06-26
# @Desc  : 1. 分词的工作
#          2. 表情符号切分的工作
import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import json
import pickle
import re


from data_utils.nlpcc_parse import Parse

punctuations = ['，', '。', '？', '~', '！', '、', '……']

class read_file:

    datas = []
    pattern = ''
    emoji_pattern = ''

    def __init__(self,path):
        parse = Parse("../../data/nlpcc2014/emotion_label.json")
        self.datas = parse.parse(path)
        self.pattern = read_file.get_pattern()
        self.emoji_pattern = r'\[(\w*)\]'

    # 静态方法
    @staticmethod
    def get_pattern():
        pattern = ''
        for punctuation in punctuations:
            pattern += punctuation + "|"
        pattern = pattern[:-1]
        return pattern

    def sentence_split(self):
        sentences = []
        # 这里是获取的每一句话，需要做分词的工作，我们再在这里做将表情符号切分的工作
        for example in self.datas:
            sentence = example['sentence']
            example['sentence_no_emoji']=[]
            # 1 首先表情符号的纯文本
            sentence_no_emoji = re.sub(self.emoji_pattern, '', sentence)
            # 2 按照标点符号切分子句
            short_sentences = re.split(self.pattern,sentence)
            punctuations = re.findall(self.pattern,sentence) # 为了保持标点符号的一致
            for short_sentence in short_sentences:
                # 4根据除去表情符号的子句，再分词
                short_entence_no_emoji = re.sub(r'\[(\w*)\]', '', short_sentence)
                index = short_sentences.index(short_sentence)
                if short_entence_no_emoji.strip() :
                    example['sentence_no_emoji'].append(short_entence_no_emoji)

        for example_data in self.datas:
            sentences_no_emoji = (example_data['sentence_no_emoji'])
            if len(sentences_no_emoji) > 0:
                sentences.extend(sentences_no_emoji)

        return sentences



if __name__=='__main__':
    train_xml_path = "../../data/nlpcc2014/Training data for Emotion Classification.xml"
    file = read_file(train_xml_path)
    sentence = file.sentence_split()
    print(sentence)