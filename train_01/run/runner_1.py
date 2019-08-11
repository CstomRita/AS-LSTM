# -*- coding: utf-8 -*-
# @File  : runner_1.py
# @Author: ChangSiteng
# @Date  : 2019-06-23
# @Desc  :

from data_utils.sentence_split import SentenceSplit

if __name__ == '__main__':
    train_xml_path = "../data/nlpcc2014/Training data for Emotion Classification.xml"
    split = SentenceSplit(train_xml_path)
    split.sentence_split()
    datas = split.datas
    # print(datas['sentence'].__len__())
    # print(datas['sentence_no_emoji_split'].__len__())
