# -*- coding: utf-8 -*-
# @File  : get_data_json.py
# @Author: ChangSiteng
# @Date  : 2019-07-06
# @Desc  :
import json

from data_utils.sentence_split import SentenceSplit

if __name__=='__main__':
    train_xml_path = "../data/nlpcc2014/Training data for Emotion Classification.xml"
    split = SentenceSplit(train_xml_path)
    split.sentence_split("../data/nlpcc2014/train_data.json")


    # test_xml_path = "../data/nlpcc2014/Tesing data for Emotion Classification.xml"
    # split = SentenceSplit(test_xml_path)
    # split.sentence_split()
    # self.nlpcc_test_datas = split.datas