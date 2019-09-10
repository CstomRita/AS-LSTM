# -*- coding: utf-8 -*-
# @File  : nlpcc_parse.py
# @Author: ChangSiteng
# @Date  : 2019-06-21
# @Desc  : nlpcc2014的训练集切分
import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import xml.dom.minidom
import re
import os
import pickle
import numpy as np
import torch
from pandas._libs import json

class Parse:

    emotion_labels = {}
    train_xml_path = "../data/nlpcc2014/Training data for Emotion Classification.xml"

    def __init__(self,path="../../data/nlpcc2014/emotion_label.json"):
        with open(path,"r") as load_f:
            self.emotion_labels = json.load(load_f)

    def parse(self, path,isTest = False):
        datas = []
        number = 0
        sentenceNum = 0
        emotionNum = {
            0 : 0,
            1 : 0,
            2 : 0,
            3 : 0,
            4 : 0,
            5 : 0,
            6 : 0,
            7: 0
        }
        # open file
        dom = xml.dom.minidom.parse(path)
        # 加载最顶级元素节点，这里获取的trainData元素
        # getElementsByTagName获取子元素列表
        # getAttribute获取标签内节点属性
        # xx.childNodes获取所有子节点列表，Text类型
        # xx.childNodes[0].data获取某一个子节点的data
        root = dom.documentElement
        weibos = root.getElementsByTagName('weibo')
        for weibo in weibos:
            number += 1
            sentens = weibo.getElementsByTagName('sentence')
            for seten in sentens:
                sentenceNum += 1
                example = {}
                data_origin = seten.childNodes[0].data
                # 在这里做一些数据清洗，去除//@xx：这种转发的信息
                # 回复@xx:
                data1 = re.sub(r'//@(.*):',"",data_origin)
                data = re.sub(r'回复@(.*):', "", data1)
                example['sentence'] = data
                opinionated = seten.getAttribute('opinionated')
                if opinionated == 'N': # 没有情感
                    emotions = (self.emotion_labels['none'])
                else :
                    emotion_1 = seten.getAttribute('emotion-1-type')
                    emotion_2 = seten.getAttribute('emotion-2-type')

                    # 只考虑第一情感
                    if emotion_2 == 'none':
                        emotions = (self.emotion_labels[emotion_1])
                    else:
                        emotions = (self.emotion_labels[emotion_2])
                    # 考虑两种情感
                    # if emotion_2 == 'none':
                    #     emotions = str(self.emotion_labels[emotion_1])
                    # elif emotion_1 == 'none':
                    #     emotions = str(self.emotion_labels[emotion_2])
                    # else :
                    #     emotions = str(self.emotion_labels[emotion_1]) + "," + str(self.emotion_labels[emotion_2])

                example['emotions'] = emotions
                emotionNum[emotions] +=1;
                datas.append(example)
        print(number , "条数据解析")
        print(sentenceNum,"句训练语料")
        print("语料统计",emotionNum)
        return datas


if __name__ == '__main__':
    split = Parse()
    datas = split.parse("../data/nlpcc2014/Training data for Emotion Classification.xml")
    print(datas)
