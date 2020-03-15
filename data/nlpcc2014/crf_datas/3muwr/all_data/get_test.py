# -*- coding: utf-8 -*-
'''
 @File  : get_test.py
 @Author: ChangSiteng
 @Date  : 2020-03-15
 @Desc  : 
 '''

# import-path
import json
import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

if __name__ == '__main__':
    path = "train_data.json"
    splits=[]
    with open(path, 'r') as load_f:
        for line in load_f:
            dict = json.loads(line)
            splits.append("/".join(str(dict['sentence_no_emoji_split']).split()))
    with open('train_data_split.txt', 'w+') as fw:
        for example_data in splits:
            print(example_data, file=fw)

    path = "test_data.json"
    splits=[]
    with open(path, 'r') as load_f:
        for line in load_f:
            dict = json.loads(line)
            splits.append("/".join(str(dict['sentence_no_emoji_split']).split()))
    with open('test_data_split.txt', 'w+') as fw:
        for example_data in splits:
            print(example_data, file=fw)