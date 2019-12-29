# -*- coding: utf-8 -*-
'''
 @File  : test.py
 @Author: ChangSiteng
 @Date  : 2019-12-28
 @Desc  : 
 '''

# import-path
import sys
import os

import jieba

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

def test():
    nums = [1,2,3]
    for i in range(len(nums)):
        try:
            num = nums[i+2]
            print(num)
        except BaseException:
            print("exception1")
            raise ValueError('A very specific bad thing happened.')

def test2():
    sentences = ["最上面两个是一种，，，中间三种"]
    tags = [ 'b', 'e', 'b', 'e', 'b', 'e', 's', 'b', 'e', 'b', 'e', 's', 'b', 'e', 'b', 'e', 's', 'b', 'e', 'b', 'm', 'm', 'm', 'e']
    training_data = []
    for sentence in sentences:
        characters = []
        for character in sentence:
            characters.append(character)
        words = ",".join(jieba.cut(sentence)).split(",")
        tags = []
        for word in words:
            # 判断 single —— s  Begin -b End-e  Medim-m
            length = len(word)
            if length == 1:
                tags.append("s")
            elif length == 2:
                tags.append("b")
                tags.append("e")
            else:
                tags.append("b")
                for i in range(1, length):
                    tags.append("m")
                tags.append("e")
        training_data.append((characters, tags))
    print(training_data)

    print(len(sentence))
    print(len(tags))
    try:
        test()
    except BaseException:
        print("exception2")