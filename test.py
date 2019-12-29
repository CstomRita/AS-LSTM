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
    sentence = [ ',', ',', ',', '中', '间', '三', '张', '是', '一', '款', ',', ',', ',', ',', '最', '下', '面', '四', '张', '是', '一', '款', '.', '.', '.', '.']
    tags = [ 'b', 'e', 'b', 'e', 'b', 'e', 's', 'b', 'e', 'b', 'e', 's', 'b', 'e', 'b', 'e', 's', 'b', 'e', 'b', 'm', 'm', 'e']

    print(len(sentence))
    print(len(tags))
    try:
        test()
    except BaseException:
        print("exception2")