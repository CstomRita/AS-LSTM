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

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

def test():
    nums = [1,2,3]
    try:
        num = nums[3]
    except BaseException:
        print("exception1")
        raise ValueError('A very specific bad thing happened.')

def test2():
    try:
        test()
    except BaseException:
        print("exception2")