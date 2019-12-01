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

class read_file_onJieba:


    def __init__(self,dataPath):
        self.path =  dataPath


    def get_splitSentenceByJieba(self):
        Efield = []
        with open(self.path,"r") as file_read:
            while True:
                lines = file_read.readline()  # 整行读取数据
                if not lines:
                    break
                    pass
                tmp = lines.split()  # 将整行数据分割处理，如果分割符是空格，括号里就不用传入参数
                # 添加新读取的数据
                Efield.append(tmp)
                pass
        return Efield
        pass

if __name__ == '__main__':
      dataPath = "../../data/nlpcc2014/data_splitHasEmoji/words_origin.txt"
      file = read_file_onJieba(dataPath)
      file.get_splitSentenceByJieba()