# -*- coding: utf-8 -*-
'''
 @File  : utils.py
 @Author: ChangSiteng
 @Date  : 2019-11-11
 @Desc  : 
 '''

# import-path
import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

def getType():
    type = sys.argv[1]
    lossType = sys.argv[2]
    if sys.argv[3] == "0" :
        isCall = True
        topic = sys.argv[4]
        # now_time = sys.argv[5]
    else :
        isCall = False
        topic = ""
        # now_time = ""


    if (type == "1"):
        if(lossType == "0"):model_path = 'model_hasSplit_lastOne.pt'
        if(lossType == "1"): model_path = 'model_hasSplit_bestAcc.pt'
        if (lossType == "2"): model_path = 'model_hasSplit_bestLoss.pt'
        dataFolder = 'data_hasEmoji'
    if (type == "2"):
        if (lossType == "0"):model_path = 'model_all_lastOne.pt'
        if (lossType == "1"): model_path = 'model_all_bestAcc.pt'
        if (lossType == "2"): model_path = 'model_all_bestLoss.pt'
        dataFolder = 'all_data'
    if (type == "3"):
        if (lossType == "0"):model_path = 'model_emojiHasSplit_lastOne.pt'
        if (lossType == "1"): model_path = 'model_emojiHasSplit_bestAcc.pt'
        if (lossType == "2"): model_path = 'model_emojiHasSplit_bestLoss.pt'
        dataFolder = 'data_splitHasEmoji'
    if (type == "4"):
        if (lossType == "0"): model_path = "model_hasEmoji_lastOne.pt"
        if (lossType == "1"): model_path = "model_hasEmoji_bestAcc.pt"
        if (lossType == "2"): model_path = "model_hasEmoji_bestLoss.pt"
        dataFolder = 'data_hasEmoji'
    if (type == "5"):
        if (lossType == "0"): model_path = "model_data_crf_lastOne.pt"
        if (lossType == "1"): model_path = "model_data_crf_bestAcc.pt"
        if (lossType == "2"): model_path = "model_data_crf_bestLoss.pt"
        dataFolder = 'data_crf'


    '''
    crf 新词识别的对比实验
     '''
    if (type == "6"):
        if (lossType == "0"): model_path = "model_data_crf_lastOne.pt"
        if (lossType == "1"): model_path = "model_data_crf_bestAcc.pt"
        if (lossType == "2"): model_path = "model_data_crf_bestLoss.pt"
        dataFolder = 'crf_datas/2origin/all_data'
    if (type == "7"):
        if (lossType == "0"): model_path = "model_data_crf_lastOne.pt"
        if (lossType == "1"): model_path = "model_data_crf_bestAcc.pt"
        if (lossType == "2"): model_path = "model_data_crf_bestLoss.pt"
        dataFolder = 'crf_datas/2origin/data_hasEmoji'
    if (type == "8"):
        if (lossType == "0"): model_path = "model_data_crf_lastOne.pt"
        if (lossType == "1"): model_path = "model_data_crf_bestAcc.pt"
        if (lossType == "2"): model_path = "model_data_crf_bestLoss.pt"
        dataFolder = 'crf_datas/3muwr/all_data'
    if (type == "9"):
        if (lossType == "0"): model_path = "model_data_crf_lastOne.pt"
        if (lossType == "1"): model_path = "model_data_crf_bestAcc.pt"
        if (lossType == "2"): model_path = "model_data_crf_bestLoss.pt"
        dataFolder = 'crf_datas/3muwr/data_hasEmoji'
    if (type == "10"):
        if (lossType == "0"): model_path = "model_data_crf_lastOne.pt"
        if (lossType == "1"): model_path = "model_data_crf_bestAcc.pt"
        if (lossType == "2"): model_path = "model_data_crf_bestLoss.pt"
        dataFolder = 'crf_datas/2origin_1/all_data'



    print(dataFolder,"-------",model_path,"------",lossType,"----",isCall,"-------")
    return dataFolder,model_path,lossType,isCall,topic