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
    if (type == "1"):
        if(lossType == "0"):model_path = 'model_hasSplit_lastOne.pt'
        if(lossType == "1"): model_path = 'model_hasSplit_bestAcc.pt'
        if (lossType == "2"): model_path = 'model_hasSplit_bestLoss.pt'
        dataFolder = 'data_hasSplit'
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
        if (lossType == "0"): model_path = "model_emoji_split_hasEmoji_lastOne.pt"
        if (lossType == "1"): model_path = "model_emoji_split_hasEmoji_bestAcc.pt"
        if (lossType == "2"): model_path = "model_emoji_split_hasEmoji_bestLoss.pt"
        dataFolder = 'data_emoji_split_hasEmoji'
    print(dataFolder,"-------",model_path,"------",lossType)
    return dataFolder,model_path,lossType