# -*- coding: utf-8 -*-
# @File  : train.py
# @Author: ChangSiteng
# @Date  : 2019-07-05
# @Desc  :
import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import time

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from train_01.lstm import LSTM

from train_01.cnn import CNN
from train_01.embedding import Tensor


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def categorical_accuracy(preds, y):
    """
    返回每个批次下的准确率
     i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    max_preds = preds.argmax(dim = 1, keepdim = True)
    # 获取最大概率的分类类别,只是这是一个批次的 [batch_size,1] 二维数组
    # 再通过squeeze函数去掉一维，变成[batch_size]的一维数组
    max_preds = max_preds.squeeze(1)

    correct = max_preds.eq(y) # 比较max_preds、y两个数组是否相等
    return correct.sum() / torch.FloatTensor([y.shape[0]])

def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch.sentence_no_emoji_split)
            loss = criterion(predictions, batch.emotions)
            acc = categorical_accuracy(predictions, batch.emotions)
            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


# 真正用模型训练的地方
def train(model, iterator, optimizer, criterion):
    print("开始训练")
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    for batch in iterator: # 这里的batch就是通过Torchtext切分的数据集,batch_size为一组一起训练的
        optimizer.zero_grad()
        predictions = model(batch.sentence_no_emoji_split) # model获取预测结果，此处会执行模型的forWord方法

        # batch.emotions没有经过向量训练，依然是[batch_size,1]的数字
        loss = criterion(predictions, batch.emotions) # loss
        acc = categorical_accuracy(predictions, batch.emotions) # 准确率

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def run_train_iterator(model,optimizer,criterion,train_iterator,N_EPOCHS):
    device = torch.device("cpu") # cpu by -1, gpu by 0
    model = model.to(device)
    criterion = criterion.to(device)
    for epoch in range(N_EPOCHS):
        start_time = time.time()
        train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')


def run_with_valid_iterator(model,model_path,optimizer,criterion,train_iterator,valid_iterator,N_EPOCHS):
    best_valid_loss = float('inf')
    device = torch.device("cpu")  # cpu by -1, gpu by 0
    model = model.to(device)
    criterion = criterion.to(device)
    for epoch in range(N_EPOCHS):
        start_time = time.time()
        train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        if epoch == N_EPOCHS-1:
        # if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            print(f'\t----存储模型-------')
            torch.save(model.state_dict(), model_path)
        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')

def run_test(model,model_path,criterion,test_iterator):
    print('开始测试-----加载模型')
    model.load_state_dict(torch.load(model_path))
    print(f'\t-----testing-------')
    test_loss, test_acc = evaluate(model, test_iterator, criterion)
    print(f'\tTest Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}%')

# 输入一句话 输出类别
# def predict_class(model, sentence, min_len = 4):
#     model.eval()
#     tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
#     if len(tokenized) < min_len:
#         tokenized += ['<pad>'] * (min_len - len(tokenized))
#     indexed = [TEXT.vocab.stoi[t] for t in tokenized]
#     tensor = torch.LongTensor(indexed).to(device)
#     tensor = tensor.unsqueeze(1)
#     preds = model(tensor)
#     max_preds = preds.argmax(dim = 1)
#     return max_preds.item()

if __name__ == '__main__':

    BATCH_SIZE = 64
    SEED = 1234
    tensor = Tensor(BATCH_SIZE,SEED)
    train_iterator = tensor.train_iterator()
    test_iterator = tensor.test_iterator()
    valid_iterator = tensor.valid_iterator()

    VOCAB = tensor.get_text_vocab()
    model_path = 'model.pt'


    #CNN
    # EMBEDDING_DIM = 100
    # N_FILTERS = 100
    # FILTER_SIZES = [2, 3, 4]
    # OUTPUT_DIM = 8
    # DROPOUT = 0.5
    # model = CNN(VOCAB, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT)

    EMBEDDING_DIM = 300
    INPUT_SIZE = 300 # EMBEDDING_DIM=INPUT_SIZE
    HIDDEN_SIZE = 128
    NUM_LAYER = 2
    LABEL_SIZE = 8
    model = LSTM(VOCAB, EMBEDDING_DIM, INPUT_SIZE, HIDDEN_SIZE, NUM_LAYER, False, 0, LABEL_SIZE)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    run_with_valid_iterator(
        model=model,
        model_path=model_path,
        optimizer=optimizer,
        criterion=criterion,
        train_iterator=train_iterator,
        valid_iterator=valid_iterator,
        N_EPOCHS = 20)

    run_test(model = model,model_path = model_path,criterion = criterion,test_iterator = test_iterator)