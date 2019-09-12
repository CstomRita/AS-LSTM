# -*- coding: utf-8 -*-
# @File  : lstm_attention.py
# @Author: ChangSiteng
# @Date  : 2019-07-06
# @Desc  :
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


'''
这里的LSTM和train_02中的LSTM在batch_size上的处理是不同的
train_02中的batch是通过TorchText中的iterator实现的，传入LSTM的sentence就是带有batch——size的
但是这里由于要各个分局合并成一个整句，无法使用iterator，使用的是LSTM里的batch——size，需要显式声明，也就是在hidden中创建的
'''

class ATTENTION_LSTM(nn.Module):
    def __init__(self, EMOJI_VOCAB,TEXT_VOCAB, EMBEDDING_DIM, INPUT_SIZE, HIDDEN_SIZE, NUM_LAYER, BIDIRECTIONAL, DROPOUT, LABEL_SIZE, BATCH_SIZE):
        super(ATTENTION_LSTM, self).__init__()

        self.EMBEDDING_DIM = EMBEDDING_DIM
        self.NUM_LAYER = NUM_LAYER
        self.HIDDEN_SIZE = HIDDEN_SIZE
        self.BATCH_SIZE = BATCH_SIZE
        self.USE_GPU = torch.cuda.is_available()
        self.TEXT_VOCAB = TEXT_VOCAB
        self.BIDIRECTIONAL = BIDIRECTIONAL
        self.LABEL_SIZE = LABEL_SIZE
        self.EMOJI_VOCAB = EMOJI_VOCAB

        self.init_embedding(TEXT_VOCAB)
        self.init_hidden()
        self.init_hidden2label()

        self.lstm = nn.LSTM(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE,
                               num_layers=NUM_LAYER, bidirectional=BIDIRECTIONAL,
                               dropout=DROPOUT)
        # 此时三维向量为[seq_len,batch_size,embedding_size]

        # 这个是为了学习所有分句整合结果的
        # 分句[batch_size,hidden_size * numlayer]
        # m 个分句[m,hidden_size * numlaye],要求输出[batch_size,hidden_size * numlayer]
        # lstm输入[seq_len,batch_size,input_size] 输出[batch_size,hidden_size * numlayer]
        self.sentence_lstm = nn.LSTM(input_size=HIDDEN_SIZE , hidden_size=HIDDEN_SIZE,
                               num_layers=NUM_LAYER, bidirectional=BIDIRECTIONAL,
                               dropout=DROPOUT)

        self.attention = nn.Linear(EMBEDDING_DIM,1)

    def init_hidden2label(self):
        sentence_num = 1
        if self.BIDIRECTIONAL: # true为双向LSTM false单向LSTM
            self.hidden2label = nn.Linear(self.HIDDEN_SIZE * 2 * sentence_num, self.LABEL_SIZE)
        else:
            self.hidden2label = nn.Linear(self.HIDDEN_SIZE  * sentence_num, self.LABEL_SIZE)

    def init_embedding(self,VOCAB):
        weight_matrix = VOCAB.vectors
        # 使用已经处理好的词向量
        self.word_embeddings = nn.Embedding(len(VOCAB), self.EMBEDDING_DIM)
        self.word_embeddings.weight.data.copy_(weight_matrix)


    def init_hidden(self, batch_size=None):
        if batch_size is None:
            batch_size = self.BATCH_SIZE

        if self.USE_GPU:
            h0 = Variable(torch.zeros(self.NUM_LAYER, batch_size, self.HIDDEN_SIZE).cuda())
            c0 = Variable(torch.zeros(self.NUM_LAYER, batch_size, self.HIDDEN_SIZE).cuda())
        else:
            h0 = Variable(torch.zeros(self.NUM_LAYER, batch_size, self.HIDDEN_SIZE))
            c0 = Variable(torch.zeros(self.NUM_LAYER, batch_size, self.HIDDEN_SIZE))
        return (h0, c0)

    def attention_net(self, lstm_out, h_n):
        # 使用什么作为注意力机制，以下只是一个例子，具有相当的自主性
        # lstm_out [seq_len,batch_size,hidden_size]
        # h_n [NUM_LAYER, batch_size, HIDDEN_SIZE]



        return torch.cat([lstm_out[0], lstm_out[-1]], dim=1)
        # 要求return [batch_size,hiddden_size*NUM_LAYER]


    def forward(self, sentences, device):
        # 这里的batch_size都是1，未做批量处理
        sentence_num = len(sentences)
        # print(sentence_num)
        all_out = []
        for sentence_index, sentence in enumerate(sentences): # 借助enumerate函数循环遍历时获取下标
            indexed = [self.TEXT_VOCAB.stoi[t] for t in sentence]  # 在词典中获取index
            senetence_tensor = torch.LongTensor(indexed).to(torch.device("cuda:0" if torch.cuda.is_available() else"cpu"))
            senetence_tensor = senetence_tensor.unsqueeze(1).to(device)  # 获取向量化后的一句话矩阵
            # sentence_tensor[sentence_length,batch_size]

            embeddings = self.word_embeddings(senetence_tensor)
            # embedding[seq_len,batch_size,embedding_size]

            # senetence_tensor.size()[0]--> sentence_lenght
            # init_hidden只是初始化一些零矩阵
            # [NUM_LAYER, batch_size, HIDDEN_SIZE]
            # 初始化hidden的原因是什么？？？？？
            h_0,c_0 = self.init_hidden(batch_size=1)

            lstm_out,(h_n,c_n) = self.lstm(embeddings,(h_0,c_0))
            attention_out = lstm_out[-1]
            # lstm_out [seq_len,batch_size,hidden_size]
            # h_n [NUM_LAYER, batch_size, HIDDEN_SIZE]

            # 融合注意力机制
            # attention_out = self.attention_net(lstm_out,h_n)
            # print(type(attention_out)) # [batch_size,hidden_size * layer_num]

            # 所有分句的Attention输出 整合在一起 all_out[sentence_num,batch_size,hidden_size * num_layer]
            if len(all_out) == 0:
                all_out = torch.unsqueeze(attention_out,0)
            else:
                attention_out = torch.unsqueeze(attention_out, 0)
                all_out = torch.cat((all_out,attention_out),0)

        # 方案A:将所有分句的输出经过额外一层LSTM学习
        print(all_out.size())
        all_out_lstm_out,all_out_lstm_hidden = self.sentence_lstm(all_out)
        # print(all_out_lstm_out.size()) # all_out_lstm_out[sentence_num,batch_size,hidden_size * num_layer]
        # 选择最后一个单元的输出作为所有分句的整体表示
        all_out_lstm_encoding = all_out_lstm_out[-1] # 选取了最后一个状态[batch_size,hidden_size * num_layer]
        # print(all_out_lstm_encoding.size())

        output = self.hidden2label(all_out_lstm_encoding)
        # output [batch_size,label_size]
        # 在这里的batch_size都是1

        return output
