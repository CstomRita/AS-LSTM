# -*- coding: utf-8 -*-
# @File  : lstm_attention.py
# @Author: ChangSiteng
# @Date  : 2019-07-06
# @Desc  :
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim




class LSTM(nn.Module):
    def __init__(self, TEXT_VOCAB, EMBEDDING_DIM, INPUT_SIZE, HIDDEN_SIZE, NUM_LAYER, BIDIRECTIONAL, DROPOUT, LABEL_SIZE):
        super(LSTM, self).__init__()

        self.EMBEDDING_DIM = EMBEDDING_DIM
        self.TEXT_VOCAB = TEXT_VOCAB
        self.init_word_embedding(TEXT_VOCAB)

        self.lstm = nn.LSTM(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE,
                               num_layers=NUM_LAYER, bidirectional=BIDIRECTIONAL,
                               dropout=DROPOUT)
        # 此时三维向量为[seq_len,batch_size,embedding_size]
        if BIDIRECTIONAL: # true为双向LSTM false单向LSTM
            self.decoder = nn.Linear(HIDDEN_SIZE * 2, LABEL_SIZE)
        else:
            self.decoder = nn.Linear(HIDDEN_SIZE * 1, LABEL_SIZE)

    def init_word_embedding(self,VOCAB):
        weight_matrix = VOCAB.vectors
        # 使用已经处理好的词向量
        self.word_embeddings = nn.Embedding(len(VOCAB), self.EMBEDDING_DIM)
        self.word_embeddings.weight.data.copy_(weight_matrix)

    def forward(self, sentence,device):
        # sentence [seq_len,batch_size]\
        indexed = [self.TEXT_VOCAB.stoi[t] for t in sentence]  # 在词典中获取index
        senetence_tensor = torch.LongTensor(indexed)
        senetence_tensor = senetence_tensor.unsqueeze(1).to(device)  # 获取向量化后的一句话矩阵
        embeddings = self.word_embeddings(senetence_tensor)
        # embedding[seq_len,batch_size,embedding_size]

        states,hidden = self.lstm(embeddings)
        # states[seq_len,batch_size,hidden_size]真正的输出
        # hidden[n,num_layer] N测试出来是2不知道代表的是什么，尚不知道hidden的意义，目前后面也没有用到的

        # torch.cat是将两个张量（tensor）拼接在一起
        # 这里用的最初始的状态和最后的状态拼起来作为分类的输入
        # 这一部分是有灵活性的，如果只用最后的状态，直接states[-1] 上面的linear(HIDDEN_SIZE, LABELS)
        # encoding = torch.cat([states[0], states[-1]], dim=1)
        # encoding [batch_size,hidden_size * 2]
        output = self.decoder(states[-1])
        # output [batch_size,label_size]
        return output
