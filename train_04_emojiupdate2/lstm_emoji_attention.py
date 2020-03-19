# -*- coding: utf-8 -*-
# @File  : lstm_split.py
# @Author: ChangSiteng
# @Date  : 2019-07-06
# @Desc  :
import os

import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from PIL import Image
'''
这里的LSTM和train_02中的LSTM在batch_size上的处理是不同的
train_02中的batch是通过TorchText中的iterator实现的，传入LSTM的sentence就是带有batch——size的
但是这里由于要各个分局合并成一个整句，无法使用iterator，使用的是LSTM里的batch——size，需要显式声明，也就是在hidden中创建的
'''

class EMOJI_ATTENTION_LSTM(nn.Module):
    def __init__(self, EMOJI_VOCAB,TEXT_VOCAB, EMBEDDING_DIM, INPUT_SIZE, HIDDEN_SIZE, NUM_LAYER, BIDIRECTIONAL, DROPOUT, LABEL_SIZE, BATCH_SIZE):
        super(EMOJI_ATTENTION_LSTM, self).__init__()

        self.EMBEDDING_DIM = EMBEDDING_DIM
        self.NUM_LAYER = NUM_LAYER
        self.HIDDEN_SIZE = HIDDEN_SIZE
        self.BATCH_SIZE = BATCH_SIZE
        self.USE_GPU = torch.cuda.is_available()
        self.TEXT_VOCAB = TEXT_VOCAB
        self.BIDIRECTIONAL = BIDIRECTIONAL
        self.LABEL_SIZE = LABEL_SIZE
        self.EMOJI_VOCAB = EMOJI_VOCAB

        self.init_word_embedding(TEXT_VOCAB)
        self.init_emoji_embedding(EMOJI_VOCAB)
        self.init_hidden_value = self.init_hidden()
        self.init_hidden2label()

        self.lstm = nn.LSTM(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE,
                               num_layers=NUM_LAYER, bidirectional=BIDIRECTIONAL,
                               dropout=DROPOUT)
        # 此时三维向量为[seq_len,batch_size,embedding_size]

        self.emoji_lstm = nn.LSTM(input_size=INPUT_SIZE, hidden_size=EMBEDDING_DIM,
                               num_layers=NUM_LAYER, bidirectional=BIDIRECTIONAL,
                               dropout=DROPOUT)

        # 这个是为了学习所有分句整合结果的
        # 分句[batch_size,hidden_size * numlayer]
        # m 个分句[m,hidden_size * numlaye],要求输出[batch_size,hidden_size * numlayer]
        # lstm输入[seq_len,batch_size,input_size] 输出[batch_size,hidden_size * numlayer]
        self.sentence_lstm = nn.LSTM(input_size=HIDDEN_SIZE, hidden_size=HIDDEN_SIZE,
                               num_layers=NUM_LAYER  , bidirectional=False,
                               dropout=DROPOUT)

        self.attention = nn.Linear(EMBEDDING_DIM,1)
        self.attn_combine = nn.Linear(EMBEDDING_DIM*2, EMBEDDING_DIM)
        self.attn = nn.Linear(EMBEDDING_DIM * 2, 1)

        '''
        CNN
        '''
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=16, kernel_size=5,
                      stride=1, padding=2),  # (4，36，36)
            # 想要con2d卷积出来的图片尺寸没有变化, padding=(kernel_size-1)/2
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # (16,18,18)
        )
        self.conv2 = nn.Sequential(  # (16,18,18)
            nn.Conv2d(16, 32, 5, 1, 2),  # (32,18,18)
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)  # (32,9,9)
        )
        # fully connected layer
        self.fc = nn.Linear(32 * 9 * 9, EMBEDDING_DIM)
        self.fc1 = nn.Linear(EMBEDDING_DIM * 2, EMBEDDING_DIM)

    def init_hidden2label(self):
        sentence_num = 1
        if self.BIDIRECTIONAL: # true为双向LSTM false单向LSTM
            self.hidden2label = nn.Linear(self.HIDDEN_SIZE * 2  * sentence_num, self.LABEL_SIZE)
        else:
            self.hidden2label = nn.Linear(self.HIDDEN_SIZE * 2, self.LABEL_SIZE)

    def init_word_embedding(self,VOCAB):
        weight_matrix = VOCAB.vectors
        # 使用已经处理好的词向量
        self.word_embeddings = nn.Embedding(len(VOCAB), self.EMBEDDING_DIM)
        self.word_embeddings.weight.data.copy_(weight_matrix)

    def init_emoji_embedding(self,VOCAB):
        weight_matrix = VOCAB.vectors
        # 使用已经处理好的词向量
        self.emoji_embeddings = nn.Embedding(len(VOCAB), self.EMBEDDING_DIM)
        self.emoji_embeddings.weight.data.copy_(weight_matrix)

    def init_hidden(self, batch_size=None):
        if batch_size is None:
            batch_size = 1

        if self.USE_GPU:
            h0 = Variable(torch.zeros(self.NUM_LAYER, batch_size, self.HIDDEN_SIZE).cuda())
            c0 = Variable(torch.zeros(self.NUM_LAYER, batch_size, self.HIDDEN_SIZE).cuda())
        else:
            h0 = Variable(torch.zeros(self.NUM_LAYER, batch_size, self.HIDDEN_SIZE))
            c0 = Variable(torch.zeros(self.NUM_LAYER, batch_size, self.HIDDEN_SIZE))
        return (h0, c0)



    def get_tensor(self,emojis,sentence,device):
        if len(emojis) > 0:  # 表示此分句下是有表情符号的，不一定只有一个可能有多个
            indexed = [self.EMOJI_VOCAB.stoi[t] for t in emojis]  # 在词典中获取index
            emoji_tensor = torch.LongTensor(indexed)
            emoji_tensor = emoji_tensor.unsqueeze(1)  # 向量化的一个分句的所有表情矩阵
            hasEmoji = True
        else : # 如果没有表情符号如何处理？？
            # 设置一个None NUK
            indexed = [self.EMOJI_VOCAB.stoi['']]
            emoji_tensor = torch.LongTensor(indexed)
            emoji_tensor = emoji_tensor.unsqueeze(1)  # 向量化的一个分句的所有表情矩阵
            hasEmoji = False

        if len(sentence) > 0: #分句下有汉字
            indexed = [self.TEXT_VOCAB.stoi[t] for t in sentence]  # 在词典中获取index
            senetence_tensor = torch.LongTensor(indexed)
            senetence_tensor = senetence_tensor.unsqueeze(1)  # 获取向量化后的一句话矩阵
            hasSentence = True
        else : # 如果没有汉字，只有表情符号，如何处理??设置一个None NUK
            indexed = [self.EMOJI_VOCAB.stoi['<pad>']]
            senetence_tensor = torch.LongTensor(indexed)
            senetence_tensor = senetence_tensor.unsqueeze(1)  # 获取向量化后的一句话矩阵
            hasSentence = False

        emoji_embeddings = self.emoji_embeddings(emoji_tensor)
        return emoji_embeddings,senetence_tensor.to(device),hasEmoji,hasSentence

    '''
     获取表情符语义向量策略2：利用图片CNN和词向量的平均向量
     '''
    def get_tensor1(self,emojis,sentence,device):
        if len(emojis) > 0:  # 表示此分句下是有表情符号的，不一定只有一个可能有多个
            # indexed = [self.EMOJI_VOCAB.stoi[t] for t in emojis]  # 在词典中获取index
            emoji_embeddings = []
            for t in emojis:
                indexed = [self.EMOJI_VOCAB.stoi[t]]
                tensor = torch.LongTensor(indexed).unsqueeze(1)
                emoji_embedding = self.emoji_embeddings(tensor.to(device))
                # 使用PIL读取图片
                path = "../data/weibo_emoji/emojis/" + t +".png"
                if (os.path.exists(path)):
                    img_pil = Image.open(path)  # PIL.Image.Image对象
                    img_pil_1 = numpy.array(img_pil)  # (H x W x C), [0, 255], RGB
                    try:
                        tensor_pil = torch.from_numpy(numpy.transpose(img_pil_1, (2, 0, 1))).float().unsqueeze(0).to(device) # 1 * 4 * 36 *36
                        # np.transpose(xxx, (2, 0, 1))  # 将 H x W x C 转化为 C x H x W
                        tensor_pil = self.conv1(tensor_pil)
                        tensor_pil = self.conv2(tensor_pil)
                        tensor_pil = tensor_pil.view(tensor_pil.size(0), -1)
                        tensor_pil = F.relu(self.fc(tensor_pil))  # 第五层为全链接，ReLu激活函数
                        # print(tensor_pil.size()) # 1 * 300
                        temp = torch.cat((emoji_embedding[0],tensor_pil),0)
                        emoji_embedding = torch.mean(temp, 0, True).unsqueeze(0)
                    except BaseException:
                        print(emojis,'--',sentence,'---',path,'---',img_pil_1.shape)

                if len(emoji_embeddings) == 0:
                    emoji_embeddings = emoji_embedding
                else :
                    emoji_embeddings = torch.cat((emoji_embedding,emoji_embeddings),0)
            hasEmoji = True
        else : # 如果没有表情符号如何处理？？
            # 设置一个None NUK
            indexed = [self.EMOJI_VOCAB.stoi['']]
            emoji_tensor = torch.LongTensor(indexed)
            emoji_tensor = emoji_tensor.unsqueeze(1)  # 向量化的一个分句的所有表情矩阵
            hasEmoji = False
            emoji_embeddings = self.emoji_embeddings(emoji_tensor.to(device))

        if len(sentence) > 0: #分句下有汉字
            indexed = [self.TEXT_VOCAB.stoi[t] for t in sentence]  # 在词典中获取index
            senetence_tensor = torch.LongTensor(indexed)
            senetence_tensor = senetence_tensor.unsqueeze(1)  # 获取向量化后的一句话矩阵
            hasSentence = True
        else : # 如果没有汉字，只有表情符号，如何处理??设置一个None NUK
            indexed = [self.EMOJI_VOCAB.stoi['<pad>']]
            senetence_tensor = torch.LongTensor(indexed)
            senetence_tensor = senetence_tensor.unsqueeze(1)  # 获取向量化后的一句话矩阵
            hasSentence = False


        return emoji_embeddings,senetence_tensor.to(device),hasEmoji,hasSentence

    '''
        获取表情符语义向量策略2：利用图片CNN和词向量的拼接
        CNN 1*300
        词向量 1*300
        ----> 1 * 600 ----> 线性层 1* 300
        ----> n * 1 * 300
        '''

    def get_tensor2(self, emojis, sentence, device):
        if len(emojis) > 0:  # 表示此分句下是有表情符号的，不一定只有一个可能有多个
            # indexed = [self.EMOJI_VOCAB.stoi[t] for t in emojis]  # 在词典中获取index
            emoji_embeddings = []
            for t in emojis:
                indexed = [self.EMOJI_VOCAB.stoi[t]]
                tensor = torch.LongTensor(indexed).unsqueeze(1)
                emoji_embedding = self.emoji_embeddings(tensor.to(device))
                # 使用PIL读取图片
                path = "../data/weibo_emoji/emojis/" + t + ".png"
                if (os.path.exists(path)):
                    img_pil = Image.open(path)  # PIL.Image.Image对象
                    img_pil_1 = numpy.array(img_pil)  # (H x W x C), [0, 255], RGB
                    try:
                        tensor_pil = torch.from_numpy(numpy.transpose(img_pil_1, (2, 0, 1))).float().unsqueeze(0).to(
                            device)  # 1 * 4 * 36 *36
                        # np.transpose(xxx, (2, 0, 1))  # 将 H x W x C 转化为 C x H x W
                        tensor_pil = self.conv1(tensor_pil)
                        tensor_pil = self.conv2(tensor_pil)
                        tensor_pil = tensor_pil.view(tensor_pil.size(0), -1)
                        tensor_pil = self.fc(tensor_pil)  # 第五层为全链接，ReLu激活函数
                    except BaseException:
                        tensor_pil = torch.zeros(self.EMBEDDING_DIM).unsqueeze(0).to(device)
                        # print(emojis, '--', sentence, '---', path, '---', img_pil_1.shape)
                else:
                    tensor_pil = torch.zeros(self.EMBEDDING_DIM).unsqueeze(0).to(device)

                temp = torch.cat((emoji_embedding[0], tensor_pil), 1)
                emoji_embedding = self.fc1(temp).unsqueeze(0)

                if len(emoji_embeddings) == 0:
                    emoji_embeddings = emoji_embedding
                else:
                    emoji_embeddings = torch.cat((emoji_embedding, emoji_embeddings), 0)
            hasEmoji = True

        else:  # 如果没有表情符号如何处理？？
            # 设置一个None NUK
            indexed = [self.EMOJI_VOCAB.stoi['']]
            emoji_tensor = torch.LongTensor(indexed)
            emoji_tensor = emoji_tensor.unsqueeze(1)  # 向量化的一个分句的所有表情矩阵
            hasEmoji = False
            emoji_embeddings = self.emoji_embeddings(emoji_tensor.to(device))

        if len(sentence) > 0:  # 分句下有汉字
            indexed = [self.TEXT_VOCAB.stoi[t] for t in sentence]  # 在词典中获取index
            senetence_tensor = torch.LongTensor(indexed)
            senetence_tensor = senetence_tensor.unsqueeze(1)  # 获取向量化后的一句话矩阵
            hasSentence = True
        else:  # 如果没有汉字，只有表情符号，如何处理??设置一个None NUK
            indexed = [self.EMOJI_VOCAB.stoi['<pad>']]
            senetence_tensor = torch.LongTensor(indexed)
            senetence_tensor = senetence_tensor.unsqueeze(1)  # 获取向量化后的一句话矩阵
            hasSentence = False

        return emoji_embeddings, senetence_tensor.to(device), hasEmoji, hasSentence

    def get_emoji_vector(self, emoji_embeddings):
        lstm_out, (h_n, c_n) = self.emoji_lstm(emoji_embeddings)
        # emoji_embeddings[emoji_len,batch_size,embedding_size]
        return lstm_out


    def forward(self, sentences,all_emojis,device):
        # 这里的batch_size都是1，未做批量处理
        all_out = []

        for sentence_index, sentence in enumerate(sentences): # 借助enumerate函数循环遍历时获取下标
            emoji_embeddings,senetence_tensor,hasEmoji,hasSentence = self.get_tensor1(all_emojis[sentence_index],sentence,device)

            emoji_attention_vector = torch.mean(emoji_embeddings, 0, True)  # 1 X 1 X 300
            # print(emoji_embeddings.size(),'-----------',emoji_attention_vector.size())
            # emoji_attention_vector = self.get_emoji_vector(emoji_embeddings)

            sentence_embeddings = self.word_embeddings(senetence_tensor)


            '''
            策略一
            '''
            lstm_out, hidden = self.lstm(sentence_embeddings)
            sentence_embeddings_permute = sentence_embeddings.permute(1, 0, 2)[0]
            emoji_ave_embeddings = emoji_attention_vector[0].expand(sentence_embeddings_permute.size())
            temp = torch.cat((sentence_embeddings_permute, emoji_ave_embeddings), 1)
            attn_weights = F.softmax(self.attn(temp), dim=1)

            lstm_out_attention = lstm_out.permute(1, 0, 2)
            attn_weights_attention = attn_weights.unsqueeze(0).permute(0, 2, 1)

            attn_applied = torch.bmm(attn_weights_attention,
                                     lstm_out_attention)

            attention_out = attn_applied[0]

            '''
            策略2
            '''
            '''
            sentence_embeddings_permute = sentence_embeddings.permute(1, 0, 2)[0]
            emoji_ave_embeddings = emoji_ave_embedding[0].expand(sentence_embeddings_permute.size())
            temp = torch.cat((sentence_embeddings_permute, emoji_ave_embeddings), 1)
            attn_weights = F.softmax(self.attn(temp), dim=1)
            attn_weights_attention = attn_weights.unsqueeze(0).permute(0, 2, 1)

            attn_applied = torch.bmm(attn_weights_attention,
                                     sentence_embeddings_permute.unsqueeze(0))
            ## 拼接

            attn_applied = attn_applied[0].expand(sentence_embeddings_permute.size())
            temp = torch.cat((sentence_embeddings_permute, attn_applied), 1)
            attention_out = self.attn_combine(temp)
            '''

            # 所有分句的Attention输出 整合在一起 all_out[sentence_num,batch_size,hidden_size]
            if len(all_out) == 0:
                all_out = attention_out
            else:
                all_out = torch.cat((attention_out,all_out),0)
            # print(all_out.size())
        # 方案:将所有分句的输出经过额外一层LSTM学习
        # print(all_out.size())  # wordNum * 1 * 128
        all_out = all_out.unsqueeze(0).permute(1,0,2)
        all_out_lstm_out,all_out_lstm_hidden = self.sentence_lstm(all_out)
        all_out_lstm_encoding = torch.cat([all_out_lstm_out[0], all_out_lstm_out[-1]], dim=1)
        output = self.hidden2label(all_out_lstm_encoding)
        return output
