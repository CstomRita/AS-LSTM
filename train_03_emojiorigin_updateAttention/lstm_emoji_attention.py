# -*- coding: utf-8 -*-
# @File  : lstm_split.py
# @Author: ChangSiteng
# @Date  : 2019-07-06
# @Desc  :
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable



class EMOJI_ATTENTION_LSTM(nn.Module):
    def __init__(self, EMOJI_VOCAB,TEXT_VOCAB, EMBEDDING_DIM, INPUT_SIZE, HIDDEN_SIZE, NUM_LAYER, BIDIRECTIONAL,
                 DROPOUT, LABEL_SIZE, BATCH_SIZE):
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

        # 这个是为了学习所有分句整合结果的
        # 分句[batch_size,hidden_size * numlayer]
        # m 个分句[m,hidden_size * numlaye],要求输出[batch_size,hidden_size * numlayer]
        # lstm输入[seq_len,batch_size,input_size] 输出[batch_size,hidden_size * numlayer]
        self.sentence_lstm = nn.LSTM(input_size=HIDDEN_SIZE, hidden_size=HIDDEN_SIZE,
                               num_layers=NUM_LAYER, bidirectional=BIDIRECTIONAL,
                               dropout=DROPOUT)

        self.attention = nn.Linear(EMBEDDING_DIM,1)
        self.attn_combine = nn.Linear(2*EMBEDDING_DIM, EMBEDDING_DIM)
        self.attn = nn.Linear(HIDDEN_SIZE *2 , 1)

    def init_hidden2label(self):
        sentence_num = 1
        if self.BIDIRECTIONAL: # true为双向LSTM false单向LSTM
            self.hidden2label = nn.Linear(self.HIDDEN_SIZE * 2  * sentence_num, self.LABEL_SIZE)
        else:
            self.hidden2label = nn.Linear(self.HIDDEN_SIZE * sentence_num, self.LABEL_SIZE)

    def init_word_embedding(self,VOCAB):
        weight_matrix = VOCAB.vectors
        # 使用已经处理好的词向量
        self.word_embeddings = nn.Embedding(len(VOCAB), self.EMBEDDING_DIM)
        self.word_embeddings.weight.data.copy_(weight_matrix)

    def init_emoji_embedding(self,VOCAB):
        weight_matrix = VOCAB.vectors
        # 使用已经处理好的词向量
        self.emoji_embeddings = nn.Embedding(len(VOCAB), self.HIDDEN_SIZE)
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


        return emoji_tensor.to(device),senetence_tensor.to(device),hasEmoji,hasSentence

    def forward(self, sentences,all_emojis,device):
        emoji_tensor, senetence_tensor, hasEmoji, hasSentence = self.get_tensor(all_emojis, sentences,
                                                                                device)
        # 1 表情符语义向量为：表情符词向量的均值
        emoji_embeddings = self.emoji_embeddings(emoji_tensor)
        emoji_ave_embedding = torch.mean(emoji_embeddings,0,True)

        # 2 以sentences分词结果
        sentence_embeddings = self.word_embeddings(senetence_tensor)
        # word_count * 1 * 300

        # 3 sentences进入lstm
        lstm_out, hidden = self.lstm(sentence_embeddings)
        # lstm_out.size() ----> len(sentences) * 1 * 128


        # 4 sentence_embeddings 和 emoji_ave_embeddingx做注意力机制
        '''
        4.1通过看sentence_embeddings 和 emoji_ave_embedding相似度得到权重矩阵，通过expand emoji_ave_embedding至 length * 300
        # 再和sentence_embeddings_permute cat成 length * 600
        # 通过线性层 length * 600 ---> length * 1的一维矩阵代表各个单词的权重
        '''
        '''
        sentence_embeddings_permute = sentence_embeddings.permute(1,0,2)[0]
        emoji_ave_embeddings = emoji_ave_embedding[0].expand(sentence_embeddings_permute.size())
        temp = torch.cat((sentence_embeddings_permute, emoji_ave_embeddings), 1)
        attn_weights = F.softmax(self.attn(temp), dim=1)
        '''
        '''
        策略二：lstm_out 和 emoji_ave_embedding相似度得到权重矩阵
        emoji_ave_embedding 1 * 1 * 128 ----> 扩展成 1 * n * 128
        lstm_out n * 1 * 128----> 1 * n * 128----> n * 128
        temp----> n * 256
        '''
        lstm_out_permute = lstm_out.permute(1, 0, 2)[0]
        emoji_ave_embeddings = emoji_ave_embedding[0].expand(lstm_out_permute.size())
        temp = torch.cat((lstm_out_permute, emoji_ave_embeddings), 1)
        attn_weights = F.softmax(self.attn(temp), dim=1)

        # 4.2 权重矩阵和lstm_out相乘相加得到 1 X 1 X hidden_size
        # attn_weights---> 7X1---->需要改成 1X1X7的格式
        # lstm_out--> 7 * 1 * 128 改为 1 * 7 * 128的格式
        lstm_out_attention = lstm_out.permute(1, 0, 2)
        attn_weights_attention = attn_weights.unsqueeze(0).permute(0, 2, 1)

        attn_applied = torch.bmm(attn_weights_attention,
                                 lstm_out_attention)
        output = self.hidden2label(attn_applied[-1])
        return output

if __name__ == '__main__':
    a = torch.Tensor([[[1, 1, 1,1,1]],[[2, 2, 2,2,2]],[[2, 2, 2,2,2]],[[2, 2, 2,2,2]]])
    b = torch.mean(a, 0, True)
    print(a.size())
    print(b)
    print(b.size())