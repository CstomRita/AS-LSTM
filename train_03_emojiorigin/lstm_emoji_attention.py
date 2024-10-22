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


        self.attention = nn.Linear(EMBEDDING_DIM,1)
        self.attn_combine = nn.Linear(2*EMBEDDING_DIM, EMBEDDING_DIM)
        self.attn = nn.Linear(HIDDEN_SIZE + EMBEDDING_DIM, 1)

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


        return emoji_tensor.to(device),senetence_tensor.to(device),hasEmoji,hasSentence

    def attention_net(self, word_embedding,h_n, emoji_attention_vector):
        # print(word_embedding.size()) 1x300
        # print(emoji_attention_vector.size()) 1x1x300
        # print(h_n.size()) 2x1x128
        '''
        1 word_embedding 和 prev_hidden结合
        temp [batch_size,embedding_dim+hidden_size]
        self.attn 线性层input[batch_size,embedding_dim+hidden_size] output [batch_size,1] 这个1可以看做max_emoji_len
        attn_weights[batch_size,1]
        '''
        temp = torch.cat((word_embedding, h_n[0]), 1)
        attn_weights = F.softmax(self.attn(temp), dim=1)
        '''
        2 得到的结果和emoji的上下文语义向量做乘积
        attn_weights[batch_size,1]---->[1,batch_size,1]
        emoji_attention_vector[1,batch_size,hidden_size]
        bmm------->[1,batch_size,1] X [1,batch_size, HIDDEN_SIZE]] ------>[1,batch_size,hidden_size]
        '''
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 emoji_attention_vector)
        '''
        3 乘积结果和word_embedding拼接起来，经过一个线性层得到最后结果
        attn_applied[0] [batch_size,hidden_size]]
        attn_combine 线性层input[batch_size,embedding_dim+hidden_size] output [batch_size,embedding_dim]
        output[1,batch_size,embedding_dim]
        '''
        output = torch.cat((word_embedding, attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        return output



    # 单字单字，一个cell一个cell的训练
    def single_word_train(self,word_embedding,emoji_attention_vector,hidden):
        # 融合注意力机制
        attention_out = self.attention_net(word_embedding, hidden[0], emoji_attention_vector)
        lstm_out, hidden = self.lstm(attention_out, hidden)
        '''
        如何传递hidden，就是将hidden作为返回值返回，在train.py时再次赋值训练
        '''
        return lstm_out,hidden

    def forward(self, sentences,all_emojis,device):
        emoji_tensor, senetence_tensor, hasEmoji, hasSentence = self.get_tensor(all_emojis, sentences,
                                                                                device)
        # 1 表情符语义向量为：表情符词向量的均值
        emoji_embeddings = self.emoji_embeddings(emoji_tensor)
        emoji_ave_embedding = torch.mean(emoji_embeddings,0,True) # 1 X 1 X 300

        # 2 以sentences分词结果
        sentence_embeddings = self.word_embeddings(senetence_tensor)
        # word_count * 1 * 300

        # 3 sentence_embedding 和 emoji_ave_embeddingx做注意力机制
        hidden = self.init_hidden(batch_size=1)
        for sentence_embedding in sentence_embeddings:
            # word_embedding[batch_size,bedding_dim]
            cell_out, hidden = self.single_word_train(
                word_embedding=sentence_embedding,
                hidden=hidden,
                emoji_attention_vector=emoji_ave_embedding)
        output = self.hidden2label(cell_out[-1])
        return output

if __name__ == '__main__':
    a = torch.Tensor([[[1, 1, 1,1,1]],[[2, 2, 2,2,2]],[[2, 2, 2,2,2]],[[2, 2, 2,2,2]]])
    b = torch.mean(a, 0, True)
    print(a.size())
    print(b)
    print(b.size())