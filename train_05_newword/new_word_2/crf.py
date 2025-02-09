# -*- coding: utf-8 -*-
'''
 @File  : crf.py
 @Author: ChangSiteng
 @Date  : 2019-12-21
 @Desc  :  BiLSTM-CRF 模型 参考 https://www.v2ex.com/t/554597
 '''

# import-path
import sys
import os

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
'''
帮助函数
'''
def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_sequence(seq, to_ix):
    idxs = []
    for w in seq:
        if(w in to_ix):
            idxs.append(to_ix[w])
        else:
            idxs.append(0)
    return torch.tensor(idxs, dtype=torch.long).to(device)


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)].to(device)
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1]).to(device)
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast))).to(device)

START_TAG = "<START>"
STOP_TAG = "<STOP>"

class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim).to(device)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True).to(device)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size).to(device)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size).to(self.device))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        if torch.cuda.is_available():
            h0 = Variable(torch.randn(2, 1, self.hidden_dim // 2)).cuda()
            c0 = Variable(torch.randn(2, 1, self.hidden_dim // 2)).cuda()
        else:
            h0 = Variable(torch.randn(2, 1, self.hidden_dim // 2))
            c0 = Variable(torch.randn(2, 1, self.hidden_dim // 2))
        return (h0,c0)

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tagset_size), -10000.).to(device)
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1).to(device)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1).to(self.device)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var).to(device)
        return alpha

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embeddings = self.word_embeds(sentence.to(device))
        embeds = embeddings.view(len(sentence), 1, -1).to(device)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        isException = False
        score = torch.zeros(1).to(self.device)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long).to(self.device), tags]).to(device)
        for i, feat in enumerate(feats):
            try:
                score = score + \
                    self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
            except BaseException:
                print("exception:",i,"length",len(feat))
                print(i,"length",len(tags))
                isException = True

        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score,isException

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.).to(device)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1).to(device)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score,isException = self._score_sentence(feats, tags)
        return (forward_score - gold_score),isException

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("crf_test!")
    START_TAG = "<START>"
    STOP_TAG = "<STOP>"
    EMBEDDING_DIM = 5
    HIDDEN_DIM = 4

    # Make up some training data
    # [([词，词],[tag tag])],([]),([])
    training_data = [(
        "the wall street journal reported today that apple corporation made money".split(),
        "B I I I O O O B I O O".split()
    ), (
        "georgia tech is a university in georgia".split(),
        "B I O O O O B".split()
    )]
    # print(training_data)

    word_to_ix = {}
    for sentence, tags in training_data:
        for word in sentence:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
    # print(word_to_ix) # word_to_ix记录非重复的词

    tag_to_ix = {"s": 0, "b": 1, "e": 2, "m": 3, START_TAG: 4, STOP_TAG: 5}  # 标签

    model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

    # Check predictions before training
    with torch.no_grad():
        precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
        precheck_tags = torch.tensor([tag_to_ix[t] for t in training_data[0][1]], dtype=torch.long).to(device)
        print(model(precheck_sent))

    # Make sure prepare_sequence from earlier in the LSTM section is loaded
    for epoch in range(2):  # again, normally you would NOT do 300 epochs, it is toy data
        for sentence, tags in training_data:

            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            # Step 2. Get our inputs ready for the network, that is,
            # turn them into Tensors of word indices.
            sentence_in = prepare_sequence(sentence, word_to_ix)
            targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long).to(device)
            # Step 3. Run our forward pass.
            loss,isException = model.neg_log_likelihood(sentence_in, targets)
            # Step 4. Compute the loss, gradients, and update the parameters by
            # calling optimizer.step()
            loss.backward()
            optimizer.step()



    # Check predictions after training
    with torch.no_grad():
        for data in training_data:
            print(data[0])
            precheck_sent = prepare_sequence(data[0], word_to_ix)
            (tensor ,tags) = model(precheck_sent)
            words = []
