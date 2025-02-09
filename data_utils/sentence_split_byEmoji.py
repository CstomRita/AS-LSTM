# -*- coding: utf-8 -*-
# @File  : sentence_split.py
# @Author: ChangSiteng
# @Date  : 2019-06-26
# @Desc  : 1. 按照表情符来拆分子句
# emoji是个二维数组
# sentence_no_Emoji是个一维数组，存储按照emoji划分的子句
import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import json
import pickle
import re

import jieba

from data_utils.nlpcc_parse import Parse

punctuations = ['，', '。', '？', '~', '！', '、', '……']


'''
text是分词后的以一句话，以空格隔开
emoji是一个二维数组
'''
def sentence_no_emoji_split_tokenizer_byEMOJI(text,emoji):
    text.strip()
    words = []
    pattern = SentenceSplit.get_pattern()
    sentences = re.split(pattern, text)
    word = []
    for index, sentence in enumerate(sentences):
        temp = [word for word in sentence.split() if word.strip()]
        if (len(temp) > 0 and len(emoji[index]) > 0) or index == len(sentences)-1: # 有词 有表情符号 或者 最后一个
            word.extend(temp)
            words.append(word)  # extend，将一个微博中多个子句的分词结果合并成一个一维数组返回
            word = []
        elif len(temp) > 0 and len(emoji[index]) == 0: # 有词，没有表情符
            word.extend(temp)
        # append返回的是二维数组，表示的是各个分句下的分词结果
    # emoi是按照分句拆分的，如果想按照表情符，需要对emoji再处理
    # 这里做验证
    emojis = []
    for em in emoji:
        if len(em) > 0:
            emojis.append(em)
    if len(words) != len(emojis):
        print("")
    return words


class SentenceSplit:

    datas = []
    pattern = ''
    emoji_pattern = ''

    def __init__(self,path):
        parse = Parse("../data/nlpcc2014/emotion_label.json")
        self.datas = parse.parse(path)
        self.pattern = SentenceSplit.get_pattern()
        self.emoji_pattern = r'\[(\w*)\]'

    # 静态方法
    @staticmethod
    def get_pattern():
        pattern = ''
        for punctuation in punctuations:
            pattern += punctuation + "|"
        pattern = pattern[:-1]
        return pattern

    def sentence_split(self, path, iftrain):
        # 这里是获取的每一句话，需要做分词的工作，我们再在这里做将表情符号切分的工作
        emoji_all_count = 0
        emoji_all_type_count = 0
        emotionNum = {
            0: 0,
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: 0,
            6: 0,
            7: 0
        }
        print("原始语料统计：",len(self.datas))
        for example in self.datas[::-1]:
            # 正序删除列表中元素时，被删元素后面的值会向前顶，然后导致漏删。
            # 倒序删除元素时，被删元素前面的值不会向后靠，所以可以完整的遍历到列表中所有的元素。
            # print(sentence)
            sentence = example['sentence']
            sentence_no_emoji_split = ''
            emoji_list = []
            emoji_count = []
            # 1 首先表情符号的纯文本
            sentence_no_emoji = re.sub(self.emoji_pattern, '', sentence)
            # 2 按照标点符号切分子句
            short_sentences = re.split(self.pattern,sentence)
            if(len(short_sentences) > 1):
                # 表示有分句
                has_split = True
            else:
                has_split = False
            punctuations = re.findall(self.pattern,sentence) # 为了保持标点符号的一致
            # 3在每个子句中看是否有表情符号，这是因为子句后的表情符号会对子句产生影响
            emoji_sentence_count = 0
            emoji_split_sentence_num = 0 # 记录几个分句有表情符
            for short_sentence in short_sentences:
                if short_sentence.strip() == '':
                    continue
                emojis = re.findall(self.emoji_pattern, short_sentence)
                if(len(emojis) > 0):
                    emoji_split_sentence_num += 1
                emojidict = {}
                for emoji in emojis:
                    count = emojidict.setdefault(emoji,0) + 1
                    emojidict[emoji] = count
                # print(emojidict,"keys:",emojidict.keys(),"values",emojidict.values())
                emoji_list.append(list(emojidict.keys()))
                emoji_count.append(list(emojidict.values()))

                emoji_all_type_count += len(emojidict.keys())

                '''
                记录整个句子的表情符个数
                '''
                for value in list(emojidict.values()):
                    emoji_sentence_count += value
                    emoji_all_count += value

                # 4根据除去表情符号的子句，再分词
                short_entence_no_emoji = re.sub(r'\[(\w*)\]', '', short_sentence)
                sentence_no_emoji_split_temp = " ".join(self.word_split(short_entence_no_emoji))
                sentence_no_emoji_split = sentence_no_emoji_split + str(sentence_no_emoji_split_temp)
                index = short_sentences.index(short_sentence)
                if len(punctuations) > index:
                    sentence_no_emoji_split = sentence_no_emoji_split + punctuations[index]
            ''' 都存储
            '''
            # example['sentence_no_emoji'] = sentence_no_emoji
            # example['emoji'] = (emoji_list)
            # example['emoji_count'] = (emoji_count)
            # example['sentence_no_emoji_split'] = sentence_no_emoji_split
            # emotionNum[example['emotions']] += 1

            '只存储多个分句有表情符号的'

            # if(emoji_split_sentence_num > 1 and has_split) :
            #     example['sentence_no_emoji'] = sentence_no_emoji
            #     example['emoji'] = (emoji_list)
            #     example['emoji_count'] = (emoji_count)
            #     example['sentence_no_emoji_split'] = sentence_no_emoji_split
            #     emotionNum[example['emotions']] += 1
            # else:
            #     self.datas.remove(example)

            '只存储有分句的'
            #
            #             # if (has_split):
            #             #     example['sentence_no_emoji'] = sentence_no_emoji
            #             #     example['emoji'] = (emoji_list)
            #             #     example['emoji_count'] = (emoji_count)
            #             #     example['sentence_no_emoji_split'] = sentence_no_emoji_split
            #             #     emotionNum[example['emotions']] += 1
            #             # else:
            #             #     self.datas.remove(example)

            '只存储有表情符的'

            if (emoji_split_sentence_num > 0):
                example['sentence_no_emoji'] = sentence_no_emoji
                example['emoji'] = (emoji_list)
                example['emoji_count'] = (emoji_count)
                example['sentence_no_emoji_split'] = sentence_no_emoji_split_tokenizer_byEMOJI(sentence_no_emoji_split,emoji_list)
                example['sentence_no_emoji_split_origin'] = sentence_no_emoji_split.strip()
                emotionNum[example['emotions']] += 1
            else:
                self.datas.remove(example)


        #https://blog.csdn.net/weixin_43896398/article/details/85559172
        # torchtext能够读取的json文件和我们一般意义上的json文件格式是不同的（这也是比较坑的地方），我们需要把上面的数据处理成如下格式：
        #
        # {"source": "10 111 2 3", "target": "1 1 2 2"}
        # {"source": "10 111 2 3", "target": "1 1 2 2"}
        # {"source": "10 111 2 3", "target": "1 1 2 2"}
        # {"source": "10 111 2 3", "target": "1 1 2 2"}
        # {"source": "10 111 2 3", "target": "1 1 2 2"}
        #可以看到，里面的内容和通常的Json并无区别，每个字段采用字典的格式存储。
        # 不同的是，多个json序列中间是以换行符隔开的，而且最外面没有列表。


        '''
        统计语料
        '''
        print("所有表情符种类:",emoji_all_type_count)
        print("所有表情符个数:",emoji_all_count)
        print("过滤后数据集：",len(self.datas))
        print("过滤后语料统计", emotionNum)

        write_time = 0
        with open(path, 'w+') as fw:
            for example_data in self.datas:
                encode_json = json.dumps(example_data)
                # 一行一行写入，并且采用print到文件的方式
                print(encode_json, file=fw)
                write_time += 1

        # json_data = json.dumps(self.datas)
        # with open(path, 'w+',encoding='utf-8') as f_six: # w+用于读写，覆盖
        #     f_six.write(json_data)
        print("load data并保存在",path,",写了",write_time,"次")

        if iftrain:
        # 将分好的词划分出来，拼接到一起，方便glove训练
            with open(path[0:path.rfind("/")]+'/words_origin.txt','w+') as fw:
                for example_data in self.datas:
                        print(example_data['sentence_no_emoji_split_origin'],file=fw)
            print("分词TXT已经保存在words_origin.txt中")
        # 将表情符单词 供glove词向量
            with open(path[0:path.rfind("/")]+'/emojis_origin.txt','w+') as fw:
                for example_data in self.datas:
                        temp = ''

                        emojis_origin = example_data['emoji']
                        for emoji_origin in emojis_origin:
                            if len(emoji_origin) > 0:
                                for emoji_temp in emoji_origin:
                                 temp += emoji_temp + ' '
                        if len(temp) > 0 :
                            print(temp,file=fw)
            print("表情分词TXT已经保存在emojis_origin.txt中")


    def word_split(self,sentence):
        words = jieba.cut(sentence)
        return words


if __name__=='__main__':
    str = 'http://manualfile.s3.amazonaws.com/pdf/gti-chis-1-user-9fb-0-7a05a56f0b91.pdf'
    name = str[0:str.rfind("/")]
    print(name)
