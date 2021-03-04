# -*- coding: utf-8 -*-
# @Time    : 2020/12/23 7:40 PM
# @Author  : Kevin


class WordSequence():
    UNK=0
    PAD=1
    START=2
    END=3

    UNK_TAG="<UNK>"
    PAD_TAG="<PAD>"
    START_TAG="<START>"
    END_TAG="<END>"


    def __init__(self):
        self.dict={
            self.UNK_TAG:self.UNK,
            self.PAD_TAG:self.PAD,
            self.START_TAG:self.START,
            self.END_TAG:self.END,
        }

        self.count={}


    def fit(self,word_list):

        for word in word_list:
            # 遍历词,统计词频
            self.count[word]=self.count.get(word,0)+1

        self.fited=True

    def build_vocab(self,count_min=None,count_max=None,dict_size=None):
        # json,按照cout排序
        self.count=dict(sorted(self.count.items(),key=lambda x:x[1],reverse=True))

        if count_min is not None:

            self.count=dict([(item[0],item[1]) for item in  self.count.items() if item[1] >= count_min])

        if count_max is not None:
            self.count = dict([(item[0], item[1]) for item in self.count.items() if item[1] < count_max])

        if dict_size is not None:
            self.count=dict(sorted(self.count.items(),key=lambda x:x[1],reverse=True)[:dict_size])

        words=self.count.keys()

        for word in words:
            # 将词放到字典里,index就是字典的长度
            self.dict[word]=len(self.dict)

    #     制做inverse_dict
        self.inverse_dict=dict(zip(self.dict.values(),self.dict.keys()))

    def transform(self,word_list,sentence_max,add_end=False):

        assert self.fited,"还未进行fit操作"

        word_sentence=[self.dict.get(word,self.UNK) for word in word_list]

        if sentence_max is not None:

            word_sentence_len=len(word_sentence)

            if sentence_max>len(word_sentence):

                if add_end:
                    # 句子短,加END>在句子尾巴加END,然后PAD到sentence_max
                    word_sentence+=[self.END]+[self.PAD for _ in range(sentence_max-word_sentence_len)]
                else:
                    word_sentence += [self.PAD for _ in range(sentence_max - word_sentence_len)]
            else:
                if add_end:
                    # 句子长,加END>句子截成sentence_max,然后末尾再加一个END,返回长度sentence_max+1
                    word_sentence=word_sentence[:sentence_max]+[self.END]
                else:
                    # 返回长度sentence_max
                    word_sentence = word_sentence[:sentence_max]
        else:
            if add_end:
                word_sentence+=[self.END]


        return word_sentence


    def inverse_transform(self,word_sequence):


        result=[]
        for index in word_sequence:
            # if index==self.END:
            #     break
            result.append(self.inverse_dict.get(index,self.UNK_TAG))

        return result


    def __len__(self):

        return len(self.dict)