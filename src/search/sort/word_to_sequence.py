# -*- coding: utf-8 -*-
# @Time    : 2020/12/23 2:26 PM
# @Author  : Kevin


class Word2Sequence:

    UNK_TAG="<UNK>"
    PAD_TAG = "<PAD>"

    UNK = 0
    PAD = 1

    '''
    1.初始化dict词典,加入初始字符,count词频统计
    '''
    def __init__(self):

        self.dict={
            self.UNK_TAG:self.UNK,
            self.PAD_TAG:self.PAD
        }

        self.counter={}

    '''
    2.接收单个wordlist,统计进词频count
    '''
    def fit(self,word_list):
        for word in word_list:
            self.counter[word]=self.counter.get(word,0)+1

    '''
    3.根据词频,造词典:最小,最大词频,词个数
    '''
    def build_vocab(self,min_count=1,max_count=None,max_features=None):
        '''
        :param min_count: 入库的最小词频
        :param max_count: 入库的最大词频
        :param max_features: 整个词库的大小
        :return:
        '''
        # 1.过滤counter
        if min_count is not None:
            self.counter={word:count for word,count in self.counter.items() if count>min_count}
        if max_count is not None:
            self.counter={word:count for word,count in self.counter.items() if count<max_count}
        if max_features is not None:
            self.counter=dict(sorted(self.counter.items(),reverse=True,key=lambda x:x[-1])[:max_features])

        # 根据counter,建立词典
        # 2.遍历counter,不断加入dict,key是word,value是索引,即dict的长度
        for word in self.counter:
            self.dict[word]=len(self.dict)

        # 3.不仅创建dict,还要创建reverse_dict
        self.reverse_dict=dict(zip(self.dict.values(),self.dict.keys()))


    '''
    4.接收文本,转数字序列:wordlist>sequence
    '''
    def transform(self,word_list,sequence_max=10):
        # 1.规定序列长度,短了补,长了切
        word_list_len=len(word_list)
        if word_list_len>sequence_max:
            word_list=word_list[:sequence_max]
        if word_list_len<sequence_max:
            #填充数组
            word_list=word_list+[self.PAD_TAG]*(sequence_max-len(word_list))
            # print(word_list)
        # 最后转成数字列表

        return [self.dict.get(word,self.UNK) for word in word_list]

    '''
    5.接收数字序列,转文本
    '''
    def inverse_transform(self,sequence_list):
        # 1.接收索引列表,调用self.reverse_dict转成真实文本word_list

        word_list=[self.reverse_dict.get(index,self.UNK_TAG) for index in sequence_list]

        return word_list

    def __len__(self):

        return len(self.dict)