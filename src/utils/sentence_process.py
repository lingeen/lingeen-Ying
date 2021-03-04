# -*- coding: utf-8 -*-
# @Time    : 2020/12/22 6:21 PM
# @Author  : Kevin
import jieba
from src import config
import string
import jieba.posseg as pos
import pickle
from src import lib

jieba.load_userdict(config.user_dict_path)


letters=string.ascii_letters+"1234567890"


def cut_sentence_by_character(sentence,filter_stopwords=False):

    result=[]

    temp=[]

    sentence_len=len(sentence)

    for i in range(sentence_len):

        char=sentence[i].lower()

        if char not in letters:

            if len(temp)>0:
                result.append("".join(temp))
                temp=[]

            result.append(char)

            if i ==sentence_len-1 and len(temp)>0:
                result.append("".join(temp))
                temp=[]

        elif char in letters:

            temp.append(char)

            if i ==sentence_len-1 and len(temp)>0:
                result.append("".join(temp))
                temp=[]
    if filter_stopwords:
        result=[char for char in result if char.strip()+"\n" not in lib.stop_words]

    return result

def cut_sentence_by_word(sentence,filter_stopwords=False):

    word_list=jieba.lcut(sentence)

    if filter_stopwords:
        word_list=[word for word in word_list if word.strip()+"\n" not in lib.stop_words]

    return word_list



def extract_sentence_title(sentence):

    lines = open("/Users/kevin/Downloads/work/pycharm/Ying/data/keywords.txt", 'r').readlines()

    lines = [line.strip().split(" ") for line in lines]

    lines = [(line[0], line[1]) for line in lines]

    word_flag_pair=[(i.word,i.flag) for i in pos.lcut(sentence) if (i.word,'kc') in lines]

    return word_flag_pair


def get_dict_model():
    return pickle.load(open(config.sort_ws_model_path,"rb"))





