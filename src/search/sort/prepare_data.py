# -*- coding: utf-8 -*-
# @Time    : 2020/12/23 2:27 PM
# @Author  : Kevin

import config
from utils.sentence_process import cut_sentence_by_character
from search.sort.word_to_sequence import Word2Sequence
import pickle

def prepare_dict_model():

    lines=open(config.sort_all_file_path,"r").readlines()
    ws=Word2Sequence()
    lines=[cut_sentence_by_character(line) for line in lines]

    for line in lines:
        ws.fit(line)

    ws.build_vocab()

    pickle.dump(ws,open(config.sort_ws_model_path,"wb"))


def test_dict_model():
    sentence="如何在linux下安装storm"
    ws=pickle.load(open(config.sort_ws_model_path,"rb"))
    sequence=ws.transform(cut_sentence_by_character(sentence))
    print(cut_sentence_by_character(sentence))
    print(sequence)


def make_data_file():
    with open(config.sort_label_file_path,"w+") as file:
        for i in range(96339):
            if i%3==0:
                file.write("0"+"\n")
            else:
                file.write("1" + "\n")





if __name__ == '__main__':
    # prepare_dict_model()
    # test_dict_model()
    make_data_file()