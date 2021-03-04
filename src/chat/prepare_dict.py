# -*- coding: utf-8 -*-
# @Time    : 2020/12/23 7:41 PM
# @Author  : Kevin
from src import config
from src.utils import sentence_process
import pickle
from src.chat.word_sequence import WordSequence
from tqdm import tqdm


def prepare_dict():
    # 准备两个词库对象,一问一答
    ask_lines = open(config.project_home_path_prefix + "/data/chat/ask_cut_by_character_clean.txt", 'r').readlines()
    answer_lines = open(config.project_home_path_prefix + "/data/chat/answer_cut_by_character_clean.txt", 'r').readlines()

    ws1=WordSequence()
    ws2=WordSequence()


    for ask_line in tqdm(ask_lines,ascii=True,desc="fit ask数据中..."):
        ws1.fit(sentence_process.cut_sentence_by_character(ask_line.strip()))


    for answer_line in tqdm(answer_lines,ascii=True,desc="fit answer数据中..."):
        # 你麻痹
        ws2.fit(sentence_process.cut_sentence_by_character(answer_line.strip()))

    ws1.build_vocab()
    ws2.build_vocab()
    print(len(ws1))
    print(len(ws2))

    pickle.dump(ws1,open(config.chat_ask_word_sequence_model_path,"wb"))
    pickle.dump(ws2,open(config.chat_answer_word_sequence_model_path,"wb"))





if __name__ == '__main__':
    prepare_dict()
