# -*- coding: utf-8 -*-
# @Time    : 2020/12/22 2:26 PM
# @Author  : Kevin

from src import config
from src.search.sort.word_to_sequence import Word2Sequence
from src.chat.word_sequence import WordSequence
import os
import pickle
import torch

stop_words=open(config.stop_words_path,"r").readlines()

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

word_sequence_model=pickle.load(open(config.sort_ws_model_path,"rb"))
if os.path.exists(config.chat_word_sequence_model_path):
    chat_word_sequence_model=pickle.load(open(config.chat_word_sequence_model_path,"rb"))

if os.path.exists(config.chat_ask_word_sequence_model_path):
    chat_ask_word_sequence_model=pickle.load(open(config.chat_ask_word_sequence_model_path,"rb"))

if os.path.exists(config.chat_answer_word_sequence_model_path):
    chat_answer_word_sequence_model=pickle.load(open(config.chat_answer_word_sequence_model_path,"rb"))

ws=Word2Sequence()

chat_ws=WordSequence()


if __name__ == '__main__':
    print(config.chat_answer_word_sequence_model_path)

