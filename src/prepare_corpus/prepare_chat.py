# -*- coding: utf-8 -*-
# @Time    : 2020/12/27 10:58 PM
# @Author  : Kevin
from src.utils.sentence_process import cut_sentence_by_character
from tqdm import tqdm
from src import config

def cut_chat_data_by_character():

    asks=open(config.project_home_path_prefix+"/data/chat/ask.txt",'r').readlines()

    with open(config.project_home_path_prefix+"/data/chat/ask_cut_by_character.txt",'w+') as file:
        for line in tqdm(asks,ascii=True,desc="正在处理问题语料..."):
            file.write(" ".join(cut_sentence_by_character(line)))

    answers = open(config.project_home_path_prefix+"/data/chat/answer.txt", 'r').readlines()

    with open(config.project_home_path_prefix+"/data/chat/answer_cut_by_character.txt", 'w+') as file:
        for line in tqdm(answers,ascii=True,desc="正在处理回答语料..."):

            file.write(" ".join(cut_sentence_by_character(line)))

def clean_blank_pair():
    asks_blank_index=[]
    answers_blank_index=[]

    asks = open(config.project_home_path_prefix + "/data/chat/ask_cut_by_character.txt", 'r').readlines()
    answers = open(config.project_home_path_prefix + "/data/chat/answer_cut_by_character.txt", 'r').readlines()

    for index in range(len(asks)):
        if len(asks[index].strip())==0:
            asks_blank_index.append(index)

    for index in range(len(answers)):
        if len(answers[index].strip())==0:
            answers_blank_index.append(index)

    blank_index=asks_blank_index + answers_blank_index

    with open(config.project_home_path_prefix + "/data/chat/answer_cut_by_character_clean.txt", 'w+') as file:
        for index in range(len(asks)):
            if index not in blank_index:
                file.write(asks[index])



    with open(config.project_home_path_prefix + "/data/chat/ask_cut_by_character_clean.txt", 'w+') as file:
        for index in range(len(answers)):
            if index not in blank_index:
                file.write(answers[index])



if __name__ == '__main__':
    # cut_chat_data_by_character()
    clean_blank_pair()
