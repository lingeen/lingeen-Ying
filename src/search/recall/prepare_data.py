# -*- coding: utf-8 -*-
# @Time    : 2020/12/22 10:08 PM
# @Author  : Kevin
from utils import sentence_process

if __name__ == '__main__':
    new_merged_q_path="/Users/kevin/Downloads/work/pycharm/Ying/data/search/new_merged_q.txt"
    question_cut_by_character_path="/Users/kevin/Downloads/work/pycharm/Ying/data/search/question_cut_by_character.txt"

    with open(new_merged_q_path,"r") as new_merged_q_file:
        with open(question_cut_by_character_path,"w+") as question_cut_by_character_file:

            lines=new_merged_q_file.readlines()

            questions=[" ".join(sentence_process.cut_sentence_by_character(line.strip())) for line in lines]

            for question in questions:
                question_cut_by_character_file.write(question+"\n")



