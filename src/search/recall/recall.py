# -*- coding: utf-8 -*-
# @Time    : 2020/12/22 7:38 PM
# @Author  : Kevin

import fasttext
import pysparnn.cluster_index as ci
import config
import os
from utils import sentence_process
import pickle

'''
回忆:
模型建立阶段
模型使用阶段:接收用户问题,返回相似问题
'''
class Memory():


    def get_fasttext_model(self):

        if os.path.exists(config.recall_fasttext_vector_model_path):

            return fasttext.load_model(config.recall_fasttext_vector_model_path)

        else:
            fasttext_model=fasttext.train_supervised(config.recall_question_cut_by_character_path,epoch=20,lr=0.001,wordNgrams=1)

            fasttext_model.save_model(config.recall_fasttext_vector_model_path)

            return fasttext_model



    def get_pysprnn_model(self,fasttext):

        if os.path.exists(config.recall_pysparnn_cp_model_path):
            return pickle.load(open(config.recall_pysparnn_cp_model_path,"rb"))
        else:
            lines=open(config.recall_merged_q_path,"r").readlines()

            lines=[line.strip() for line in lines]

            quesions_string_cut=[" ".join(sentence_process.cut_sentence_by_character(setence)) for setence in lines]

            quesions_vectors=[fasttext.get_sentence_vector(quesion_string_cut) for quesion_string_cut in quesions_string_cut]

            # fasttest
            cp=ci.MultiClusterIndex(quesions_vectors,quesions_string_cut,num_indexes=3)

            pickle.dump(cp,open(config.recall_pysparnn_cp_model_path,"wb"))

            return cp


    def search_memory(self,question_sentence,fasttext_model):
        # 问题>向量>cp>记忆

        question_word_list=sentence_process.cut_sentence_by_character(question_sentence)

        # 用fasttext把用户问题转成向量
        question_sequence=fasttext_model.get_sentence_vector(" ".join(question_word_list))

        cp=self.get_pysprnn_model(fasttext_model)

        # k返回几个候选,num_indexes一个数最多同属几个聚类,
        candidates=cp.search(question_sequence,k=4,num_indexes=3)

        return candidates[0]













