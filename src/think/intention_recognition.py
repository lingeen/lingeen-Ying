# -*- coding: utf-8 -*-
# @Time    : 2020/12/22 4:49 PM
# @Author  : Kevin
'''
接收用户问题,判断是否是问答模型
'''
import fasttext
import config
from utils import sentence_process
import os


def train():
    # 1.直接给文件路径 2.文本分词 __label__ 3.文件里的label前缀默认是__label__
    model=fasttext.train_supervised(config.think_train_data_path,epoch=20,lr=0.001,wordNgrams=2,label="__label__")
    print(config.think_intention_recognition_model_path)
    model.save_model(config.think_intention_recognition_model_path)
    return model

class IntentionRecognition():


    def __init__(self):
        # 加载模型,如果加载不到,则进入训练模式,训练出一个模型,并保存
        if os.path.exists(config.think_intention_recognition_model_path):
            self.model = fasttext.load_model(config.think_intention_recognition_model_path)
        else:
            self.model = train()


    def if_ask_question(self,sentence):
        word_list=sentence_process.cut_sentence_by_character(sentence)
        label,scores=self.model.predict(" ".join(word_list))
        return label[0],scores[0]

