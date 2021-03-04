# -*- coding: utf-8 -*-
# @Time    : 2020/12/22 1:40 PM
# @Author  : Kevin
import torch
import config
from src.think.intention_recognition import IntentionRecognition
from src.search.recall.recall import Memory
from src.search.sort.siamese import Siamese



#TODO 
def intention_recognition(sentence):
    model=IntentionRecognition()
    label,score=model.if_ask_question(sentence)
    return label


def search_recall(sentence):
    memory=Memory()
    questions=memory.search_memory(sentence,memory.get_fasttext_model())

    return questions

def search_sort(sentence,recall_requestions):
    siamese=Siamese()
    # 没有相似问题数据集,未保存模型
    siamese.load_state_dict(torch.load(config.sortpath))


    pass



def chat(sentence):

    pass




if __name__ == '__main__':
    while True:
        question=input()
        model_type=intention_recognition(question)
        answer="对不起,您的问题太难了"
        # 1.先做意图识别
        if model_type=="__label__qa":
            recall_requestions=search_recall(question)
            answer=search_sort(question,recall_requestions)



        elif model_type=="__label__chat":
            answer=chat(question)


        print(answer)





        # 2.