# -*- coding: utf-8 -*-
# @Time    : 2020/12/23 2:28 PM
# @Author  : Kevin
from torch.utils.data import Dataset,DataLoader
import config
import lib
import torch
from utils.sentence_process import cut_sentence_by_character

'''
dataset只为模型服务,不为prepare服务
'''
class SearchSortDadaset(Dataset):

    def __init__(self):
        super(SearchSortDadaset,self).__init__()

        # 读取input1和input2
        self.input1=open(config.sort_input1_file_path,"r").readlines()
        self.input2=open(config.sort_input2_file_path,"r").readlines()
        self.label=open(config.sort_label_file_path,"r").readlines()


    def __getitem__(self, item):
        # 带上句子长度
        return self.input1[item],self.input2[item],self.label[item],len(self.input1[item]),len(self.input2[item])


    def __len__(self):
        return len(self.label)

def collate_fn(batch):
    '''
    batch 就是__getitem__返回的内容
    :param batch:
    :return:
    '''
    # 按照input1的句子长度降序
    batch=sorted(batch,key=lambda x:x[-2],reverse=True)

    input1s,input2s,labels,input1_lens,input2_lens=zip(*batch)

    # input1,input2转成index向量,其余转成LongTensor

    input1s=torch.LongTensor([lib.word_sequence_model.transform(cut_sentence_by_character(input1)) for input1 in input1s])
    input2s=torch.LongTensor([lib.word_sequence_model.transform(cut_sentence_by_character(input2)) for input2 in input2s])

    labels=torch.LongTensor([float(label.strip())  for label in labels])

    input1_lens=torch.LongTensor(input1_lens)
    input2_lens=torch.LongTensor(input2_lens)

    return input1s,input2s,labels,input1_lens,input2_lens


def get_data_loader():
    return DataLoader(dataset=SearchSortDadaset(),batch_size=config.sort_data_batch_size,shuffle=True,collate_fn=collate_fn)


if __name__ == '__main__':
    dataloader=get_data_loader()

    for index,(input1s,input2s,labels,input1_lens,input2_lens) in enumerate(dataloader):
        print(input1s)
        print(input2s)
        print(labels)
        print(input1_lens)
        print(input2_lens)
        print("*"*30)
