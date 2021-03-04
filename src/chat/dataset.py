# -*- coding: utf-8 -*-
# @Time    : 2020/12/23 7:42 PM
# @Author  : Kevin
from src import config
from torch.utils.data import Dataset,DataLoader
from src import lib
import torch

class ChatDataset(Dataset):


    def __init__(self):

        super(ChatDataset,self).__init__()

        self.ask_lines=open(config.chat_ask_file_path,"r").readlines()
        self.answer_lines=open(config.chat_answer_file_path,"r").readlines()

        assert len(self.ask_lines)==len(self.answer_lines),"ask_lines和answer_lines长度不一致"


    def __getitem__(self, index):
        ask_word_list=self.ask_lines[index].strip().split()
        answer_word_list=self.answer_lines[index].strip().split()
        # 对长度进行裁剪
        ask_len=len(ask_word_list) if len(ask_word_list)<config.chat_ask_sentence_max_len else config.chat_ask_sentence_max_len

        answer_len=len(answer_word_list) if len(answer_word_list)<config.chat_answer_sentence_max_len+1 else config.chat_answer_sentence_max_len+1

        return ask_word_list,answer_word_list,ask_len,answer_len

    def __len__(self):

        return len(self.ask_lines)


def collate_fn(batch):

    batch=sorted(batch,key=lambda x:x[-2],reverse=True)

    asks,answers,ask_lens,answer_lens=zip(*batch)
    asks=torch.LongTensor([lib.chat_ask_word_sequence_model.transform(ask,sentence_max=config.chat_ask_sentence_max_len) for ask in asks])

    answers=torch.LongTensor([lib.chat_answer_word_sequence_model.transform(answer,sentence_max=config.chat_answer_sentence_max_len,add_end=True) for answer in answers])
    # label有的有尾巴,有的没尾巴?按理说应该max 包含END
    ask_lens=torch.LongTensor(ask_lens)

    answer_lens=torch.LongTensor(answer_lens)

    return asks,answers,ask_lens,answer_lens




def get_dataloader(shuffle=True):
    return DataLoader(dataset=ChatDataset(),shuffle=shuffle,batch_size=config.chat_data_batch_size,collate_fn=collate_fn)




