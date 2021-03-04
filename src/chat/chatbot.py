# -*- coding: utf-8 -*-
# @Time    : 2020/12/31 5:40 PM
# @Author  : Kevin
from src.chat.seq2seq import ChatSeq2Seq
import torch
from src import config,lib
from src.utils import sentence_process
import numpy as np

class Chatbot:


    def __init__(self):
        # 可以先不训练,直接写eval逻辑
        self.seq2seq = ChatSeq2Seq()

        self.seq2seq.load_state_dict(torch.load(config.chat_seq_2_seq_model_path, map_location=lib.device))


    def predict(self,sentence):

        self.seq2seq.eval()

        word_list = sentence_process.cut_sentence_by_character(sentence)

        word_sequence = lib.chat_ask_word_sequence_model.transform(word_list,
                                                                   sentence_max=config.chat_ask_sentence_max_len,
                                                                   add_end=False)

        if len(word_sequence) == 0:
            return "请说人话"
        # [batch size,seq len]
        encoder_input = torch.LongTensor(word_sequence)

        encoder_input_lens = torch.LongTensor([len(word_list) if len(
            word_list) < config.chat_ask_sentence_max_len else config.chat_ask_sentence_max_len])

        encoder_input = encoder_input.unsqueeze(0)

        # 之前要对word_list裁剪,如果超过max,就用max
        # indices=seq2seq.evaluate(encoder_input,encoder_input_lens)
        indices = self.seq2seq.evaluate_with_beamsearch(encoder_input, encoder_input_lens)

        # [[1],[2],[3]]>[1,2,3]
        indices = np.array(indices).flatten()

        sen = lib.chat_answer_word_sequence_model.inverse_transform(indices)

        print("".join(sen))



