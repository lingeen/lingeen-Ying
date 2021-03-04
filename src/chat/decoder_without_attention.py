# -*- coding: utf-8 -*-
# @Time    : 2020/12/23 7:40 PM
# @Author  : Kevin

import torch
from torch import nn
from src.lib import chat_word_sequence_model
from src import config
import torch.nn.functional as F

class ChatDecoder(nn.Module):


    def __init__(self):
        super(ChatDecoder,self).__init__()
        self.embedding=nn.Embedding(num_embeddings=len(chat_word_sequence_model)
                                    ,embedding_dim=config.chat_decoder_embedding_dim
                                    ,padding_idx=chat_word_sequence_model.PAD)
        self.gru=nn.GRU(input_size=config.chat_decoder_embedding_dim
                        ,hidden_size=config.chat_decoder_gru_hidden_size
                        ,batch_first=True
                        ,bidirectional=config.chat_gru_bidirectional
                        ,num_layers=config.chat_gru_num_layers
                        )

        self.linear=nn.Linear(config.chat_decoder_gru_hidden_size*1,len(chat_word_sequence_model))

    def forward(self,encoder_outputs,encoder_hidden_state,decoder_label):
        # encoder_outputs[batch size,seq len,hidden size*bi]
        # encoder_hidden_state[bi*layers,batch size,hidden size]
        batch_size=decoder_label.size(0)
        seq_len=decoder_label.size(1)
        # 1.初始化一个START,连同encoder_hidden_state输入一个GRU
        # start [batch size,1]
        decoder_step_input=torch.LongTensor(torch.ones(batch_size,1,dtype=torch.long)*chat_word_sequence_model.START)

        # one_step_output[]
        # decoder_hidden_state[]
        # 循环seq len长度的input,得到每一步的output,汇总[seq]
        decoder_outputs=torch.zeros(batch_size,seq_len,len(chat_word_sequence_model))
        for i in range(seq_len):
            # one_step_output[batch size,dict size]
            one_step_output_softmax,decoder_hidden_state=self.forward_one_step(decoder_step_input,encoder_hidden_state,encoder_outputs)

            # print(one_step_output.size())
            # print(decoder_hidden_state.size())

            value,index=torch.topk(one_step_output_softmax,k=1)

            # forward step需要[batch size,1],decoder_step_input是预测出来词索引值
            decoder_step_input=index

            encoder_hidden_state=decoder_hidden_state

            #在seq len上收集所有output
            decoder_outputs[:,i,:]=one_step_output_softmax

        # decoder_outputs[batch size,seq len,hidden size*bi]
        return decoder_outputs,decoder_hidden_state



    def forward_one_step(self,batch_inputs_index,encoder_hidden_state,encoder_outputs):
        # 1.输入GRU
        # batch_inputs_index[batch size,seq len]>[10,1]
        input_embedding = self.embedding(batch_inputs_index)

        # encoder_hidden_state[1,10,300]
        # one_step_output[batch size,seq len,bi*hidden size]>[10,1,1*300]
        # decoder_hidden_state[bi*layers,batch size,hidden size]>[1*1,10,300]
        one_step_output, decoder_hidden_state = self.gru(input=input_embedding, hx=encoder_hidden_state)


        # one_step_output[batch size,1,bi*hidden size]>[batch size,bi*hidden size]
        one_step_output=one_step_output.squeeze(1)
        # [batch size, bi * hidden size]>[batch size,dict size]
        one_step_output=self.linear(one_step_output)
        one_step_output_softmax=F.log_softmax(one_step_output,dim=-1)

        # 最终one_step_output需要是[batch size,seq len]
        # decoder_hidden_state[bi*layers,batch size,hidden size]
        return one_step_output_softmax, decoder_hidden_state

