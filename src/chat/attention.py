# -*- coding: utf-8 -*-
# @Time    : 2020/12/24 4:41 PM
# @Author  : Kevin
import torch
from torch import nn
from src import config

class ChatDecoderAttention(nn.Module):

    def __init__(self,match_type="concat"):

        super(ChatDecoderAttention,self).__init__()

        assert match_type in ["dot","general","concat"],"match type error"

        self.match_type=match_type

        if match_type=='general':
            # 把encoder的outputs乘以Wa变形
            self.Wa=nn.Linear(config.chat_encoder_gru_hidden_size,config.chat_decoder_gru_hidden_size,bias=False)
        if match_type=='concat':
            self.Wa=nn.Linear(config.chat_encoder_gru_hidden_size+config.chat_decoder_gru_hidden_size,config.chat_decoder_gru_hidden_size,bias=False)
            self.tanh=nn.Tanh()
            self.Va=nn.Linear(config.chat_decoder_gru_hidden_size,1)

        self.softmax=nn.Softmax(dim=-1)

    def forward(self,encoder_outputs,decoder_hidden_state):
        '''
        :param decoder_hidden_state:[bi*num_layers,batch_size,decoder_hidden_size]
        :param encoder_outputs:[batch_size,seq_len,encoder_hidden_size]
        :return:
        '''
        batch_size = encoder_outputs.size(0)
        seq_len = encoder_outputs.size(1)

        # 1.实现dot操作,decoder_hidden_state*encoder_outputs
        if self.match_type=='dot':
            # decoder_hidden_state[layers,batch_size,hidden_size]变成[batch_size,hidden_size,1]
            decoder_hidden_state=decoder_hidden_state[-1,:,:].permute(1,2,0)
            # encoder_outputs[batch size,seq len,hidden size]**decoder_hidden_state[batch size,hidden size,1]>[batch size,seq len,1]
            attention_score=torch.bmm(encoder_outputs,decoder_hidden_state)
            # 把[batch size,seq len,1]的1维度消除
            attention_score=attention_score.squeeze(-1)
            # 做softmax得到比重attention_weight[batch size,seq len]
            attention_weight=self.softmax(attention_score)
        elif self.match_type=='general':
            # encoder outputs变形>和decoder hidden state矩阵乘法
            # [batch size ,seq len,encoder hidden size]变成[batch size,seq len,decoder hidden size]
            # [batch*seq,encoder hidden]>[batch*seq,decoder hidden]
            encoder_outputs=self.Wa(encoder_outputs.view(batch_size*seq_len,-1))
            encoder_outputs=encoder_outputs.view(batch_size,seq_len,-1)
            # decoder_hidden_state[layers,batch size,decoder hidden size]>[batch size,decoder hidden size,1]
            decoder_hidden_state=decoder_hidden_state[-1,:,:].permute(1,2,0)
            # 三维矩阵乘法
            attention_score=torch.bmm(encoder_outputs,decoder_hidden_state)

            attention_score=attention_score.squeeze(dim=-1)

            attention_weight=self.softmax(attention_score)

        elif self.match_type=='concat':
            # encoder_outputs[batch size,seq len,encoder hidden size*bi]
            # weight=V*tanh(W[outputs,hidden])
            # decoder_hidden_state[layers*bi,batch size,decoder hidden size]
            # 取hidden部分layers*bi上最后一个>[1,batch size,decoder hidden size]>squeeze(0)
            # [batch size,decoder hidden size]

            # decoder_hidden_state[-1,:,:]取到的是[batch size,decoder hidden size]
            decoder_hidden_state=decoder_hidden_state[-1,:,:]
            # 1.拼接[batch size,decoder hidden size],[batch size,seq len,encoder hidden size*bi]
            # decoder_hidden_state[batch size,decoder hidden size]扩充成outputs[batch size,seq len,encoder hidden size*bi]>repeat参数是每个位置上重复次数
            # decoder_hidden_state[batch size,seq len,decoder hidden size]
            # 1,1,300>1,3,300
            decoder_hidden_state=decoder_hidden_state.unsqueeze(1).repeat(1,encoder_outputs.size(1),1)
            # cated[batch size,seq len,encoder hidden size+decoder hidden size]

            cated=torch.cat([encoder_outputs,decoder_hidden_state],dim=-1)

            # weight = V * tanh(W[outputs, hidden])
            # attention_score[batch size,seq len,decoder hidden state]
            attention_score=self.tanh(self.Wa(cated))
            # attention_score[batch size,seq len,1]
            attention_score=self.Va(attention_score)
            # 2.score转成weight[batch size,seq len]
            attention_weight=self.softmax(attention_score.squeeze(-1))

        # [batch size,seq len]
        return attention_weight

