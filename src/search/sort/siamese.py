# -*- coding: utf-8 -*-
# @Time    : 2020/12/23 2:35 PM
# @Author  : Kevin

import torch
from torch import nn
import torch.nn.functional as F
import config
from search.sort.word_to_sequence import Word2Sequence
import lib

class Siamese(nn.Module):

    def __init__(self):
        super(Siamese,self).__init__()
        '''
        准备孪生网络需要的模型层
        '''
        self.embedding=nn.Embedding(len(lib.word_sequence_model)
                                    ,config.sort_input_embedding_dim
                                    ,padding_idx=lib.ws.PAD)

        self.gru1=nn.GRU(input_size=config.sort_input_embedding_dim
                         ,hidden_size=config.sort_gru1_hidden_size
                         ,bidirectional=True
                         ,num_layers=config.sort_gru1_layers
                         ,batch_first=True)

        self.gru2=nn.GRU(input_size=config.sort_gru1_hidden_size*4
                         ,hidden_size=config.sort_gru2_hidden_size
                         ,bidirectional=False
                         ,num_layers=config.sort_gru2_layers
                         ,batch_first=True)

        self.dnn=nn.Sequential(
            nn.BatchNorm1d(config.sort_gru2_hidden_size*4),

            nn.Linear(config.sort_gru2_hidden_size*4,config.dnn_linear1_out_dim),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(config.dnn_linear1_out_dim),
            nn.Dropout(config.sort_drop_out),

            nn.Linear(config.dnn_linear1_out_dim,config.dnn_label_size),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(config.dnn_label_size),
            nn.Dropout(config.sort_drop_out),
        )


    def forward(self,input1,input2):
        mask1=input1.eq(Word2Sequence().PAD)
        mask2=input2.eq(Word2Sequence().PAD)

        # input[batch size,seq len]
        # input_embedding[batch size,seq len,embedding dim]
        input1_embedding=self.embedding(input1)
        input2_embedding=self.embedding(input2)

        # input_gru1_outputs[batch size,seq len,hidden size*bi]>[batch size,seq len,hidden size*2]
        # input_gru1_hidden_state[bi*layers,batch size,hidden size]>[2*layers,batch size,hidden size]
        input1_gru1_outputs,input1_gru1_hidden_state=self.gru1(input1_embedding)
        input2_gru1_outputs,input2_gru1_hidden_state=self.gru1(input2_embedding)

        # 注意力
        # input1_outputs_weight[batch size,seq len2,seq len1]
        # input2_outputs_weight[batch size,seq len1,seq len2]
        input1_outputs_weight,input2_outputs_weight=self.apply_attention(input1_gru1_outputs,input2_gru1_outputs,mask1,mask2)

        # input2_gru1_outputs[batch size,seq len2,hidden size*2]
        # input2_outputs_weight[batch size,seq len1,seq len2]
        # input1_implement_vector[batch size,seq len1,hidden size*2]
        input1_implement_vector=torch.bmm(input2_outputs_weight,input2_gru1_outputs)
        # input1_outputs_weight[batch size,seq len2,seq len1]
        # input1_gru1_outputs[batch size,seq len1,hidden size*2]
        # input2_implement_vector[batch size,seq len2,hidden size*2]
        input2_implement_vector=torch.bmm(input1_outputs_weight,input1_gru1_outputs)



        # decoder的output拼接自己的weight
        # input2_gru1_outputs[batch size,seq len2,hidden size*2]
        # input2_implement_vector[batch size,seq len2,hidden size*2]
        # input2_gru1_outputs[batch size,seq len,hidden size*4]
        input2_gru1_outputs=torch.cat([input2_gru1_outputs,input2_implement_vector],dim=-1)
        input1_gru1_outputs=torch.cat([input1_gru1_outputs,input1_implement_vector],dim=-1)

        # gur2
        # [batch size, seq len, hidden size * 8]
        # [batch size, seq len, config.sort_gru2_hidden_size]
        input2_gru2_outputs,input2_gru2_hidden_state=self.gru2(input2_gru1_outputs)
        input1_gru2_outputs,input1_gru2_hidden_state=self.gru2(input1_gru1_outputs)

        #池化[batch size,config.sort_gru2_hidden_size*2]
        input2_gru2_outputs=self.apply_pooling(input2_gru2_outputs)
        input1_gru2_outputs=self.apply_pooling(input1_gru2_outputs)

        # 二者拼接
        # assembel_outputs[batch size,config.sort_gru2_hidden_size*4]
        assembel_outputs=torch.cat([input1_gru2_outputs,input2_gru2_outputs],dim=-1)

        final_outputs=self.dnn(assembel_outputs)

        log_softmax_weight=F.log_softmax(final_outputs,dim=-1)

        return log_softmax_weight


    def apply_attention(self,outputs1,outputs2,mask1,mask2):
        # mask是权重
        mask1=mask1.float().masked_fill(mask1,float("-inf"))
        mask2=mask2.float().masked_fill(mask2,float("-inf"))
        '''
        outputs*hidden state>softmax>weight*outputs
        :param hidden_state:
        :param outputs:
        :return:
        '''
        # outputs1[batch size,seq len1,hidden size*bi]
        # outputs2[batch size,seq len2,hidden size*bi]
        # 左为encoder,右为decoder,2*1 +mask2  softmax
        # [batch size,seq len1,seq len2]+mask2[batch size,1,seq len2]
        # output2_score=torch.bmm(outputs1,outputs2.transpose(1,2)+mask2.unsqueeze(1))
        output2_score=torch.bmm(outputs1,outputs2.transpose(1,2))
        # 右为encoder,左为decoder
        # outputs1[batch size,hidden size*bi,seq len1]
        # [batch size,seq len2,seq len1]+mask1[batch size,1,seq len1]
        output1_score=torch.bmm(outputs2,outputs1.transpose(1,2))
        # output1_score=torch.bmm(outputs2,outputs1.transpose(1,2)+mask1.unsqueeze(1))

        # input1_outputs_weight[batch size,seq len2,seq len1]
        input1_outputs_weight=F.softmax(output1_score,dim=-1)
        # input2_outputs_weight[batch size,seq len1,seq len2]
        input2_outputs_weight=F.softmax(output2_score,dim=-1)

        return input1_outputs_weight,input2_outputs_weight




    def apply_pooling(self,outputs):
        '''
        在seq维度进行pooling,词维度,一个句子pooling成1
        :param outputs:
        :return:
        '''
        # outputs[batch size, seq len, config.sort_gru2_hidden_size]
        # outputs[batch size, config.sort_gru2_hidden_size,seq len]
        outputs=outputs.transpose(1,2)
        # [batch size,hidden size,1]>[batch size,hidden size]
        # 窗口大小seq len
        outputs_max_pooling=F.max_pool1d(outputs,outputs.size(2))
        outputs_max_pooling=outputs_max_pooling.squeeze(-1)
        outputs_avg_pooling=F.avg_pool1d(outputs,kernel_size=outputs.size(2))
        outputs_avg_pooling=outputs_avg_pooling.squeeze(-1)
        # 拼接[batch size,config.sort_gru2_hidden_size*2]
        return torch.cat([outputs_max_pooling,outputs_avg_pooling],dim=-1)

