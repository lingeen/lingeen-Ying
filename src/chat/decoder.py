# -*- coding: utf-8 -*-
# @Time    : 2020/12/23 7:40 PM
# @Author  : Kevin

import torch
from torch import nn
from src.lib import chat_answer_word_sequence_model,device
from src import config
from src.chat.attention import ChatDecoderAttention
import numpy as np
import heapq
from src.chat.beam import Beam

class ChatDecoder(nn.Module):


    def __init__(self):
        super(ChatDecoder,self).__init__()
        self.embedding=nn.Embedding(num_embeddings=len(chat_answer_word_sequence_model)
                                    ,embedding_dim=config.chat_decoder_embedding_dim
                                    ,padding_idx=chat_answer_word_sequence_model.PAD
                                    )
        self.gru=nn.GRU(input_size=config.chat_decoder_embedding_dim
                        ,hidden_size=config.chat_decoder_gru_hidden_size
                        ,batch_first=True
                        ,bidirectional=config.chat_gru_bidirectional
                        ,num_layers=config.chat_gru_num_layers
                        )

        self.dnn=nn.Sequential(

            nn.BatchNorm1d(config.chat_encoder_gru_hidden_size+config.chat_decoder_gru_hidden_size),

            nn.Linear(config.chat_encoder_gru_hidden_size + config.chat_decoder_gru_hidden_size,config.chat_decoder_gru_hidden_size),

            nn.ELU(),

            nn.Dropout(config.chat_gru_dropout),

            nn.BatchNorm1d(config.chat_decoder_gru_hidden_size),

            nn.Linear(config.chat_decoder_gru_hidden_size,len(chat_answer_word_sequence_model)),

            nn.Softmax(dim=-1)
        )

        self.attention = ChatDecoderAttention(match_type=config.chat_decoder_attention_match_type)

    def forward(self,encoder_outputs,decoder_label,encoder_hidden_state,answer_lens):
        decoder_hidden=encoder_hidden_state
        # encoder_outputs[batch size,seq len,hidden size*bi]
        # encoder_hidden_state[bi*layers,batch size,hidden size]
        batch_size=decoder_label.size(0)
        # 1.初始化一个START,连同encoder_hidden_state输入一个GRU
        # start [batch size,1]
        decoder_step_input=torch.LongTensor(torch.ones(batch_size,1,dtype=torch.long)*chat_answer_word_sequence_model.START).to(device)

        # one_step_output[]
        # decoder_hidden_state[]
        # 循环seq len长度的input,得到每一步的output,汇总[seq]
        # +1是因为dataset里add END,结果里是句子规定长度+END
        decoder_outputs=torch.zeros(batch_size,config.chat_answer_sentence_max_len+1,len(chat_answer_word_sequence_model)).to(device)

        # 引入teacher forcing:用真实标签里的赋值给下一步的output
        teacher_help=np.random.rand()
        # 整个句子是否使用teacher forcing,如果使用,全用真实标签当下一次输入,否则用topk预测出index的当输入
        if teacher_help>config.chat_teacher_forcing_ratio:
            # +1是因为循环里有END,匹配上面的长度,循环一次得到一个output,dataset里最label做了最大长度处理,是在句子最大长度之后加END,所以得出的长度要匹配label
            for i in range(config.chat_answer_sentence_max_len + 1):
                # one_step_output[batch size,dict size]
                one_step_output_softmax, decoder_hidden_state = self.forward_one_step(decoder_step_input,decoder_hidden, encoder_outputs)

                # 在seq len上收集所有output
                decoder_outputs[:, i, :] = one_step_output_softmax

                # forward step需要[batch size,1]
                # decoder_label[batch size,1]取出[batch size,1]
                decoder_step_input = decoder_label[:,i].unsqueeze(-1)

        else:
            # +1是因为循环里有END,匹配上面的长度,循环一次得到一个output
            for i in range(config.chat_answer_sentence_max_len+1):
                # one_step_output[batch size,dict size]
                one_step_output_softmax,decoder_hidden_state=self.forward_one_step(decoder_step_input,decoder_hidden,encoder_outputs)

                # 在seq len上收集所有output
                decoder_outputs[:, i, :] = one_step_output_softmax

                value,index=torch.topk(one_step_output_softmax,k=1)

                # forward step需要[batch size,1],decoder_step_input是预测出来词索引值
                decoder_step_input = index

        # decoder_outputs[batch size,seq len,hidden size*bi]
        return decoder_outputs,decoder_hidden_state
    '''
    不能污染单步的gru,因为测试要用,预测也要用,但是预测没有输入
    单步骤只和decoder input,和encoder hidden有关
    '''
    def forward_one_step(self,batch_inputs_index,decoder_hidden,encoder_outputs):

        # 1.输入GRU
        # batch_inputs_index[batch size,seq len]>[10,1]
        input_embedding = self.embedding(batch_inputs_index)

        # encoder_hidden_state[1,10,300]
        # one_step_output[batch size,seq len,bi*hidden size]>[10,1,1*300]
        # decoder_hidden_state[bi*layers,batch size,hidden size]>[1*1,10,300]
        one_step_output, decoder_hidden_state = self.gru(input=input_embedding, hx=decoder_hidden)

        # one_step_output[batch size,1,bi*hidden size]>[batch size,bi*hidden size]
        one_step_output = one_step_output.squeeze(1)

        # 出gru就注意了!
        # 加入attention:[batch size,seq len]结果是要拼接到当前one_step_output_softmax上的
        # encoder outputs[batch size,seq len,bi*hidden size]
        # decoder hiddens[bi*layers,batch size,hidden size]
        encoder_outputs_weight=self.apply_attention(encoder_outputs,decoder_hidden_state)

        # one_step_attention_vector[batch size,1,hidden size* bi]
        one_step_attention_vector = torch.bmm(encoder_outputs_weight, encoder_outputs)
        one_step_attention_vector = one_step_attention_vector.squeeze(1)

        # one_step_attention_vector需要是[batch size,bi*hidden size]
        # 拼接后是[batch size,hidden size*2]

        one_final_step_output=torch.cat([one_step_output,one_step_attention_vector],dim=-1)

        one_step_output_softmax=self.dnn(one_final_step_output)

        # 最终one_step_output需要是[batch size,seq len]
        # decoder_hidden_state[bi*layers,batch size,hidden size]
        return one_step_output_softmax, decoder_hidden_state


    def apply_attention(self,encoder_outputs,decoder_hidden_state):

        # encoder_outputs_weight[batch size,seq len1]
        encoder_outputs_weight=self.attention(encoder_outputs,decoder_hidden_state)

        # 与encoder_outputs[batch size,seq len1,hidden size*bi]相乘
        encoder_outputs_weight=encoder_outputs_weight.unsqueeze(1)

        return encoder_outputs_weight

    def evaluate(self,encoder_outputs,encoder_hidden_state):
        '''
        每一步计算,搜集index
        :param encoder_hidden_state:
        :return:
        '''

        decoder_hidden=encoder_hidden_state

        batch_size=decoder_hidden.size(1)

        decoder_input=torch.LongTensor(torch.ones(batch_size,1,dtype=torch.long)*chat_answer_word_sequence_model.START)

        # 不保存softmax,直接保存index
        # [chat_answer_sentence_max_len,batch size]
        indices=[]

        #当不是END,且未超过解码句子长度,则循环
        # while decoder_input!=chat_answer_word_sequence_model.END and decoder_outputs.size() <config.chat_answer_sentence_max_len:
        # +5尽量长一些,
        for i in range(config.chat_answer_sentence_max_len+5):

            one_step_output_softmax, decoder_hidden_state=self.forward_one_step(decoder_input,decoder_hidden,encoder_outputs)

            # one_step_output_softmax[batch size,dict size]被topk挑选出1个
            # index[batch size,1]
            value,index=torch.topk(one_step_output_softmax,k=1,dim=-1)

            decoder_input=index

            # index[batch size,1]>[batch size]

            indices.append(index.squeeze(-1).cpu().detach().numpy())

            # [chat_answer_sentence_max_len,batch size]
        return indices

    def evaluate_with_beamsearch(self,encoder_outputs,encoder_hidden_state):
        '''
        每一步计算,搜集index
        :param encoder_hidden_state:
        :return:
        '''
        batch_size=encoder_hidden_state.size(1)
        # 1.准备START[batch size,1]
        start_input=torch.LongTensor([[chat_answer_word_sequence_model.START]*batch_size]).to(device)

        # 2.准备input beam
        pre_input_beam=Beam()

        # 3.START进input beam,开始概率为1,joint_probability,if_end,outputs,step_output,hidden
        # 只存decoder自己的东西
        # todo 只要存的是列表,就会无缘无故冒出来4
        pre_input_beam.add(1,False,[start_input],start_input,encoder_hidden_state)

        # 4.开始输入decoder
        while True:
            curr_output_beam=Beam()
        # 5.遍历input beam,取元素,判断是否输入,输入forward step,得到step output softmax[batch size,dict size],hidden[1,batch size,hidden size]
            for pre_joint_probability,pre_if_end,pre_outputs,pre_step_output,pre_hidden in pre_input_beam:
                if pre_if_end is False:
                    # step_output_softmax[batch size,dict size],hidden_state[1,batch size,decoder hidden size]
                    curr_step_output_softmax,curr_hidden_state=self.forward_one_step(pre_step_output,pre_hidden,encoder_outputs)

                    # 6.step output top k个 >(概率..),(index...)
                    # [batch size,dict size]top成[batch size,3]
                    values,indexs=torch.topk(curr_step_output_softmax,k=config.chat_decoder_beam_width,dim=-1)
                    # [batch size,3]>[1,1,1...]
                    # 7.对齐(概率,index),遍历用概率,index处理成output beam(总概率,if end,[句子index...],input[batch size,1],hidden state)
                    # probability_indexs_pair=zip(values.squeeze(0), indexs.squeeze(0))
                    probability_indexs_pair=zip(values[0], indexs[0])
                    # 这俩都是scalar
                    for single_probability,singlle_index in probability_indexs_pair:

                        curr_joint_probability=pre_joint_probability*single_probability

                        if_end=False

                        if singlle_index.item()==chat_answer_word_sequence_model.END:
                            if_end=True
                        # outputs需要的是和START[batch size,1]一样的形状

                        # step_output[batch size,1]
                        curr_step_output=torch.LongTensor([[singlle_index]])
                        # append不改变原来的
                        curr_outputs = pre_outputs + [curr_step_output]
                        # 将当前step上获得的所有信息存如output beam中
                        curr_output_beam.add(curr_joint_probability,if_end,curr_outputs,curr_step_output,curr_hidden_state)
                else:
                    # 输入的时候如果遇到已经是END的结果,直接存下来
                    curr_output_beam.add(pre_joint_probability,pre_if_end,pre_outputs,pre_step_output,pre_hidden)

            # 循环取完input beam,并得到所有输出的结果存进output beam后,取出output beam里概率最大的,判断是不是END,如果是,组装成完整的outputs返回
            # 否则当前output beam变成input beam,继续while true输入进入下一个forward step

            curr_best_joint_probability, curr_best_if_end, curr_best_outputs, curr_best_step_output, curr_best_hidden=max(curr_output_beam)

            # -1是因为里面有个start
            if curr_best_if_end and len(curr_best_outputs) - 1 >= config.chat_answer_sentence_max_len:

                # outputs[seq len,batch size,1]>[seq len,batch size]
                return self.handle_outputs(curr_best_outputs)

            else:
                pre_input_beam=curr_output_beam




    def handle_outputs(self,curr_best_outputs):
        '''
        斩头去尾
        :param curr_best_outputs:
        :return:
        '''
        # curr_best_outputs > [tensor([[2]]), [tensor(5)], [tensor(5)], [tensor(17)], [tensor(3)]]   > [batch size,1]
        if curr_best_outputs[0].item()==chat_answer_word_sequence_model.START:
            curr_best_outputs=curr_best_outputs[1:]
        if curr_best_outputs[-1].item()==chat_answer_word_sequence_model.END:
            curr_best_outputs = curr_best_outputs[:-1]
        indices=[best_output_index.item() for best_output_index in  curr_best_outputs]

        return indices

