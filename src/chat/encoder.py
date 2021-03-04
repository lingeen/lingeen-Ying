# -*- coding: utf-8 -*-
# @Time    : 2020/12/23 7:40 PM
# @Author  : Kevin
from torch import nn
from src.lib import chat_ask_word_sequence_model
from src import config



class ChatEncoder(nn.Module):

    def __init__(self):
        super(ChatEncoder,self).__init__()

        # 准备网络
        self.embedding=nn.Embedding(num_embeddings=len(chat_ask_word_sequence_model)
                                    ,embedding_dim=config.chat_encoder_embedding_dim
                                    ,padding_idx=chat_ask_word_sequence_model.PAD)

        self.gru=nn.GRU(input_size=config.chat_encoder_embedding_dim
                        ,hidden_size=config.chat_encoder_gru_hidden_size
                        ,batch_first=True
                        ,num_layers=config.chat_gru_num_layers
                        ,bidirectional=config.chat_gru_bidirectional
                        )

    def forward(self,input,asks_len):
        # input[batch size,seq len]
        # input_embedding[batch size,seq len,embedding dim]

        input_embedding=self.embedding(input)

        input_embedding=nn.utils.rnn.pack_padded_sequence(input=input_embedding,lengths=asks_len,batch_first=True)

        # outputs[batch size,seq len,hidden size*bi]
        # hidden_state[bi*layers,batch size,hidden size]
        encoder_outputs,hidden_state=self.gru(input_embedding)

        encoder_outputs, outputs_len=nn.utils.rnn.pad_packed_sequence(sequence=encoder_outputs,batch_first=True,padding_value=chat_ask_word_sequence_model.PAD)


        return encoder_outputs,hidden_state

