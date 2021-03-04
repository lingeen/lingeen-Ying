# -*- coding: utf-8 -*-
# @Time    : 2020/12/24 12:44 PM
# @Author  : Kevin
from torch import nn
from src.chat.encoder import ChatEncoder
from src.chat.decoder import ChatDecoder

class ChatSeq2Seq(nn.Module):

    def __init__(self):
        super(ChatSeq2Seq, self).__init__()
        self.encoder=ChatEncoder()
        self.decoder=ChatDecoder()

    def forward(self,encoder_input,decoder_label,ask_lens,answers_length):

        # outputs[batch size,seq len,hidden size*bi]>[10,10,300*2]
        # hidden_state[bi*layers,batch size,hidden size]>[2*2,10,300]
        encoder_outputs,encoder_hidden_state=self.encoder(encoder_input,ask_lens)

        # hidden_state作为decoder的init hidden state
        # outputs用于decoder的attention
        decoder_outputs,decoder_hidden_state=self.decoder(encoder_outputs,decoder_label,encoder_hidden_state,answers_length)

        # [batch size,seq len,hidden size*bi]>[10,10,300*1]
        return decoder_outputs,decoder_hidden_state


    def evaluate(self,encoder_input,ask_lens):
        encoder_outputs, encoder_hidden_state = self.encoder(encoder_input, ask_lens)
        indices=self.decoder.evaluate(encoder_outputs,encoder_hidden_state)
        return indices

    def evaluate_with_beamsearch(self,encoder_input,ask_lens):

        encoder_outputs, encoder_hidden_state = self.encoder(encoder_input, ask_lens)
        indices=self.decoder.evaluate_with_beamsearch(encoder_outputs,encoder_hidden_state)
        return indices













