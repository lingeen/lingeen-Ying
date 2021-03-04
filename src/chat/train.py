# -*- coding: utf-8 -*-
# @Time    : 2020/12/24 3:48 PM
# @Author  : Kevin
from src.chat import dataset
from src.chat.seq2seq import ChatSeq2Seq
from torch.optim import Adam
import torch.nn.functional as F
import torch
from src import config
from tqdm import tqdm
from src.lib import device,chat_answer_word_sequence_model
from torch import nn

def train(epoch):
    # 1.准备数据
    dataloader = dataset.get_dataloader()
    # 2.建立模型
    seq2seq = ChatSeq2Seq().to(device)

    optimizer=Adam(seq2seq.parameters(),lr=0.001)

    former_acc=0.

    seq2seq.train()

    bar=tqdm(enumerate(dataloader),ascii=True,desc="training...")
    # 3.训练
    for index, (asks, answers, ask_lens, answer_lens) in bar:

        asks=asks.to(device)
        answers=answers.to(device)

        optimizer.zero_grad()

        decoder_outputs_softmax, decoder_hidden_state = seq2seq(asks, answers,ask_lens,answer_lens)

        # [batch size,seq len]>[batch size*seq len]
        answers=answers.view(-1)
        # decoder_outputs[batch size,seq len,dict size]>[batch size*seq len,dict size]
        # -1就是保留
        decoder_outputs_softmax=decoder_outputs_softmax.view(decoder_outputs_softmax.size(0)*decoder_outputs_softmax.size(1),-1)
        # 保留hidden size维度
        # loss ouputs二维,label一维 21  损失21金维他
        loss=F.cross_entropy(decoder_outputs_softmax,answers,ignore_index=chat_answer_word_sequence_model.PAD)

        loss.backward()

        # 梯度裁剪,裁剪掉过大梯度,避免梯度爆炸
        # 下划线是直接修改
        nn.utils.clip_grad_norm_(seq2seq.parameters(),config.caht_train_grad_clip_max)

        optimizer.step()

        # 计算正确率
        acc = decoder_outputs_softmax.max(dim=-1)[-1]
        acc = acc.eq(answers).float().mean()

        bar.set_description(f"eporch:{epoch}\tindex:{index}\tloss:{loss.item()}\t正确率:{acc}")


        if acc>former_acc:
            torch.save(seq2seq.state_dict(), config.chat_seq_2_seq_model_path)
            torch.save(optimizer.state_dict(), config.chat_seq_optimizer_model_path)
            if epoch%10==0:
                torch.save(seq2seq.state_dict(), config.chat_seq_2_seq_model_path+str(epoch))
                torch.save(optimizer.state_dict(), config.chat_seq_optimizer_model_path+str(epoch))

            former_acc=acc

    return former_acc




if __name__ == '__main__':


    epoch=30
    acc=[]
    for i in range(epoch):
        former_acc=train(i)
    print(acc)
    # eval()