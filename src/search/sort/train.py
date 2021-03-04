# -*- coding: utf-8 -*-
# @Time    : 2020/12/23 5:05 PM
# @Author  : Kevin
from search.sort.dataset import get_data_loader
from search.sort.siamese import Siamese
import torch.nn.functional as F
from torch.optim import Adam
import torch




def train():
    # 1.准备数据
    datalaoder=get_data_loader()


    # 2.建立模型
    siamese_network=Siamese()

    optimizer=Adam(siamese_network.parameters(),lr=0.001)

    siamese_network.train()
    # 3.训练
    for index, (input1s, input2s, labels, input1_lens, input2_lens) in enumerate(datalaoder):

        log_softmax_result=siamese_network(input1s,input2s)

        loss=F.nll_loss(log_softmax_result,labels).mean()

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        acc=log_softmax_result.max(dim=-1)[-1].eq(labels)
        accuracy=acc.float().mean()

        print(f"第{index}次,损失是{loss},正确率:{accuracy}")


    # 4.测试


    # 5.保存模型









if __name__ == '__main__':


    train()
    #
    # a=torch.LongTensor([[0.6052, 0.3948],
    #     [0.8467, 0.1533],
    #     [0.5000, 0.5000],
    #     [0.5000, 0.5000],
    #     [0.5071, 0.4929],
    #     [0.7459, 0.2541],
    #     [0.9879, 0.0121],
    #     [0.0148, 0.9852],
    #     [0.7917, 0.2083],
    #     [0.0158, 0.9842]])
    #
    # b=torch.LongTensor([1, 1, 1, 1, 1, 1, 0, 1, 1, 1])
    # print(a)
    #
    # print(b)
    # los=F.cross_entropy(a,b)
    # # loss=F.nll_loss(a,b)
    # # print(loss.mean())
    # print(los.mean())