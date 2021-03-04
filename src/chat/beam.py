# -*- coding: utf-8 -*-
# @Time    : 2020/12/30 3:25 PM
# @Author  : Kevin
import heapq
from src import config

class Beam:

    def __init__(self):

        self.context=[]

        self.beam_width=config.chat_decoder_beam_width

    def add(self,joint_probability,if_end,outputs,step_output,hidden):

        heapq.heappush(self.context,[joint_probability,if_end,outputs,step_output,hidden])

        # 更新
        if len(self.context)>self.beam_width:
            # heapq每次取最小,想保留3个最大的,就取最小的pop掉,去除最小的,就是最大的
            heapq.heappop(self.context)


    def __iter__(self):
        return iter(self.context)


