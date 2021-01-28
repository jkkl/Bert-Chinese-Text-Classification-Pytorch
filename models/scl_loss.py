# !/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Author: yuanzhuo@xiaoice.com
Created on: 2020/12/23 13:35
'''

import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.autograd import Variable


def get_class_num(input_y):
    '''
    input_y: [batch_size]
    return: [batch_size] , 值为在该batch中，该类样本的个数
    '''
    batch_size = len(input_y)
    class_num = torch.zeros([batch_size, 1])
    class_num_dict = {}
    for val in input_y:
        val = val.item()
        if val in class_num_dict.keys():
            class_num_dict[val] += 1
        else:
            class_num_dict[val] = 1
    for index, val in enumerate(input_y):
        minus_self = class_num_dict[val.item()] - 1
        class_num[index][0] = minus_self if minus_self > 0 else 1
    return class_num


def get_class_mask(input_y):
    batch_size = len(input_y)
    class_mask = torch.zeros([batch_size, batch_size])
    for row, i in enumerate(input_y):
        for col, j in enumerate(input_y):
            if i == j:
                class_mask[row, col] = 1
    return class_mask


def get_self_mask(batch_size):
    class_mask = torch.ones([batch_size, batch_size])
    for row in range(batch_size):
        for col in range(batch_size):
            if row == col:
                class_mask[row, col] = 0
    return class_mask


class SCLLoss(nn.Module):
    def __init__(self, t):
        super(SCLLoss, self).__init__()
        self.t = t

    def forward(self, input_x, input_y):
        '''
        input_x : [batch_size, class_num]
        input_y : [batch_size]
        return loss: float
        '''
        batch_size = len(input_y)
        # L2 normalization
        input_x_l2 = func.normalize(input_x, p=2, dim=1)

        # 每个样本与其他样本的距离
        # sample_distance: [batch_size, batch_size]
        sample_distance = torch.mm(input_x_l2, input_x_l2.t())
        sample_distance = torch.exp(sample_distance / self.t)
        # 构造类别掩码矩阵
        class_mask = get_class_mask(input_y)
        # 构造自身相乘掩码矩阵
        self_mask = get_self_mask(batch_size)
        sample_distance_sum = sample_distance * self_mask
        sample_distance_sum = torch.sum(sample_distance_sum, dim=1, keepdim=True)  # [batch_size]
        sample_distance_sum_tile = sample_distance_sum.expand(batch_size, batch_size)
        sample_distance = torch.log(sample_distance / sample_distance_sum_tile)
        sample_distance = sample_distance * self_mask
        sample_distance = sample_distance * class_mask
        class_num = get_class_num(input_y)
        inner_class_distance_sum = -1.0 * torch.sum(sample_distance, dim=1, keepdim=True) / class_num
        # 当一个batch中，某个类别只有一个样本时，class_num = 1 -1 = 0， 当除以 class_num后，产生inf值, 故将inf置为0。
        inner_class_distance_sum = torch.where(torch.isinf(inner_class_distance_sum), torch.full_like(inner_class_distance_sum, 0), inner_class_distance_sum)
        inner_class_distance_sum = torch.where(torch.isnan(inner_class_distance_sum), torch.full_like(inner_class_distance_sum, 0), inner_class_distance_sum)
        inner_class_distance_sum = torch.sum(inner_class_distance_sum)
        return inner_class_distance_sum
