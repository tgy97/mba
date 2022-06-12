#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: An Tao
@Contact: ta19@mails.tsinghua.edu.cn
@File: utils.py
@Time: 2020/1/2 10:26 AM
"""

import os
import sys
import torch


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def learning_rate_decay(optimizer, t, lr_0):
    for param_group in optimizer.param_groups:
        lr = lr_0 / np.sqrt(1 + lr_0 * param_group['weight_decay'] * t)
        param_group['lr'] = lr
        

class Logger(object):
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()

def drop_random(input,num = 500):
    return input[:,:-num,:]

def SOR(input, num = 10):
    return knn(input,num)



def knn(x,num):
    x = x.permute(0, 2, 1).contiguous()
    # b x 3 x n 
    batch_size = x.size(0)
    num_points = x.size(2)

    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=2, dim=-1)[0]            # (batch_size, num_points, k)



    idx = idx.mean(-1) # b x n
    idx = idx.topk(k = num_points-num, dim = -1)[1]

    idx_base = torch.arange(0, batch_size).view(-1, 1)*num_points

    idx = idx+idx_base
    idx = idx.view(-1)

    x = x.transpose(2, 1).contiguous() 
    # b x n x 3

    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points-num, -1) 
    return feature
