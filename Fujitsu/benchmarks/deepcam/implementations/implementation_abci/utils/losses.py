# The MIT License (MIT)
#
# Copyright (c) 2018 Pyjcsx
# Modifications Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import torch
from torch.autograd import Variable
from torch import nn
import numpy as np
import time

criterion = None

def fp_loss(logit, target, weight, fpw_1=0, fpw_2=0):
    global criterion
    ts_01 = time.perf_counter()

    n, c, h, w = logit.size()
    # logit = logit.permute(0, 2, 3, 1)
    target = target.squeeze(1)
    
    ts_02 = time.perf_counter()
    #later should use cuda
    if criterion is None:
    #if True:
        criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array(weight)).float().to(target.device), reduction='none')
        ts_025 = time.perf_counter()
        if torch.cuda.is_available():
            criterion = criterion.cuda()

    ts_03 = time.perf_counter()
    losses = criterion(logit, target.long())
    
    ts_04 = time.perf_counter()
    preds = torch.max(logit, 1)[1]
    
    #is fp 1
    is_fp_one = (torch.eq(preds, 1) & torch.ne(preds, 1)).float()
    fp_matrix_one = (is_fp_one * fpw_1) + 1
    losses = torch.mul(fp_matrix_one, losses)
        
    ts_05 = time.perf_counter()
    #is fp 1
    is_fp_two = (torch.eq(preds, 2) & torch.ne(preds, 2)).float()
    fp_matrix_two = (is_fp_two * fpw_2) + 1
    losses = torch.mul(fp_matrix_two, losses)
    
    ts_06 = time.perf_counter()
    loss = torch.mean(losses)
    ts_07 = time.perf_counter()

    #print("fp_loss(): squeeze: {}".format(ts_02 - ts_01))
    #print("fp_loss(): crossentr: {}".format(ts_03 - ts_02))
    #print("fp_loss(): crossentr transfer: {}".format(ts_03 - ts_025))
    #print("fp_loss(): criterion: {}".format(ts_04 - ts_03))
    #print("fp_loss(): mul1: {}".format(ts_05 - ts_04))
    #print("fp_loss(): mul2: {}".format(ts_06 - ts_05))
    #print("fp_loss(): mean: {}".format(ts_07 - ts_06))
    return loss, losses

criterion_cpu = None

def fp_loss_cpu(logit, target, weight, fpw_1=0, fpw_2=0):
    global criterion_cpu
    ts_01 = time.perf_counter()

    n, c, h, w = logit.size()
    # logit = logit.permute(0, 2, 3, 1)
    target = target.squeeze(1)
    
    ts_02 = time.perf_counter()
    if criterion_cpu is None:
    #if True:
        criterion_cpu = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array(weight)).float(), reduction='none')
        ts_025 = time.perf_counter()

    ts_03 = time.perf_counter()
    losses = criterion_cpu(logit, target.long())
    
    ts_04 = time.perf_counter()
    preds = torch.max(logit, 1)[1]
    
    #is fp 1
    is_fp_one = (torch.eq(preds, 1) & torch.ne(preds, 1)).float()
    fp_matrix_one = (is_fp_one * fpw_1) + 1
    losses = torch.mul(fp_matrix_one, losses)
        
    ts_05 = time.perf_counter()
    #is fp 1
    is_fp_two = (torch.eq(preds, 2) & torch.ne(preds, 2)).float()
    fp_matrix_two = (is_fp_two * fpw_2) + 1
    losses = torch.mul(fp_matrix_two, losses)
    
    ts_06 = time.perf_counter()
    loss = torch.mean(losses)
    ts_07 = time.perf_counter()

    #print("fp_loss(): squeeze: {}".format(ts_02 - ts_01))
    #print("fp_loss(): crossentr: {}".format(ts_03 - ts_02))
    #print("fp_loss(): crossentr transfer: {}".format(ts_03 - ts_025))
    #print("fp_loss(): criterion: {}".format(ts_04 - ts_03))
    #print("fp_loss(): mul1: {}".format(ts_05 - ts_04))
    #print("fp_loss(): mul2: {}".format(ts_06 - ts_05))
    #print("fp_loss(): mean: {}".format(ts_07 - ts_06))
    return loss, losses
