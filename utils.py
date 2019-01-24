'''Some helper functions for PyTorch, including:
    - progress_bar: progress bar mimic xlua.progress.
    - set_lr : set the learning rate
    - clip_gradient : clip gradient
'''

import os
import sys
import time
import math
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Function

_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 31


def cal_run_time(start_time):
    ss = int(time.time() - start_time)
    mm = 0
    hh = 0
    dd = 0
    if ss >= 60:
        mm = ss / 60
        ss %= 60
    if mm >= 60:
        hh = mm / 60
        mm %= 60
    if hh >= 24:
        dd = hh / 24
        hh %= 24

    str = ''
    if dd != 0:
        str += '%dD ' % dd
    if hh != 0:
        str += '%dh ' % hh
    if mm != 0:
        str += '%dm ' % mm
    if ss != 0:
        str += '%ds' % ss

    return str


def progress_bar(current, total, msg=None):
    cur_len = int(TOTAL_BAR_LENGTH * (current + 1)/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for _ in range(cur_len):
        sys.stdout.write('=')
    if cur_len != TOTAL_BAR_LENGTH:
        sys.stdout.write('>')
    for _ in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    L = []
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for _ in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    str = ' %d/%d ' % (current + 1, total)
    for _ in range(term_width - 3 - int((TOTAL_BAR_LENGTH - len(str)) / 2)):
        sys.stdout.write('\b')
    sys.stdout.write(str)

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        #print(group['params'])
        for param in group['params']:
            param.grad.data.clamp_(-grad_clip, grad_clip)
