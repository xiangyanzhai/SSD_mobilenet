# !/usr/bin/python
# -*- coding:utf-8 -*-
import numpy as np
import cv2

font = cv2.FONT_HERSHEY_SIMPLEX
chars_dic = {}
for i in range(10):
    t = chr(ord('0') + i)
    chars_dic[i] = t
    chars_dic[t] = i
chars_dic[10] = 'x'
chars_dic['x'] = 10


def process_ssd_res(self, res):
    inds = res[:, 4] > 0.05
    res = res[inds]

    hw = res[:, 2:4] - res[:, :2]
    inds = (hw[:, 0] >= 10) & (hw[:, 1] >= 15)
    res = res[inds]
    return res


def decode( v):
    if len(v) == 0:
        return ''
    v = [chars_dic[x] for x in v]
    print(v)
    if len(v) != 3:
        return '#'
    if v.count('x') > 1:
        return '#'
    if v.count('x') == 1:
        if v[0] == 'x':
            v[0] = str(int(v[2]) - int(v[1]))
        elif v[1] == 'x':
            v[1] = str(int(v[2]) - int(v[0]))
        else:
            v[2] = str(int(v[1]) + int(v[0]))
    if int(v[0]) + int(v[1]) == int(v[2]):
        return ''.join(v)
    return '#'
