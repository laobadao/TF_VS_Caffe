# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import numpy as np
cimport numpy as np
#cimport 是 Cython 中用来引入 .pxd 文件的命令。
# 有关 .pxd 文件，可以简单理解成 C/C++ 中用来写声明的头文件
'''
使用cdef关键字，用来定义C变量以及各种结构体，枚举类型变量.
注意：cdef只是用来定义类型，并不能用来作为引用对象。
'''
cdef inline np.float32_t max(np.float32_t a, np.float32_t b):#取a,b之间最大值
    return a if a >= b else b

cdef inline np.float32_t min(np.float32_t a, np.float32_t b):#取a,b之间最小值
    return a if a <= b else b

def cpu_nms(np.ndarray[np.float32_t, ndim=2] dets, np.float thresh):
    #np.ndarray[np.float32_t, ndim=2] 就是一个类型名就像 int 一样，只是它比较长
    # 而且信息量比较大而已。它的意思是，这是个类型为 np.float32_t 的2维 np.ndarray

    #依次取出左上角和右下角坐标以及分类器得分（置信度）
    cdef np.ndarray[np.float32_t, ndim=1] x1 = dets[:, 0]#X1是一维的
    cdef np.ndarray[np.float32_t, ndim=1] y1 = dets[:, 1]
    cdef np.ndarray[np.float32_t, ndim=1] x2 = dets[:, 2]
    cdef np.ndarray[np.float32_t, ndim=1] y2 = dets[:, 3]
    cdef np.ndarray[np.float32_t, ndim=1] scores = dets[:, 4]

    cdef np.ndarray[np.float32_t, ndim=1] areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    cdef np.ndarray[np.int_t, ndim=1] order = scores.argsort()[::-1]#将scores降序排列，并得到索引
    #where the [::-1] is used to sort the column from high to low, instead of low
    #  to high

    cdef int ndets = dets.shape[0]#读取第一维（行）的长度
    cdef np.ndarray[np.int_t, ndim=1] suppressed = \
            np.zeros((ndets), dtype=np.int)

    # nominal indices
    cdef int _i, _j
    # sorted indices
    cdef int i, j
    # temp variables临时变量 for box i's (the box currently under consideration)
    cdef np.float32_t ix1, iy1, ix2, iy2, iarea
    # variables for computing overlap with box j (lower scoring box)
    cdef np.float32_t xx1, yy1, xx2, yy2
    cdef np.float32_t w, h
    cdef np.float32_t inter, ovr

    keep = []
    for _i in range(ndets):
        i = order[_i]#获得排序后的索引
        if suppressed[i] == 1:
            continue
        keep.append(i)
        ix1 = x1[i]
        iy1 = y1[i]
        ix2 = x2[i]
        iy2 = y2[i]
        iarea = areas[i]
        for _j in range(_i + 1, ndets):#_j是除了当前_i之外其他的框
            j = order[_j]
            if suppressed[j] == 1:
                continue
            #计算相交面积
            xx1 = max(ix1, x1[j])## calculate the points of overlap
            yy1 = max(iy1, y1[j])
            xx2 = min(ix2, x2[j])
            yy2 = min(iy2, y2[j])
            w = max(0.0, xx2 - xx1 + 1)## the weights of overlap
            h = max(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (iarea + areas[j] - inter)
            if ovr >= thresh:#舍弃重叠率大于阈值的框
                suppressed[j] = 1

    return keep
