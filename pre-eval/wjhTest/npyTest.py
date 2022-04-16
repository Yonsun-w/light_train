import os

import numpy
import numpy as np
import torch

# 读取由wrf转成的npy格式的文件


if __name__ == '__main__':
    path = '/home/wenjiahua/np_test.npy'
    grid = np.ones([2,3,5])
    grid = np.load(path)
    print(grid)





