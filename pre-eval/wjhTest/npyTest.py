import os

import numpy
import numpy as np
import torch

def readNc():
    # euqal_distanc_path = 'D:\\雷电预测文件\\雷电预测\\github\\light_train\\pre-eval\\evaluation_eqdis\\input_examples\\pre_dis\\Equal_Distance\\202006110150_h9.npy'
    # a = np.load(euqal_distanc_path)
    # print(np.sum(a == 0))
    # a = a[:,1]
    #
    # print(a.shape)
    # 假设预测的闪电是2个小时 每个 大小是3行3列 3 * 3
    # a代表第一个小时的数据
    a = np.array([[0, 0, 3],[3, 3, 3],[3, 3, 3]])
    b = np.array([[1, 2, 3],[4, 5, 6],[3, 3, 3]])
    b = np.array(())
    res = np.zeros((2, 3, 3))
    res[0] = a
    print(res[0][0])
    print("-----")
    print(res[:1])


if __name__ == '__main__':
    a = torch.zeros(1,1,1).requires_grad()
    print(a.shape)
    print(a)



