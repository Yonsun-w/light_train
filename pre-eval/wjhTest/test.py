import numpy as np
import torch
import os
import config




if __name__ == '__main__':
    father_path = '/data/wenjiahua/light_data/ADSNet_testdata/WRF_data/20200831/00'
    config_dic = config.read_config()
    getHoursGridFromSmallNC_npy(father_path,00,config_dic)

