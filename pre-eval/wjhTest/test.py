import datetime

import numpy as np
import torch
import os
import config




if __name__ == '__main__':
    filePath = '/data/wenjiahua/light_data/ADSNet_testdata/light_data_ori_18-20/2018'
    save_path = '/home/wenjiahua/data.txt'
    f = open(save_path, 'w')
    start = datetime.datetime.strptime('20180101', '%Y%m%d')
    end = start + datetime.timedelta(days=364)
    tt = start
    while tt <= end :
        txt_path = os.path.join(filePath, tt.strftime('%Y_%m_%d') + '.txt')
        if os.path.exists(txt_path):
            date = tt.strftime('%Y%m%d%H%M') + '\n'
            f.write(date)
        tt += datetime.timedelta(hours=3)
    f.close()
    print('end')
