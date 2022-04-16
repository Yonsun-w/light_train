import datetime

import numpy as np
import torch
import os
import config
import datetime


def time_data_iscomplete(time_str, WRFFileDir, TruthFileDir):
    time_str = time_str.rstrip('\n')
    time_str = time_str.rstrip('\r\n')
    dt = datetime.datetime.strptime(time_str, "%Y%m%d%H%M")
    is_complete = True
    # 首先检查wrf
    wrf_path = os.path.join(WRFFileDir, dt.strftime("%Y%m%d"))
    wrf_data = ['00','06','12','18']
    for s in wrf_data :
        if not os.path.isdir(os.path.join(wrf_path,s)):
            is_complete = False
        else :

            print(os.path.join(wrf_path,s, 'V.npy'))
            a = np.load(os.path.join(wrf_path,s, 'V.npy') )
            print(a[a!=0].sum())
    #检测obs真实数据
    obs_path = os.path.join(TruthFileDir,dt.strftime("%Y"))
    txt_path = os.path.join(obs_path, dt.strftime("%Y_%m_%d") + '.txt')
    if not os.path.exists(txt_path):
        is_complete = False
    return is_complete


def getDataPath():
    filePath = '/data/wenjiahua/light_data/ADSNet_testdata/light_data_ori_18-20'
    wrf_path = '/data/wenjiahua/light_data/ADSNet_testdata/WRF_data'
    save_path = '/data/wenjiahua/light_data/ADSNet_testdata/val.txt'
    f = open(save_path, 'w')
    start = datetime.datetime.strptime('20180101', '%Y%m%d')
    end = start + datetime.timedelta(days=700)
    tt = start
    while tt <= end :
        tt_str = datetime.datetime.strftime(tt,'%Y%m%d%H%M')

        if time_data_iscomplete(tt_str,wrf_path,filePath):
            tt_str = tt_str + '\n'

           # f.write(tt_str)
        tt += datetime.timedelta(hours=1)
    f.close()
    print('end')



if __name__ == '__main__':
    getDataPath()
