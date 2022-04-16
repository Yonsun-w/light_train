# -*- coding: utf-8 -*-
import numpy as np
import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as py_Dataset
import datetime
from config import read_config
from layers.ADSNet_model import ADSNet_Model
from layers.LightNet_model import LightNet_Model
from layers.OnlyObsNet_model import OnlyObsNet_Model
from layers.OnlyWRFNet_model import OnlyWRFNet_Model
from generator import DataGenerator
from scores import Cal_params_epoch, Model_eval
from generator import getTimePeriod

def selectModel(config_dict):
    if config_dict['NetName'] == 'ADSNet':
        model = ADSNet_Model(obs_tra_frames=config_dict['TruthHistoryHourNum'], obs_channels=1, wrf_tra_frames=config_dict['ForecastHourNum'],
                  wrf_channels=config_dict['WRFChannelNum'], config_dict=config_dict).to(config_dict['Device'])
    elif config_dict['NetName'] == 'LightNet':
        model = LightNet_Model(obs_tra_frames=config_dict['TruthHistoryHourNum'], obs_channels=1, wrf_tra_frames=config_dict['ForecastHourNum'],
                             wrf_channels=config_dict['WRFChannelNum'], config_dict=config_dict).to(config_dict['Device'])
    elif config_dict['NetName'] == 'OnlyObs':
        model = OnlyObsNet_Model(obs_tra_frames=config_dict['TruthHistoryHourNum'], obs_channels=1,
                               pre_frames=config_dict['ForecastHourNum'], config_dict=config_dict).to(config_dict['Device'])
    elif config_dict['NetName'] == 'OnlyWRF':
        model = OnlyWRFNet_Model(wrf_tra_frames=config_dict['ForecastHourNum'],
                             wrf_channels=config_dict['WRFChannelNum'], config_dict=config_dict).to(config_dict['Device'])
    else:
        print('`{}` not support'.format(config_dict['NetName']))
        assert False
    return model

def time_data_iscomplete(time_str, config_dict):

    time_str = time_str.rstrip('\n')
    time_str = time_str.rstrip('\r\n')
    if time_str == '':
        return False
    ddt = datetime.datetime.strptime(time_str, "%Y%m%d%H%M")
    is_complete = True
    # 首先检查wrf
    wrf_path = os.path.join(config_dict['WRFFileDir'], ddt.strftime("%Y%m%d"))
    wrf_data = ['00','06','12','18']
    for s in wrf_data :
        if not os.path.isdir(os.path.join(wrf_path,s)):
            is_complete = False

    #检测obs真实数据
    obs_path = os.path.join(config_dict['TruthFileDir'], ddt.strftime("%Y"))
    if not os.path.exists(os.path.join(obs_path, ddt.strftime("%Y_%m_%d") + '.txt')):
        is_complete = False

    # read WRF
    # UTC是世界时
    utc = ddt + datetime.timedelta(hours=-8)
    ft = utc + datetime.timedelta(hours=(-6))
    nchour, delta_hour = getTimePeriod(ft)
    delta_hour += 6
    npyFilepath = os.path.join(config_dict['WRFFileDir'], ft.strftime("%Y%m%d"), nchour)
    if not os.path.exists(npyFilepath):
        is_complete = False

    # read labels
    for hour_plus in range(config_dict['ForecastHourNum']):
        dt = ddt + datetime.timedelta(hours=hour_plus)
        tFilePath = config_dict['TruthFileDirGrid'] + dt.strftime('%Y%m%d%H%M') + '_truth' + '.npy'
        if not os.path.exists(tFilePath):
            is_complete = False
    # read history observations
    for hour_plus in range(config_dict['TruthHistoryHourNum']):
        dt = ddt + datetime.timedelta(hours=hour_plus - config_dict['TruthHistoryHourNum'])
        tFilePath = config_dict['TruthFileDirGrid'] + dt.strftime('%Y%m%d%H%M') + '_truth.npy'
        if not os.path.exists(tFilePath):
            is_complete = False
    return is_complete



def DoTrain(config_dict):
    # data index
    TrainSetFilePath = 'TrainCase.txt'
    ValSetFilePath = 'ValCase.txt'
    train_list = []
    with open(TrainSetFilePath) as file:
        for line in file:
            # 由于数据不全 所以需要校验数据的完整
            if time_data_iscomplete(line, config_dict):
                train_list.append(line.rstrip('\n').rstrip('\r\n'))
    val_list = []
    with open(ValSetFilePath) as file:
        for line in file:
            # 由于数据不全 所以需要校验数据的完整
            if time_data_iscomplete(line, config_dict):
                val_list.append(line.rstrip('\n').rstrip('\r\n'))

    print('加载数据完毕，一共有{}训练集，val{}测试集'.format(len(train_list), len(val_list)))

    # data
    train_data = DataGenerator(train_list, config_dict)
    train_loader = DataLoader(dataset=train_data, batch_size=config_dict['Batchsize'], shuffle=True, num_workers=0)
    val_data = DataGenerator(val_list, config_dict)
    val_loader = DataLoader(dataset=val_data, batch_size=config_dict['Batchsize'], shuffle=True, num_workers=0)

    # model
    model = selectModel(config_dict)

    # loss function
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(16))

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config_dict['LearningRate'])

    # eval
    model_eval_valdata = Model_eval(config_dict)


    for epoch in range(config_dict['EpochNum']):
        # train_calparams_epoch = Cal_params_epoch()
        for i, (X, y) in enumerate(train_loader):
            wrf, obs = X
            label = y
            wrf = wrf.to(config_dict['Device'])

            obs = obs.to(config_dict['Device'])

            label = label.to(config_dict['Device'])

            pre_frames = model(wrf, obs)


            # backward
            optimizer.zero_grad()
            loss = criterion(torch.flatten(pre_frames), torch.flatten(label))
            loss.backward()

            # update weights
            optimizer.step()

            # output
            print('TRAIN INFO: epoch:{} ({}/{}) loss:{:.5f}'.format(epoch, i+1, len(train_loader), loss.item()))
            # pod, far, ts, ets = train_calparams_epoch.cal_batch(label, pre_frames)
            # sumpod, sumfar, sumts, sumets = train_calparams_epoch.cal_batch_sum(label, pre_frames)
            # info = 'TRAIN INFO: epoch:{} ({}/{}) loss:{:.5f}\nPOD:{:.5f}  FAR:{:.5f}  TS:{:.5f}  ETS:{:.5f}\nsumPOD:{:.5f}  sumFAR:{:.5f}  sumTS:{:.5f}  sumETS:{:.5f}\n'\
            #     .format(epoch, i+1, len(train_loader), loss.item(), pod, far, ts, ets, sumpod, sumfar, sumts, sumets)
            # print(info)

        model_eval_valdata.eval(val_loader, model, epoch)

# 生产原始训练数据,主要工作室生成各个时间段内的闪电npy文件 flag代表是否覆盖生成，默认为否
def init_old_data(config_dict,flag = False):

    # generator light grid
    if os.path.exists(config_dict['TruthFileDirGrid']) and len(glob.glob(config_dict['TruthFileDirGrid']+'*')) != 0 and not flag:
        print('Light grid data existed')
    else:
        from ConvertToGird import LightingToGird
        print('Converting light data to grid...')
        light_grid_generator = LightingToGird(config_dict)
        if not os.path.exists(config_dict['TruthFileDirGrid']):
            os.makedirs(config_dict['TruthFileDirGrid'])

        st = datetime.datetime.strptime(config_dict['ScanStartTime'], '%Y%m%d%H%M')
        et = datetime.datetime.strptime(config_dict['ScanEndTime'], '%Y%m%d%H%M')
        tt = st
        datetimelist = []
        while (tt < et):
            datetimelist.append(tt)
            tt += datetime.timedelta(hours=1)
        for dt in datetimelist:
            truthfilepath = os.path.join(config_dict['TruthFileDir'], dt.strftime('%Y'), dt.strftime('%Y_%m_%d') + '.txt')

            if not os.path.exists(truthfilepath):
                print('Lighting data file `{}` not exist!'.format(truthfilepath))
                continue
            grid = light_grid_generator.getPeroid1HourGridFromFile(truthfilepath, dt)
            dt_str = dt.strftime('%Y%m%d%H%M')
            truthgridfilename = dt_str + '_truth'
            truthgridfilepath = config_dict['TruthFileDirGrid'] + truthgridfilename
            if not os.path.exists(truthgridfilepath) or flag:
                np.save(truthgridfilepath + '.npy', grid)
                print('{}_truth.npy generated successfully'.format(dt_str))
            else:
                print('{}.npy已经存在并且模式为不覆盖'.format(truthgridfilepath))


    # Constructing set automatically
    if os.path.exists('TrainCase.txt') and os.path.exists('ValCase.txt'):
        print('set list existed')
    else:
        from ConstructSet import constructSet
        print('Constructing train and val set automatically...')
        constructSet(config_dict)

    if not os.path.isdir(config_dict['ModelFileDir']):
        os.makedirs(config_dict['ModelFileDir'])

    if not os.path.isdir(config_dict['RecordFileDir']):
        os.makedirs(config_dict['RecordFileDir'])

    print('初始化数据已经生产完毕,现在可以开始训练了')




if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '6'

    config_dict = read_config()

    init_old_data(config_dict)

    # #train
    DoTrain(config_dict)



