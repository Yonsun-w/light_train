import os

import numpy
import numpy as np
import torch

# 读取由wrf转成的npy格式的文件
def read_wrfnpy():
    root = '/data/wenjiahua/light_data/ADSNet_testdata/WRF_data/20200831'
    time = ['00', '06', '12', '18']
    W_maxs = np.zeros([4,25,1,159,159])
    for index,t in enumerate(time):
        w_max = np.load(os.path.join(root,t,'W_max.npy'))
        W_maxs[index] = w_max
        print(w_max.shape)
    print(W_maxs[3].shape)

#该方法是对 getHoursGridFromSmallNC 的重写  getHoursGridFromSmallNC 方法读取的是nc文件
# 由于之前学长们已经处理过了，nc转成了多个npy，这里我们直接读取npy文件就好
def getHoursGridFromSmallNC_npy(npy_father_filepath, delta_hour, config_dict):  # 20200619
    variables3d = ['QICE', 'QGRAUP', 'QSNOW']
    variables2d = ['W_max']
    sumVariables2d = ['RAINNC']
    param_list = ['QICE', 'QSNOW', 'QGRAUP', 'W_max', 'RAINNC']

    m = config_dict['GridRowColNum']
    n = config_dict['GridRowColNum']
    grid_list = []
    if config_dict['WRFChannelNum'] == 217:
        grid = np.load(os.path(npy_father_filepath, 'V.npy'))
        grid = grid[delta_hour:delta_hour + config_dict['ForecastHourNum'], :, 0:m, 0:n]
        grid = np.transpose(grid, (0, 2, 3, 1))  # (12, 159, 159, n)
        with Dataset(ncfilepath) as nc:
            grid = nc.variables['varone'][delta_hour:delta_hour + config_dict['ForecastHourNum'], :, 0:m, 0:n]
            grid = np.transpose(grid, (0, 2, 3, 1))  # (12, 159, 159, n)
    else:
        with Dataset(ncfilepath) as nc:
            for s in param_list:
                if s in variables3d:
                    temp = nc.variables[s][delta_hour:delta_hour + config_dict['ForecastHourNum'], :, 0:m, 0:n]  # (12, 27, 159, 159)
                    temp[temp < 0] = 0
                    if config_dict['WRFChannelNum'] == 29:
                        ave_3 = np.zeros((config_dict['ForecastHourNum'], m, n, 9))
                        for i in range(9):
                            ave_3[:, :, :, i] = np.mean(temp[:, 3*i:3*(i+1), :, :], axis=1) # (12, 159, 159, 9)
                        grid_list.append(ave_3)
                    else:
                        temp = np.transpose(temp, (0, 2, 3, 1))  # (12, 159, 159, 27)
                        grid_list.append(temp)
                elif s in variables2d:
                    if s == 'W_max':
                        tmp = nc.variables['W'][delta_hour:delta_hour + config_dict['ForecastHourNum'], :, 0:m, 0:n]
                        tmp = np.transpose(tmp, (0, 2, 3, 1))
                        temp = np.max(tmp, axis=-1, keepdims=True)
                    else:
                        temp = nc.variables[s][delta_hour:delta_hour + config_dict['ForecastHourNum'], 0:m, 0:n]
                    grid_list.append(temp)
                elif s in sumVariables2d:
                    temp = nc.variables[s][delta_hour + 1:delta_hour + config_dict['ForecastHourNum'] + 1, 0:m, 0:n] - \
                           nc.variables[s][delta_hour:delta_hour + config_dict['ForecastHourNum'], 0:m, 0:n]
                    temp = temp[:, :, :, np.newaxis]
                    grid_list.append(temp)
        grid = np.concatenate(grid_list, axis=-1)
    return grid



def getHoursGridFromSmallNC_npy(ncfilepath, delta_hour, config_dict):  # 20200619
    variables3d = ['QICE', 'QGRAUP', 'QSNOW']
    variables2d = ['W_max']
    sumVariables2d = ['RAINNC']
    param_list = ['QICE', 'QSNOW', 'QGRAUP', 'W_max', 'RAINNC']

    m = config_dict['GridRowColNum']
    n = config_dict['GridRowColNum']
    grid_list = []
    if config_dict['WRFChannelNum'] == 217:
        with Dataset(ncfilepath) as nc:
            grid = nc.variables['varone'][delta_hour:delta_hour + config_dict['ForecastHourNum'], :, 0:m, 0:n]
            grid = np.transpose(grid, (0, 2, 3, 1))  # (12, 159, 159, n)
    else:
        with Dataset(ncfilepath) as nc:
            for s in param_list:
                if s in variables3d:
                    temp = nc.variables[s][delta_hour:delta_hour + config_dict['ForecastHourNum'], :, 0:m, 0:n]  # (12, 27, 159, 159)
                    temp[temp < 0] = 0
                    if config_dict['WRFChannelNum'] == 29:
                        ave_3 = np.zeros((config_dict['ForecastHourNum'], m, n, 9))
                        for i in range(9):
                            ave_3[:, :, :, i] = np.mean(temp[:, 3*i:3*(i+1), :, :], axis=1) # (12, 159, 159, 9)
                        grid_list.append(ave_3)
                    else:
                        temp = np.transpose(temp, (0, 2, 3, 1))  # (12, 159, 159, 27)
                        grid_list.append(temp)
                elif s in variables2d:
                    if s == 'W_max':
                        tmp = nc.variables['W'][delta_hour:delta_hour + config_dict['ForecastHourNum'], :, 0:m, 0:n]
                        tmp = np.transpose(tmp, (0, 2, 3, 1))
                        temp = np.max(tmp, axis=-1, keepdims=True)
                    else:
                        temp = nc.variables[s][delta_hour:delta_hour + config_dict['ForecastHourNum'], 0:m, 0:n]
                    grid_list.append(temp)
                elif s in sumVariables2d:
                    temp = nc.variables[s][delta_hour + 1:delta_hour + config_dict['ForecastHourNum'] + 1, 0:m, 0:n] - \
                           nc.variables[s][delta_hour:delta_hour + config_dict['ForecastHourNum'], 0:m, 0:n]
                    temp = temp[:, :, :, np.newaxis]
                    grid_list.append(temp)
        grid = np.concatenate(grid_list, axis=-1)
    return grid


if __name__ == '__main__':
    np.zeros([2,3,5])
    print(np.reshape([2,3].shape))





