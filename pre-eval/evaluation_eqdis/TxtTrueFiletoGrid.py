# -*- coding: utf-8 -*-
import numpy as np
import datetime
import math
import os
from netCDF4 import Dataset
import config

class TxtTrueFiletoGrid(object):
    def __init__(self, config_dict):
        self.config_dict = config_dict
        mn = 159
        self.lat_max = (float)(config_dict['latEnd'])
        self.lat_min = (float)(config_dict['latBegin'])
        self.lon_max = (float)(config_dict['lonEnd'])
        self.lon_min = (float)(config_dict['lonBegin'])

        # 单元格为 m km * n km
        # m代表经度上的单元格 也就是一个单元格 经度上的km数
        lon_m = 4
        # n代表纬度上的
        lat_n = 4
        # 计算起点到终点的距离,并且将它弄成矩形，也就是说计算的是对角线
        distance = self._cal_distance(self.lat_min, self.lon_min, self.lat_max, self.lon_max)
        # 边长为
        edge = distance/math.sqrt(2)
        # 矩阵的行 为
        row = (int)(edge / lon_m)
        # 矩阵的列
        col = (int)(edge / lat_n)


        latlon = np.zeros(shape=[row * col, 2], dtype=float)

        # 返回的矩阵
        grid = np.zeros(shape=[row, col], dtype=float)

        self.start_time = 0
        self.end_time = 22200711000236

        # forecast region to scan, rough estimation
        # self.lat_max = self.lat_max + (self.lat_max - self.lat_min) / mn
        # self.lat_min = self.lat_min - (self.lat_max - self.lat_min) / mn
        # self.lon_max = self.lon_max + (self.lon_max - self.lon_min) / mn
        # self.lon_min = self.lon_min - (self.lon_max - self.lon_min) / mn
        # single grid point range, rough estimation
        self.sin_distance = (self._cal_distance(latlon[0][0], latlon[0][1], latlon[1][0], latlon[1][1]) +
                             self._cal_distance(latlon[-1][0], latlon[-1][1], latlon[-2][0], latlon[-2][1])) / 2
        self.latlon = latlon

    # 数据转化为矩阵
    def txt_grid(self):
        tFilePath = 'real.txt'
        grid = np.zeros(159 * 159, dtype=int)
        with open(tFilePath, 'r', encoding='GBK') as tfile:
            for line in tfile:
                # 每条闪电数据
                lightning_data = {}
                linedata = line.split()
                temp_date = linedata[1]
                temp_time = linedata[2]
                temp_dt = datetime.datetime.strptime(temp_date + ' ' + temp_time[0:8], "%Y-%m-%d %H:%M:%S")
                lightning_data['data'] = temp_dt.strftime("%Y%m%d%H%M%S")
                lightning_data['lat'] = linedata[3].lstrip('纬度=')
                lightning_data['lon'] = linedata[4].lstrip('经度=')
                # 如果不在范围内 则不绘制，否则 绘制
                if not self.check_in_area(lightning_data):
                    print('不绘制', lightning_data)
                    continue
                print('绘制', lightning_data)

    def _cal_distance(self, la1, lo1, la2, lo2):
        ER = 6370.8
        radLat1 = (np.pi / 180.0) * la1
        radLat2 = (np.pi / 180.0) * la2
        radLng1 = (np.pi / 180.0) * lo1
        radLng2 = (np.pi / 180.0) * lo2
        d = 2.0 * np.arcsin(np.sqrt(
            np.power(np.sin((radLat1 - radLat2) / 2.0), 2) + np.cos(radLat1) * np.cos(radLat2) * np.power(
                np.sin((radLng1 - radLng2) / 2.0), 2))) * ER
        return d

    def _lalo_to_grid_new(self, linedata):
        la = (float)(linedata['lat'])
        lo = (float)(linedata['lon'])
        datetime = (int)(linedata['data'])

        if lo < self.lon_min or lo > self.lon_max or la < self.lat_min or la > self.lat_max or datetime < self.start_time or datetime > self.end_time:
            return -1

        # 这是什么意思
        d = self._cal_distance(la, lo, self.latlon[:, 0], self.latlon[:, 1])
        if np.min(d) > self.sin_distance * np.sqrt(2):
            return -1
        idx = np.argmin(d)
        return idx

    # 检测该点是否在范围呢
    def check_in_area(self, linedata):
        la = (float)(linedata['lat'])
        lo = (float)(linedata['lon'])
        datetime = (int)(linedata['data'])

        if lo < self.lon_min or lo > self.lon_max or la < self.lat_min or la > self.lat_max or datetime < self.start_time or datetime > self.end_time:
            return False

        return True


    def getPeroid1HourGridFromFile(self, tFilePath, t1):
        mn = self.config_dict['GridRowColNum']
        grid = np.zeros(mn * mn, dtype=int)
        t2 = t1 + datetime.timedelta(hours=1)
        with open(tFilePath, 'r', encoding='GBK') as tfile:
            for line in tfile:
                linedata = line.split()
                temp_date = linedata[1]
                temp_time = linedata[2]
                temp_dt = datetime.datetime.strptime(temp_date + ' ' + temp_time[0:8], "%Y-%m-%d %H:%M:%S")
                if not t1 <= temp_dt <= t2:
                    continue
                linedata[3] = linedata[3].lstrip('纬度=')
                linedata[4] = linedata[4].lstrip('经度=')
                la = float(linedata[3])
                lo = float(linedata[4])
                idx = self._lalo_to_grid_new(la, lo)
                if idx == -1:
                    continue
                grid[idx] += 1
        return grid


    # 检查该条闪电数据是否在我们选取的范围内
    def checkavailable(self, linedata):
        is_available = True


if __name__ == "__main__":
    config = config.read_config()
    txt = TxtTrueFiletoGrid(config)

    # 计算起点到终点的距离,并且将它弄成矩形，也就是说计算的是对角线
    distance = txt._cal_distance(1,2, 3, 4)
    # 边长为
    edge = distance / math.sqrt(2)
    print(txt.latlon.shape)