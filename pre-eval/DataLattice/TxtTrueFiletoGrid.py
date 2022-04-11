# -*- coding: utf-8 -*-
import numpy as np
import datetime
import math
import os
import netCDF4 as nc
import config


class TxtTrueFiletoGrid(object):
    def __init__(self, config_dict):
        self.config_dict = config_dict
        self.lat_max = float(config_dict['latEnd'])
        self.lat_min = float(config_dict['latBegin'])
        self.lon_max = float(config_dict['lonEnd'])
        self.lon_min = float(config_dict['lonBegin'])

        # 单元格为 m km * n km
        # m代表经度上的单元格 也就是一个单元格 经度上的km数
        self.lon_gap = 4
        # n代表纬度上的
        self.lat_gap = 4
        # 计算横着的长度 也就是根据经度计算 由于中国在上半球 所以我们直接计算纬度最大的那部分就好 也就是下边那条线
        distance_row = self._cal_distance(self.lat_max, self.lon_min, self.lat_max, self.lon_max)
        # 计算竖着的长度
        distance_col = self._cal_distance(self.lat_min, self.lon_min, self.lat_max, self.lon_min)

        # 矩阵的行 为
        self.row = int(distance_row / self.lat_gap)
        # 矩阵的列
        self.col = int(distance_col / self.lon_gap)

        # 返回的矩阵
        self.grid = np.zeros(shape=[self.row, self.col], dtype=float)

        self.start_time = datetime.datetime.strptime(config_dict['startTime'], "%Y%m%d%H%M%S")
        self.end_time = datetime.datetime.strptime(config_dict['endTime'], "%Y%m%d%H%M%S")


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

    def txt_to_wrf(self):
        st = self.start_time
        et = self.end_time
        tt = st
        datetimelist = []
        while (tt < et):
            datetimelist.append(tt)
            tt += datetime.timedelta(hours=self.config_dict['timeGap'])
        for dt in datetimelist:
            truthfilepath = self.config_dict['TruthFileDir'] + dt.strftime('%Y_%m_%d') + '.txt'
            if not os.path.exists(truthfilepath):
                print('Lighting data file `{}` not exist!'.format(truthfilepath))
                continue
            cur_grid = self.txt_to_grid(truthfilepath, dt)
            self.grid += cur_grid
        return self.grid



    def _draw_to_grid_new(self, lightning_data, cur_grid):
        if not self.check_in_area(lightning_data):
            return
        lat = lightning_data['lat']
        lon = lightning_data['lon']
        row = int(self._cal_distance(lat, lon, lat, self.lon_min) / self.lat_gap)
        col = int(self._cal_distance(lat, lon, self.lat_min, lon) / self.lon_gap)
        if row >= self.row or col >= self.col:
            return
        cur_grid[row][col] = 1
        return cur_grid

    # 检测该点是否在范围呢
    def check_in_area(self, linedata):
        la = float(linedata['lat'])
        lo = float(linedata['lon'])
        datetime = linedata['data']
        if lo < self.lon_min or lo > self.lon_max or la < self.lat_min or la > self.lat_max or datetime < self.start_time or datetime > self.end_time:
            return False

        return True

    # 数据转化为矩阵
    def txt_to_grid(self, tFilePath, t1):
        # 当前时间段的矩阵 最多为1 他的形状和grid是一样的
        cur_grid = np.zeros(shape=[self.row, self.col], dtype=int)
        t2 = t1 + datetime.timedelta(hours=self.config_dict['timeGap'])
        with open(tFilePath, 'r', encoding='GBK') as tfile:
            for line in tfile:
                # 每条闪电数据
                lightning_data = {}
                linedata = line.split()
                temp_date = linedata[1]
                temp_time = linedata[2]
                temp_dt = datetime.datetime.strptime(temp_date + ' ' + temp_time[0:8], "%Y-%m-%d %H:%M:%S")
                lightning_data['data'] = temp_dt
                lightning_data['lat'] = float(linedata[3].lstrip('纬度='))
                lightning_data['lon'] = float(linedata[4].lstrip('经度='))
                if not t1 <= temp_dt <= t2:
                    continue
                # 如果不在范围内 则不绘制，否则 绘制
                if not self.check_in_area(lightning_data):
                    print('不绘制', lightning_data)
                    continue
                # 绘制到grid上
                cur_grid = self._draw_to_grid_new(lightning_data, cur_grid)
        return cur_grid


if __name__ == "__main__":
    config_dict = config.read_config()
    txt = TxtTrueFiletoGrid(config_dict)

    grid = txt.txt_to_wrf()
    print(grid.shape)
    print(grid[grid!=0].sum())
