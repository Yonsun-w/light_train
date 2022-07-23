# -*- coding: GBK -*-
import numpy as np
import datetime
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
        self.lon_gap = config_dict['lonGap']
        # n代表纬度上的
        self.lat_gap = config_dict['latGap']
        # 计算横着的长度 也就是根据经度计算 由于中国在上半球 所以我们直接计算纬度最大的那部分就好 也就是下边那条线
        distance_row = self._cal_distance(self.lat_max, self.lon_min, self.lat_max, self.lon_max)
        # 计算竖着的长度
        distance_col = self._cal_distance(self.lat_min, self.lon_min, self.lat_max, self.lon_min)
        # 矩阵的行 为
        self.row = int(distance_row / self.lat_gap)
        # 矩阵的列
        self.col = int(distance_col / self.lon_gap)

        # 如果是等经纬度 直接算就可以了
        if self.config_dict['equal_dis'] == 0:
            self.lon_max = int(self.lon_max * 10000)
            self.lon_min = int(self.lon_min * 10000)
            self.lon_gap = int(self.lon_gap * 10000)
            self.lat_gap = int(self.lat_gap * 10000)
            self.lat_max = int(self.lat_max * 10000)
            self.lat_min = int(self.lat_min * 10000)
            self.row = int((self.lat_max - self.lat_min) / self.lat_gap)
            self.col = int((self.lon_max - self.lon_min) / self.lon_gap)
            self.lon_max = float(self.lon_max) / 10000
            self.lon_min = float(self.lon_min) / 10000
            self.lon_gap = float(self.lon_gap) / 10000
            self.lat_gap = float(self.lat_gap) / 10000
            self.lat_max = float(self.lat_max) / 10000
            self.lat_min = float(self.lat_min) / 10000



        # 返回的矩阵
        self.grid = np.zeros(shape=[self.row, self.col], dtype=float)
        # 开始扫描的时间段
        self.start_time = datetime.datetime.strptime(config_dict['startTime'], "%Y%m%d%H%M%S")
        self.end_time = datetime.datetime.strptime(config_dict['endTime'], "%Y%m%d%H%M%S")

        # 代表时间纬度 用于nc文件存储上
        self.time = 0
        # 时间分辨率 存储了每个要输出的时间段，每个时间段对应一个nc文件
        self.time_slot = []
        st = self.start_time
        while st <= self.end_time:
            self.time += 1
            self.time_slot.append(st)
            # 时间分辨率
            st += datetime.timedelta(hours=config_dict['timeGap'])


        dataTime = self.start_time
        # 存在的日期文件列表 注意 这里存放的文件是不带父目录的 父目录为 config_dict['TruthFileDir']
        # 这里只保存日期时间 不保留具体的小时
        self.txtFiles = []
        while (dataTime <= self.end_time):
            truthfilepath = self.config_dict['TruthFileDir'] + dataTime.strftime('%Y_%m_%d') + '.txt'
            if not os.path.exists(truthfilepath):
                print('Lighting data file `{}` not exist!'.format(truthfilepath))
            else:
                self.txtFiles.append(dataTime.strftime('%Y_%m_%d') + '.txt')
            dataTime += datetime.timedelta(days=1)




        if not os.path.isdir(self.config_dict['outputFileDir']):
            print('您的nc文件输出目录指定不存在，请检查')

        if not os.path.isdir(self.config_dict['TruthFileDir']):
            print('您的txt文件输入目录指定不存在，请检查')


        print('-----txt数据格点化文件初始化完毕, 这些文件可用={}，等待写入中----------'.format(self.txtFiles))





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

    # 将文件夹下所有的txt记录数据转换为nc数据格式
    def all_trueFile_txt_to_wrf(self):
        print('-----------------开始输出nc--------------')
        # 获取每个时间段
        for time in self.time_slot:
            truthfilepath = self.config_dict['TruthFileDir'] + time.strftime('%Y_%m_%d') + '.txt'
            if not os.path.exists(truthfilepath):
                continue
            output_file =os.path.join(self.config_dict['outputFileDir'], time.strftime('%Y_%m_%d_%H_%M') + '.nc')
            self.create_nc(output_file,truthfilepath, time)
        print('您好，您的txt文件={},已经转化为格点化数据'.format(truthfilepath))
        print('------------------nc文件输出完毕----------------------')

    # 将文件夹下所有的txt记录数据转换为npy数据格式
    def all_trueFile_txt_to_npy(self):
        print('-----------------开始输出npz--------------')
        # 获取每个时间段
        for time in self.time_slot:
            truthfilepath = self.config_dict['TruthFileDir'] + time.strftime('%Y_%m_%d') + '.txt'
            if not os.path.exists(truthfilepath):
                # print('Lighting data file  `{}` not exist! (这个文件不存在 {})'.format(truthfilepath, truthfilepath))
                continue
            output_file = os.path.join(self.config_dict['outputFileDir'],time.strftime('%Y_%m_%d_%H_%M') + '.npy')
            grid = self.txt_to_grid(truthfilepath,time)

            if (np.sum(grid>0)) :
                print('{},有{}'.format(output_file, np.sum(grid)))
            np.save(output_file, grid)

        print('您好，您的txt文件={},已经转化为格点化数据'.format(truthfilepath))

        print('-----------------npz输出完毕--------------')




    # 三个参数 分别是 输出文件夹,读取的txt文件,和time当前时间段
    def create_nc(self, output_path, txt_path, time):
        f_w = nc.Dataset(output_path, 'w', format='NETCDF4')  # 创建一个格式为.nc的
        if self.config_dict['equal_dis'] == 1:
            f_w.FileOrigins = 'equidistant'
            lon_lat_Gap = '{}km'.format(self.config_dict['latGap'])
            f_w.delta_dis = lon_lat_Gap
        else :
            f_w.FileOrigins = 'equal_latlon'
            # 这里等经纬度 等距离不知道怎么写
            f_w.delta_lat = self.config_dict['latGap']
            f_w.delta_lon = self.config_dict['lonGap']

        f_w.lon_begin = self.lon_min
        f_w.lon_end = self.lon_max
        f_w.lat_begin = self.lat_min
        f_w.lat_end = self.lat_max

        time_gap = int(self.config_dict['timeGap'])

        # 这里设置的是时间分辨率
        f_w.delta_time = time_gap
        # --------------------上述是写入头文件-----------------------------
        # 确定基础变量的维度信息。相对与坐标系的各个轴(x,y,z)
        f_w.createDimension('south_north', self.row)
        f_w.createDimension('west_east', self.col)
        f_w.createVariable('Flash_pre', np.float32, ('south_north', 'west_east'))
        f_w.variables['Flash_pre'].MemoryOrder = 'XY'
        f_w.variables['Flash_pre'].units = 'BJTime'
        f_w.variables['Flash_pre'].description = 'hourly grid prediction lightning'
        f_w.variables['Flash_pre'].coordinates = 'XLONG XLAT'

        cur_time = datetime.datetime.strftime(time, "%Y%m%d%H%M%S")
        data = datetime.datetime.strptime(cur_time, "%Y%m%d%H%M%S")
        start = datetime.datetime.strftime(data, "%Y%m%d_%H%M%S")
        data = data + datetime.timedelta(hours=time_gap)
        end = datetime.datetime.strftime(data, "%Y%m%d_%H%M%S")
        f_w.variables['Flash_pre'].FillValue = 1e+20
        f_w.variables['Flash_pre'].init_time = start
        f_w.variables['Flash_pre'].valid_time = end
        # --------------------上述是创建变量信息-----------------------------x
        # 获取该时间段下的矩阵数据 也就是该时间分辨率下的格点数据
        cur_grid = self.txt_to_grid(txt_path, time)
        f_w.variables['Flash_pre'][:] = cur_grid
        # 关闭文件
        f_w.close()


    # 数据转化为矩阵 将t1时间段内的闪电格点化
    def txt_to_grid(self, tFilePath, t1):
        # 当前时间段的矩阵 最多为1 他的形状和grid是一样的
        cur_grid = np.zeros(shape=[self.row, self.col], dtype=int)
        t2 = t1 + datetime.timedelta(hours=self.config_dict['timeGap'])


        #todo del
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
                    continue
                # 绘制到grid上
                lat = lightning_data['lat']
                lon = lightning_data['lon']
                x = int(self._cal_distance(lat, lon, lat, self.lon_min) / self.lat_gap)
                y = int(self._cal_distance(lat, lon, self.lat_min, lon) / self.lon_gap)

                # 如果是等经纬度 直接算
                if self.config_dict['equal_dis'] == 0:
                    y = int((lon - self.lon_min) / self.lat_gap)
                    x = int((lat - self.lat_min) / self.lon_gap)


                # 假如坐标越界，抛弃
                if x >= self.row or y >= self.col:
                    continue

                cur_grid[x][y] += 1

        return cur_grid


    # 检测该点是否在范围呢
    def check_in_area(self, linedata):
        la = float(linedata['lat'])
        lo = float(linedata['lon'])
        if lo < self.lon_min or lo > self.lon_max or la < self.lat_min or la > self.lat_max:
            return False

        return True


if __name__ == "__main__":
    nc_path = 'TrueNc/2021_07_11_04_00.nc'
    f_w = nc.Dataset(nc_path, 'r', format='NETCDF4')  # 创建一个格式为.nc的
