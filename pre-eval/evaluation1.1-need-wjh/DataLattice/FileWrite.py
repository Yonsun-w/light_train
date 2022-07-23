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
        # ��Ԫ��Ϊ m km * n km
        # m�������ϵĵ�Ԫ�� Ҳ����һ����Ԫ�� �����ϵ�km��
        self.lon_gap = config_dict['lonGap']
        # n����γ���ϵ�
        self.lat_gap = config_dict['latGap']
        # ������ŵĳ��� Ҳ���Ǹ��ݾ��ȼ��� �����й����ϰ��� ��������ֱ�Ӽ���γ�������ǲ��־ͺ� Ҳ�����±�������
        distance_row = self._cal_distance(self.lat_max, self.lon_min, self.lat_max, self.lon_max)
        # �������ŵĳ���
        distance_col = self._cal_distance(self.lat_min, self.lon_min, self.lat_max, self.lon_min)
        # ������� Ϊ
        self.row = int(distance_row / self.lat_gap)
        # �������
        self.col = int(distance_col / self.lon_gap)

        # ����ǵȾ�γ�� ֱ����Ϳ�����
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



        # ���صľ���
        self.grid = np.zeros(shape=[self.row, self.col], dtype=float)
        # ��ʼɨ���ʱ���
        self.start_time = datetime.datetime.strptime(config_dict['startTime'], "%Y%m%d%H%M%S")
        self.end_time = datetime.datetime.strptime(config_dict['endTime'], "%Y%m%d%H%M%S")

        # ����ʱ��γ�� ����nc�ļ��洢��
        self.time = 0
        # ʱ��ֱ��� �洢��ÿ��Ҫ�����ʱ��Σ�ÿ��ʱ��ζ�Ӧһ��nc�ļ�
        self.time_slot = []
        st = self.start_time
        while st <= self.end_time:
            self.time += 1
            self.time_slot.append(st)
            # ʱ��ֱ���
            st += datetime.timedelta(hours=config_dict['timeGap'])


        dataTime = self.start_time
        # ���ڵ������ļ��б� ע�� �����ŵ��ļ��ǲ�����Ŀ¼�� ��Ŀ¼Ϊ config_dict['TruthFileDir']
        # ����ֻ��������ʱ�� �����������Сʱ
        self.txtFiles = []
        while (dataTime <= self.end_time):
            truthfilepath = self.config_dict['TruthFileDir'] + dataTime.strftime('%Y_%m_%d') + '.txt'
            if not os.path.exists(truthfilepath):
                print('Lighting data file `{}` not exist!'.format(truthfilepath))
            else:
                self.txtFiles.append(dataTime.strftime('%Y_%m_%d') + '.txt')
            dataTime += datetime.timedelta(days=1)




        if not os.path.isdir(self.config_dict['outputFileDir']):
            print('����nc�ļ����Ŀ¼ָ�������ڣ�����')

        if not os.path.isdir(self.config_dict['TruthFileDir']):
            print('����txt�ļ�����Ŀ¼ָ�������ڣ�����')


        print('-----txt���ݸ�㻯�ļ���ʼ�����, ��Щ�ļ�����={}���ȴ�д����----------'.format(self.txtFiles))





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

    # ���ļ��������е�txt��¼����ת��Ϊnc���ݸ�ʽ
    def all_trueFile_txt_to_wrf(self):
        print('-----------------��ʼ���nc--------------')
        # ��ȡÿ��ʱ���
        for time in self.time_slot:
            truthfilepath = self.config_dict['TruthFileDir'] + time.strftime('%Y_%m_%d') + '.txt'
            if not os.path.exists(truthfilepath):
                continue
            output_file =os.path.join(self.config_dict['outputFileDir'], time.strftime('%Y_%m_%d_%H_%M') + '.nc')
            self.create_nc(output_file,truthfilepath, time)
        print('���ã�����txt�ļ�={},�Ѿ�ת��Ϊ��㻯����'.format(truthfilepath))
        print('------------------nc�ļ�������----------------------')

    # ���ļ��������е�txt��¼����ת��Ϊnpy���ݸ�ʽ
    def all_trueFile_txt_to_npy(self):
        print('-----------------��ʼ���npz--------------')
        # ��ȡÿ��ʱ���
        for time in self.time_slot:
            truthfilepath = self.config_dict['TruthFileDir'] + time.strftime('%Y_%m_%d') + '.txt'
            if not os.path.exists(truthfilepath):
                # print('Lighting data file  `{}` not exist! (����ļ������� {})'.format(truthfilepath, truthfilepath))
                continue
            output_file = os.path.join(self.config_dict['outputFileDir'],time.strftime('%Y_%m_%d_%H_%M') + '.npy')
            grid = self.txt_to_grid(truthfilepath,time)

            if (np.sum(grid>0)) :
                print('{},��{}'.format(output_file, np.sum(grid)))
            np.save(output_file, grid)

        print('���ã�����txt�ļ�={},�Ѿ�ת��Ϊ��㻯����'.format(truthfilepath))

        print('-----------------npz������--------------')




    # �������� �ֱ��� ����ļ���,��ȡ��txt�ļ�,��time��ǰʱ���
    def create_nc(self, output_path, txt_path, time):
        f_w = nc.Dataset(output_path, 'w', format='NETCDF4')  # ����һ����ʽΪ.nc��
        if self.config_dict['equal_dis'] == 1:
            f_w.FileOrigins = 'equidistant'
            lon_lat_Gap = '{}km'.format(self.config_dict['latGap'])
            f_w.delta_dis = lon_lat_Gap
        else :
            f_w.FileOrigins = 'equal_latlon'
            # ����Ⱦ�γ�� �Ⱦ��벻֪����ôд
            f_w.delta_lat = self.config_dict['latGap']
            f_w.delta_lon = self.config_dict['lonGap']

        f_w.lon_begin = self.lon_min
        f_w.lon_end = self.lon_max
        f_w.lat_begin = self.lat_min
        f_w.lat_end = self.lat_max

        time_gap = int(self.config_dict['timeGap'])

        # �������õ���ʱ��ֱ���
        f_w.delta_time = time_gap
        # --------------------������д��ͷ�ļ�-----------------------------
        # ȷ������������ά����Ϣ�����������ϵ�ĸ�����(x,y,z)
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
        # --------------------�����Ǵ���������Ϣ-----------------------------x
        # ��ȡ��ʱ����µľ������� Ҳ���Ǹ�ʱ��ֱ����µĸ������
        cur_grid = self.txt_to_grid(txt_path, time)
        f_w.variables['Flash_pre'][:] = cur_grid
        # �ر��ļ�
        f_w.close()


    # ����ת��Ϊ���� ��t1ʱ����ڵ������㻯
    def txt_to_grid(self, tFilePath, t1):
        # ��ǰʱ��εľ��� ���Ϊ1 ������״��grid��һ����
        cur_grid = np.zeros(shape=[self.row, self.col], dtype=int)
        t2 = t1 + datetime.timedelta(hours=self.config_dict['timeGap'])


        #todo del
        with open(tFilePath, 'r', encoding='GBK') as tfile:
            for line in tfile:
                # ÿ����������
                lightning_data = {}
                linedata = line.split()
                temp_date = linedata[1]
                temp_time = linedata[2]
                temp_dt = datetime.datetime.strptime(temp_date + ' ' + temp_time[0:8], "%Y-%m-%d %H:%M:%S")
                lightning_data['data'] = temp_dt
                lightning_data['lat'] = float(linedata[3].lstrip('γ��='))
                lightning_data['lon'] = float(linedata[4].lstrip('����='))
                if not t1 <= temp_dt <= t2:
                    continue
                # ������ڷ�Χ�� �򲻻��ƣ����� ����
                if not self.check_in_area(lightning_data):
                    continue
                # ���Ƶ�grid��
                lat = lightning_data['lat']
                lon = lightning_data['lon']
                x = int(self._cal_distance(lat, lon, lat, self.lon_min) / self.lat_gap)
                y = int(self._cal_distance(lat, lon, self.lat_min, lon) / self.lon_gap)

                # ����ǵȾ�γ�� ֱ����
                if self.config_dict['equal_dis'] == 0:
                    y = int((lon - self.lon_min) / self.lat_gap)
                    x = int((lat - self.lat_min) / self.lon_gap)


                # ��������Խ�磬����
                if x >= self.row or y >= self.col:
                    continue

                cur_grid[x][y] += 1

        return cur_grid


    # ���õ��Ƿ��ڷ�Χ��
    def check_in_area(self, linedata):
        la = float(linedata['lat'])
        lo = float(linedata['lon'])
        if lo < self.lon_min or lo > self.lon_max or la < self.lat_min or la > self.lat_max:
            return False

        return True


if __name__ == "__main__":
    nc_path = 'TrueNc/2021_07_11_04_00.nc'
    f_w = nc.Dataset(nc_path, 'r', format='NETCDF4')  # ����һ����ʽΪ.nc��
