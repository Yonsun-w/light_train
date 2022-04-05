import netCDF4 as nc
import numpy as np
from config import read_config
import datetime
import os


# path传入的是npy文件的路径
def createDistanceNc(config_dict, npyPath, ncPath ='test.nc'):
    f_w = nc.Dataset('wjhTest.nc', 'w', format='NETCDF4')  # 创建一个格式为.nc的，名字为 ‘hecheng.nc’的文件
    # 确定基础变量的维度信息。相对与坐标系的各个轴(x,y,z)
    f_w.createDimension('south_north', config_dict['GridRowColNum'])
    f_w.createDimension('west_east', config_dict['GridRowColNum'])
    f_w.createDimension('Time', config_dict['ForecastHourNum'])
    # 创建变量。参数依次为：‘变量名称’，‘数据类型’，‘基础维度信息’
    f_w.createVariable('Flash_pre', np.float32, ('south_north', 'west_east', 'Time'))

    # 按小时来写入结果
    for hour_plus in range(config_dict['ForecastHourNum']):
        dt_d = datetime.datetime.strptime(config_dict['Datetime'], '%Y%m%d%H%M') + datetime.timedelta(hours=hour_plus)
        dis_npy_path = os.path.join(config_dict['ResultDistanceSavePath'],'{}_h{}.npy'.format(dt_d.strftime('%Y%m%d%H%M'), hour_plus))
        f_w.variables['Flash_pre'][hour_plus] = np.load(dis_npy_path)

    f_w.variables['Flash_pre'].MemoryOrder  = 'XY'
    f_w.variables['Flash_pre'].units = 'BJTime'

    f_w.variables['Flash_pre'].description = 'hourly grid prediction lightning'
    f_w.variables['Flash_pre'].coordinates = 'XLONG XLAT'
    f_w.variables['Flash_pre'].init_time = '20201003_210000'

    f_w.variables['Flash_pre'].valid_time = '20201003_220000'

    # 关闭文件
    f_w.close()




if __name__ == "__main__":

    config_dict = read_config()
    # 按小时来写入结果
    for hour_plus in range(config_dict['ForecastHourNum']):
        # 首先获取他写入的路径
        dt_d = datetime.datetime.strptime(config_dict['Datetime'], '%Y%m%d%H%M') + datetime.timedelta(hours=hour_plus)
        equal_dis_path = os.path.join(config_dict['ResultDistanceSavePath'], '{}_h{}.npy'.format(dt_d.strftime('%Y%m%d%H%M'), hour_plus))


    createDistanceNc(config_dict)