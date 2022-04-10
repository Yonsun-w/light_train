import netCDF4 as nc
import numpy as np

def createNc():
    f_w = nc.Dataset('wjhTest.nc', 'w', format='NETCDF4')  # 创建一个格式为.nc的，名字为 ‘hecheng.nc’的文件

    # time纬度为12。注意，第2个参数 表示维度，但是必须是 integer整型，也就是只能创建一个基础单一维度信息。
    # 如果后面要创建一个变量维度>1，则必须由前面的单一维度组合而来。后面会介绍。
    # 确定基础变量的维度信息。相对与坐标系的各个轴(x,y,z)
    f_w.createDimension('south_north', 159)
    f_w.createDimension('west_east', 159)
    f_w.createDimension('Time', 12)
    ##创建变量。参数依次为：‘变量名称’，‘数据类型’，‘基础维度信息’
    f_w.createVariable('Flash_pre', np.float32, ('south_north', 'west_east', 'Time'))

    f_w.variables['Flash_pre'].MemoryOrder  = 'XY'

    f_w.variables['Flash_pre'].init_time = '20201003_210000'

    f_w.variables['Flash_pre'].valid_time = '20201003_220000'

    f_w.variables['Flash_pre'].units = 'BJTime'

    f_w.variables['Flash_pre'].description = 'hourly grid prediction lightning'

    f_w.variables['Flash_pre'].coordinates = 'XLONG XLAT'

    a = f_w.variables['Flash_pre']


    f_w.close()


def readNc():
    npc_path = 'wjhTest.nc'
    f = nc.Dataset(npc_path)
    print(f.variables['Flash_pre'][:])

def change():
    npc_path = 'wjhTest.nc'
    f = nc.Dataset(npc_path)
    print(f.variables['Flash_pre'])

def readNpy():
    npy_path = '/Users/yonsun/gitTest/light_train/pre-eval/pytorch_pre1.4/Result/Equal_Distance/202006101650_h0.npy'
    print(np.load(npy_path).shape)


if __name__ == '__main__':
    readNpy() 


