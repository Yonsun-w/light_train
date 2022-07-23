import netCDF4 as nc
import os
import numpy
import numpy as np

if __name__ == '__main__':
    npy_path = os.path.join("TrueNc", "2020_05_21_00_00.npy")
    nc_path = os.path.join("TrueNc", "2020_05_21_00_00.nc")
    print(npy_path, os.path.exists(npy_path), nc_path, os.path.exists(npy_path))
    print("npy闪电一共 = {}".format(numpy.sum(np.load(npy_path))))
    f_w = nc.Dataset(nc_path, 'r', format='NETCDF4')  # 创建一个格式为.nc的
    grid = f_w.variables['Flash_pre'][:]
    print(np.sum(grid))