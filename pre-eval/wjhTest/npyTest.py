import os

import numpy
import numpy as np
import torch
import netCDF4 as nc

# 读取由wrf转成的npy格式的文件


if __name__ == '__main__':
    latlon_nc = nc.Dataset('latlon.nc')
    lat_ = latlon_nc.variables['lat'][:, :]
    lon_ = latlon_nc.variables['lon'][:, :]
    latlon_nc.close()
    mn = 159
    latlon = np.zeros(shape=[159 * 159, 2], dtype=float)
    latlon[:, 0] = lat_.reshape(mn * mn)
    latlon[:, 1] = lon_.reshape(mn * mn)

    lat_max = np.max(latlon[:, 0], keepdims=False)
    lat_min = np.min(latlon[:, 0], keepdims=False)
    lon_max = np.max(latlon[:, 1], keepdims=False)
    lon_min = np.min(latlon[:, 1], keepdims=False)

    print(lat_max)
    print(lat_min)
    print(lon_min)
    print(lon_max)
