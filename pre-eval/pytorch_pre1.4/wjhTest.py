import os
import netCDF4 as nc



if __name__ == "__main__":
    pwd = os.getcwd()

    npy_path = 'BJ.latlon.nc'

    f = nc.Dataset(npy_path)

    all_vars = f.variables.keys()  # 获取所有变量名称
    print(all_vars)

    all_vars_info = f.variables.items()  # 获取所有变量信息
    var = 'lat'

    ###最直接的办法，获取每个变量的缩写名字，标准名字(long_name),units和shape大小。这样很方便后续操作
    all_vars_name = []
    all_vars_long_name = []
    all_vars_units = []
    all_vars_shape = []

    print('f.variables[\'lat\']' ,f.variables['lat'])

    for key in f.variables.keys():
        print(key)
        all_vars_name.append(key)
        all_vars_long_name.append(f.variables[key].long_name)
        print(f.variables[key])
        all_vars_units.append(f.variables[key].units)
        all_vars_shape.append(f.variables[key].shape)


    print("-------all_vars_name-----")
    print(all_vars_name)
    print("-------all_vars_long_name-----")
    print(all_vars_long_name)
    print("-------all_vars_units-----")
    print(all_vars_units)
    print("-------all_vars_shape-----")
    print(all_vars_shape)


