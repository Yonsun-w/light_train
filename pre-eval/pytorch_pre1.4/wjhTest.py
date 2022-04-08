import os
import netCDF4 as nc



if __name__ == "__main__":
    pwd = os.getcwd()

    npc_path = 'example.nc'

    test_path = 'Equal_distance.nc'

    f = nc.Dataset('Equal_distance.nc', 'r+', format='NETCDF4')  # 创建一个格式为.nc的
    print(f)
    f.close()

    # print("-------")
    #
    # f = nc.Dataset(npc_path)
    # print(f)
    #
    # print(var_data.shape)
    #
    # print("---")
    #
    # print(f.variables['units'].type())


    # 最直接的办法，获取每个变量的缩写名字，标准名字(long_name),units和shape大小。这样很方便后续操作
    # all_vars_name = []
    # all_vars_lat = []
    # all_vars_units = []
    # all_vars_shape = []
    # #
    # for key in f.variables.keys():
    #     all_vars_name.append(key)
    #     all_vars_lat.append(f.variables[key])
    # #
    # #
    # print("-------all_vars_name-----")
    # print(all_vars_name)
    #
    # print("-------all_vars_lat-----")
    # print(all_vars_lat)
    # # print("-------all_vars_units-----")
    # print(all_vars_units)
    # print("-------all_vars_shape-----")
    # print(all_vars_shape)


