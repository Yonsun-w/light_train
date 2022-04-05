import os
import netCDF4 as nc



if __name__ == "__main__":
    pwd = os.getcwd()

    npc_path = 'BJ.latlon.nc'

    f = nc.Dataset(npc_path)



    all_vars = f.variables.keys()  # 获取所有变量名称  这里获取的是 odict_keys(['lat', 'lon'])

    all_vars_info = f.variables.items()  # 获取所有变量的属性 一大串

    print(all_vars_info)
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


