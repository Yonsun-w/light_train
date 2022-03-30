import os
import numpy as np
import pandas as pd
from config import read_config
from readFiles import EvalData
from scores import Cal_params_neighbor
#from pandas import datetime
import datetime


def checkData(time, path, pre_duration, time_step):
    is_predata = True
    is_obsdata = True
    is_predis_data = True
    for i in range(0, pre_duration // time_step):
        pre_path = os.path.join(path, 'pre', '{}_h{}.dat'.format(time.strftime('%Y%m%d%H%M'), i))
        if not os.path.exists(pre_path):
            is_predata = False
        predis_path = os.path.join(path, 'pre_dis', '{}_h{}.npy'.format(time.strftime('%Y%m%d%H%M'), i))
        if not os.path.exists(predis_path):

            is_predis_data = False
        obs_path = os.path.join(path, 'obs', 'RealDF{}_60.DAT'.format(time.strftime('%Y%m%d%H%M')))
        if not os.path.exists(obs_path):
            is_obsdata = False
        time += datetime.timedelta(minutes=time_step)
    print(is_predata,is_predis_data,is_obsdata)
    return is_predata and is_obsdata and is_predis_data


def main(config_dict):
    eval_results = pd.DataFrame(
        columns=['Time', 'Time Period', 'Threshold', 'Neighborhood Range'] + config_dict['EvaluationMethod'])
    path = config_dict['FilePath']
    pre_duration = config_dict['PreDuration']
    time_step = config_dict['TimeStep']

    st = datetime.datetime.strptime(config_dict['StartTime'], '%Y%m%d%H%M')
    et = datetime.datetime.strptime(config_dict['EndTime'], '%Y%m%d%H%M')
    all_available_time = []
    while st <= et:
        if checkData(st, path, pre_duration, time_step):
            all_available_time.append(st.strftime('%Y%m%d%H%M'))
        else:
            print('{} data is not complete.'.format(st.strftime('%Y%m%d%H%M')))
        st += datetime.timedelta(hours=1)

    for time in all_available_time:
        print(datetime.datetime.strptime(time, '%Y%m%d%H%M'))
        for ptl in config_dict['PreTimeLimit']:
            for threshold in config_dict['Threshold']:
                p = EvalData(path, time, ptl, time_step, threshold)
                for nbh in config_dict['NeighborhoodRange']:
                    # print('\nTime = {}  start&end = [{},{})  Threshold = {}  Neighborhood = {}'.
                    #       format(time, ptl[0], ptl[1], threshold, nbh))
                    cal = Cal_params_neighbor(neighbor_size=nbh)
                    scores = cal.cal_params_ones(p.obs_data, p.pre_data_dis)
                    results_dict = {}
                    results_dict['Time'] = datetime.datetime.strptime(time, '%Y%m%d%H%M')
                    results_dict['Time Period'] = ptl
                    results_dict['Threshold'] = threshold
                    results_dict['Neighborhood Range'] = nbh
                    for name in config_dict['EvaluationMethod']:
                        # print(name, scores[name])
                        results_dict[name] = scores[name]
                    eval_results = eval_results.append(results_dict, ignore_index=True)
    eval_results.to_csv('result.csv', index=False)


if __name__ == "__main__":
    config_dict = read_config()
    main(config_dict)
