import math
import argparse
import os

import numpy as np

from function.load_data import *
from concat_data import concat_data


def strfill(src, lg, str1):
    n = math.ceil((lg - len(src)) / len(str1))
    newstr = src + str1 * n
    return newstr[0:lg]


def generate_opendss_files(opt):
    if opt.file_type == 'json':
        y_true_load, y_pred_load, y_true_pv, y_pred_pv, y_true_wind, y_pred_wind, _, _ = load(opt.path) if opt.path else load()
    elif opt.file_type == 'npy':
        y_true_load, y_pred_load, y_true_pv, y_pred_pv, y_true_wind, y_pred_wind = load_npy(opt.true_data, opt.pred_data)
    elif opt.file_type == 'dg':
        y_true_load, y_pred_load = load_dg(opt.path_dg_true, opt.path_dg_pred)
    elif opt.file_type == 'urbangpt':
        y_true_load, y_pred_load, y_true_pv_wind, y_pred_pv_wind = load_urbangpt_npy(opt.path_urbangpt_true, opt.path_urbangpt_pred)
    power_load = np.array(y_true_load[opt.index] if opt.pred_or_true == 'true' else y_pred_load[opt.index]).reshape(-1, 1177)
    power_load = np.round(np.array(power_load), 2)
    power_load[power_load >= 1000] = np.round(power_load[power_load >= 1000], 1)

    with open(r'..\..\data\template\Loads.dss', 'r') as f:
        str_temp = f.read().split("\n")[:-1]
    head_str = str_temp[:12]
    str_temp = str_temp[12:1189]

    for col, col_idx in enumerate(np.arange(power_load.shape[0])):
        with open(rf'..\..\result\Load\Loads_test{col + 1}.dss', 'w+') as f:
            for str_ in head_str:
                f.write(str_ + "\n")
            for i in range(len(str_temp)):
                str_ = str_temp[i]
                v_ = power_load[col, i]
                str_ = str_[:98] + str(v_).ljust(7, " ") + str_[98 + 7:]
                f.write(str_ + "\n")
