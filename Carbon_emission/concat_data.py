import argparse
import os

import numpy as np
import pandas as pd

from function.load_data import *


def concat_data(opt):
    if opt.file_type == 'json':
        y_true_load, y_pred_load, y_true_pv, y_pred_pv, y_true_wind, y_pred_wind, _, _ = load(opt.path) if opt.path else load()
    elif opt.file_type == 'npy':
        y_true_load, y_pred_load, y_true_pv, y_pred_pv, y_true_wind, y_pred_wind = load_npy(opt.true_data, opt.pred_data)
    elif opt.file_type == 'dg':
        y_true_load, y_pred_load = load_dg(opt.path_dg_true, opt.path_dg_pred)
    elif opt.file_type == 'urbangpt':
        y_true_load, y_pred_load, y_true_pv_wind, y_pred_pv_wind = load_urbangpt_npy(opt.path_urbangpt_true, opt.path_urbangpt_pred)

    if opt.pred_or_true == 'pred':
        if opt.file_type == 'json' or opt.file_type == 'npy':
            y_pred_load = np.array(y_pred_load[opt.index]).reshape(-1, 1177)
            y_pred_pv = np.array(y_pred_pv[opt.index]).reshape(-1, 1177)
            y_pred_wind = np.array(y_pred_wind[opt.index]).reshape(-1, 1177)

            min_pv = np.min(y_pred_pv)
            y_pred_pv[y_pred_pv < abs(min_pv) + 1e-4] = 0

            # y_pred_wind[y_pred_wind <= 98] = 0

            y_pred_wind = np.where(y_pred_wind >= np.partition(y_pred_wind, -2, axis=1)[:, -2][:, None], y_pred_wind, 0)

            for i in range(12):
                df = pd.DataFrame({
                    "y_pred_load": y_pred_load[i],
                    "y_pred_pv": y_pred_pv[i],
                    "y_pred_wind": y_pred_wind[i]
                })
                df.to_csv(rf'..\..\result\concat_data\concat_data{i + 1}.csv', index=False)

        elif opt.file_type == 'dg':
            y_pred_load = np.array(y_pred_load[opt.index]).reshape(-1, 1177)
            for i in range(12):
                df = pd.DataFrame({
                    "y_pred_dg": y_pred_load[i]
                })
                df.to_csv(rf'..\..\result\concat_data\concat_data{i + 1}.csv', index=False)

        elif opt.file_type == 'urbangpt':
            y_pred_load = np.array(y_pred_load[opt.index]).reshape(-1, 1177)
            y_pred_pv_wind = np.array(y_pred_pv_wind[opt.index]).reshape(-1, 1177)
            min_pv_wind = np.min(y_pred_pv_wind)
            y_pred_pv_wind[y_pred_pv_wind < abs(min_pv_wind) + 1e-4] = 0
            for i in range(12):
                df = pd.DataFrame({
                    "y_pred_load": y_pred_load[i],
                    "y_pred_pv_wind": y_pred_pv_wind[i]
                })
                df.to_csv(rf'..\..\result\concat_data\concat_data{i + 1}.csv', index=False)

    elif opt.pred_or_true == 'true':
        if opt.file_type == 'json' or opt.file_type == 'npy':
            y_true_load = np.array(y_true_load[opt.index])
            y_true_pv = np.array(y_true_pv[opt.index])
            y_true_wind = np.array(y_true_wind[opt.index])

            for i in range(12):
                df = pd.DataFrame({
                    "y_true_load": y_true_load[i],
                    "y_true_pv": y_true_pv[i],
                    "y_true_wind": y_true_wind[i]
                })
                df.to_csv(rf'..\..\result\concat_data\concat_data{i + 1}.csv', index=False)

        elif opt.file_type == 'dg':
            y_true_load = np.array(y_true_load[opt.index]).reshape(-1, 1177)
            for i in range(12):
                df = pd.DataFrame({
                    "y_true_dg": y_true_load[i]
                })
                df.to_csv(rf'..\..\result\concat_data\concat_data{i + 1}.csv', index=False)

        elif opt.file_type == 'urbangpt':
            y_true_load = np.array(y_true_load[opt.index]).reshape(-1, 1177)
            y_true_pv_wind = np.array(y_true_pv_wind[opt.index]).reshape(-1, 1177)
            for i in range(12):
                df = pd.DataFrame({
                    "y_true_load": y_true_load[i],
                    "y_true_pv_wind": y_true_pv_wind[i]
                })
                df.to_csv(rf'..\..\result\concat_data\concat_data{i + 1}.csv', index=False)
