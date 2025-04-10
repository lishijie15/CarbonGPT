import json
import os
import re

import numpy as np


def extract_numbers(filename):
    return list(map(int, re.findall(r'\d+', filename)))


def load(folder_path=r'C:\Users\HP\Desktop\Python\Carbon_emission\data\JsonData'):
    y_pred_load = []
    y_true_load = []
    y_pred_pv = []
    y_true_pv = []
    y_pred_wind = []
    y_true_wind = []
    y_pred_net_load = []
    y_true_net_load = []

    y_true_load_regionlist = []
    y_pred_load_regionlist = []
    y_true_pv_regionlist = []
    y_pred_pv_regionlist = []
    y_pred_wind_regionlist = []
    y_true_wind_regionlist = []
    y_true_net_load_regionlist = []
    y_pred_net_load_regionlist = []

    index_all = 0

    # Retrieve all JSON files from a folder and sort them by filename
    file_list = sorted([filename for filename in os.listdir(folder_path) if filename.endswith(".json")], key=extract_numbers)

    for idx, filename in enumerate(file_list):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, "r") as file:
            data_t = json.load(file)

        for i in range(len(data_t)):
            i_data = data_t[i]
            y_load = np.array(i_data["y_load"])
            y_pv = np.array(i_data["y_pv"])
            y_wind = np.array(i_data["y_wind"])
            st_pre_load = np.array(i_data["st_pre_load"])
            st_pre_pv = np.array(i_data["st_pre_pv"])
            st_pre_wind = np.array(i_data["st_pre_wind"])
            y_net_load = y_load - y_pv
            st_pre_net_load = st_pre_load - st_pre_pv

            i4data_all = int(data_t[i]["id"].split('_')[6])
            if index_all != i4data_all:
                y_true_load_region = np.stack(y_true_load, axis=-1)
                y_pred_load_region = np.stack(y_pred_load, axis=-1)
                y_true_pv_region = np.stack(y_true_pv, axis=-1)
                y_pred_pv_region = np.stack(y_pred_pv, axis=-1)
                y_true_wind_region = np.stack(y_true_wind, axis=-1)
                y_pred_wind_region = np.stack(y_pred_wind, axis=-1)
                y_true_net_load_region = np.stack(y_true_net_load, axis=-1)
                y_pred_net_load_region = np.stack(y_pred_net_load, axis=-1)

                y_true_load_regionlist.append(y_true_load_region)
                y_pred_load_regionlist.append(y_pred_load_region)
                y_true_pv_regionlist.append(y_true_pv_region)
                y_pred_pv_regionlist.append(y_pred_pv_region)
                y_true_wind_regionlist.append(y_true_wind_region)
                y_pred_wind_regionlist.append(y_pred_wind_region)
                y_true_net_load_regionlist.append(y_true_net_load_region)
                y_pred_net_load_regionlist.append(y_pred_net_load_region)

                y_pred_load = []
                y_true_load = []
                y_pred_pv = []
                y_true_pv = []
                y_pred_wind = []
                y_true_wind = []
                y_pred_net_load = []
                y_true_net_load = []
                index_all = i4data_all
            y_true_load.append(y_load)
            y_pred_load.append(st_pre_load)
            y_true_pv.append(y_pv)
            y_pred_pv.append(st_pre_pv)
            y_true_wind.append(y_wind)
            y_pred_wind.append(st_pre_wind)
            y_true_net_load.append(y_net_load)
            y_pred_net_load.append(st_pre_net_load)

            if i == len(data_t) - 1 and idx == len(file_list) - 1:
                y_true_load_region = np.stack(y_true_load, axis=-1)
                y_pred_load_region = np.stack(y_pred_load, axis=-1)
                y_true_pv_region = np.stack(y_true_pv, axis=-1)
                y_pred_pv_region = np.stack(y_pred_pv, axis=-1)
                y_true_wind_region = np.stack(y_true_wind, axis=-1)
                y_pred_wind_region = np.stack(y_pred_wind, axis=-1)
                y_true_net_load_region = np.stack(y_true_net_load, axis=-1)
                y_pred_net_load_region = np.stack(y_pred_net_load, axis=-1)
                y_true_load_regionlist.append(y_true_load_region)
                y_pred_load_regionlist.append(y_pred_load_region)
                y_true_pv_regionlist.append(y_true_pv_region)
                y_pred_pv_regionlist.append(y_pred_pv_region)
                y_true_wind_regionlist.append(y_true_wind_region)
                y_pred_wind_regionlist.append(y_pred_wind_region)
                y_true_net_load_regionlist.append(y_true_net_load_region)
                y_pred_net_load_regionlist.append(y_pred_net_load_region)
                y_pred_load = []
                y_true_load = []
                y_pred_pv = []
                y_true_pv = []
                y_pred_wind = []
                y_true_wind = []
                y_pred_net_load = []
                y_true_net_load = []

    y_true_load = np.stack(y_true_load_regionlist, axis=0)
    y_pred_load = np.stack(y_pred_load_regionlist, axis=0)
    y_true_pv = np.stack(y_true_pv_regionlist, axis=0)
    y_pred_pv = np.stack(y_pred_pv_regionlist, axis=0)
    y_true_wind = np.stack(y_true_wind_regionlist, axis=0)
    y_pred_wind = np.stack(y_pred_wind_regionlist, axis=0)
    y_true_net_load = np.stack(y_true_net_load_regionlist, axis=0)
    y_pred_net_load = np.stack(y_pred_net_load_regionlist, axis=0)
    y_pred_load, y_pred_pv, y_pred_wind = np.abs(y_pred_load), np.abs(y_pred_pv), np.abs(y_pred_wind)
    return y_true_load, y_pred_load, y_true_pv, y_pred_pv, y_true_wind, y_pred_wind, y_true_net_load, y_pred_net_load  # (1, 12, 1177)


def load_npy(true_data, pred_data):
    name = pred_data.split('\\')[-1].split('_')[0]
    true_data = np.load(true_data)
    pred_data = np.load(pred_data)
    if name == 'MPGTN':
        true_data = true_data[10:]
        pred_data = pred_data[10:]
    return true_data[..., 0], pred_data[..., 0], true_data[..., 1], pred_data[..., 1], true_data[..., 2], pred_data[..., 2]


def load_dg(true_data, pred_data):
    true_data = np.load(true_data).squeeze(-1)
    pred_data = np.load(pred_data).squeeze(-1)
    return true_data, pred_data


def load_urbangpt_npy(true_data, pred_data):
    true_data = np.load(true_data)
    pred_data = np.load(pred_data)
    return true_data[..., 0], pred_data[..., 0], true_data[..., 1], pred_data[..., 1]


def load_urbangpt(folder_path=r'C:\Users\HP\Desktop\Python\Carbon_emission\data\JsonData\UrbanGPT'):
    y_pred_load = []
    y_true_load = []
    y_pred_pv_wind = []
    y_true_pv_wind = []
    y_pred_net_load = []
    y_true_net_load = []

    y_true_load_regionlist = []
    y_pred_load_regionlist = []
    y_true_pv_wind_regionlist = []
    y_pred_pv_wind_regionlist = []
    y_true_net_load_regionlist = []
    y_pred_net_load_regionlist = []

    index_all = 0

    # Retrieve all JSON files from a folder and sort them by filename
    file_list = sorted([filename for filename in os.listdir(folder_path) if filename.endswith(".json")], key=extract_numbers)

    for idx, filename in enumerate(file_list):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, "r") as file:
            data_t = json.load(file)

        for i in range(len(data_t)):
            i_data = data_t[i]
            y_load = np.array(i_data["y_in"])
            y_pv_wind = np.array(i_data["y_out"])
            st_pre_load = np.array(i_data["st_pre_infolow"])
            st_pre_pv_wind = np.array(i_data["st_pre_outfolow"])
            y_net_load = y_load - y_pv_wind
            st_pre_net_load = st_pre_load - st_pre_pv_wind

            i4data_all = int(data_t[i]["id"].split('_')[6])
            if index_all != i4data_all:
                y_true_load_region = np.stack(y_true_load, axis=-1)
                y_pred_load_region = np.stack(y_pred_load, axis=-1)
                y_true_pv_wind_region = np.stack(y_true_pv_wind, axis=-1)
                y_pred_pv_wind_region = np.stack(y_pred_pv_wind, axis=-1)
                y_true_net_load_region = np.stack(y_true_net_load, axis=-1)
                y_pred_net_load_region = np.stack(y_pred_net_load, axis=-1)

                y_true_load_regionlist.append(y_true_load_region)
                y_pred_load_regionlist.append(y_pred_load_region)
                y_true_pv_wind_regionlist.append(y_true_pv_wind_region)
                y_pred_pv_wind_regionlist.append(y_pred_pv_wind_region)
                y_true_net_load_regionlist.append(y_true_net_load_region)
                y_pred_net_load_regionlist.append(y_pred_net_load_region)

                y_pred_load = []
                y_true_load = []
                y_pred_pv_wind = []
                y_true_pv_wind = []
                y_pred_net_load = []
                y_true_net_load = []
                index_all = i4data_all
            y_true_load.append(y_load)
            y_pred_load.append(st_pre_load)
            y_true_pv_wind.append(y_pv_wind)
            y_pred_pv_wind.append(st_pre_pv_wind)
            y_true_net_load.append(y_net_load)
            y_pred_net_load.append(st_pre_net_load)

            if i == len(data_t) - 1 and idx == len(file_list) - 1:
                y_true_load_region = np.stack(y_true_load, axis=-1)
                y_pred_load_region = np.stack(y_pred_load, axis=-1)
                y_true_pv_wind_region = np.stack(y_true_pv_wind, axis=-1)
                y_pred_pv_wind_region = np.stack(y_pred_pv_wind, axis=-1)
                y_true_net_load_region = np.stack(y_true_net_load, axis=-1)
                y_pred_net_load_region = np.stack(y_pred_net_load, axis=-1)
                y_true_load_regionlist.append(y_true_load_region)
                y_pred_load_regionlist.append(y_pred_load_region)
                y_true_pv_wind_regionlist.append(y_true_pv_wind_region)
                y_pred_pv_wind_regionlist.append(y_pred_pv_wind_region)
                y_true_net_load_regionlist.append(y_true_net_load_region)
                y_pred_net_load_regionlist.append(y_pred_net_load_region)
                y_pred_load = []
                y_true_load = []
                y_pred_pv_wind = []
                y_true_pv_wind = []
                y_pred_net_load = []
                y_true_net_load = []

    y_true_load = np.stack(y_true_load_regionlist, axis=0)
    y_pred_load = np.stack(y_pred_load_regionlist, axis=0)
    y_true_pv_wind = np.stack(y_true_pv_wind_regionlist, axis=0)
    y_pred_pv_wind = np.stack(y_pred_pv_wind_regionlist, axis=0)
    y_true_net_load = np.stack(y_true_net_load_regionlist, axis=0)
    y_pred_net_load = np.stack(y_pred_net_load_regionlist, axis=0)
    y_pred_load, y_pred_pv_wind = np.abs(y_pred_load), np.abs(y_pred_pv_wind)
    return y_true_load, y_pred_load, y_true_pv_wind, y_pred_pv_wind, y_true_net_load, y_pred_net_load  # (1, 12, 1177)


if __name__ == '__main__':
    y_true_load, y_pred_load, y_true_pv, y_pred_pv, y_true_wind, y_pred_wind, _, _ = load(
        r'C:\Users\HP\Desktop\Python\Carbon_emission\data\JsonData\Causal_pv10_new_lin_hid256_noCoA_addffn_noNorm_onlyRES_noPLUS_')
    np.save(r'C:\Users\HP\Desktop\Python\Carbon_emission\data\NpyData\CarbonGPTonlyRES_groundtruth.npy',
            np.stack((y_true_load, y_true_pv, y_true_wind), axis=-1))
    np.save(r'C:\Users\HP\Desktop\Python\Carbon_emission\data\NpyData\CarbonGPTonlyRES_prediction.npy',
            np.stack((y_pred_load, y_pred_pv, y_pred_wind), axis=-1))

    # y_true_load_urbangpt, y_pred_load_urbangpt, y_true_pv_wind_urbangpt, y_pred_pv_wind_urbangpt, y_true_net_load_urbangpt, _ = load_urbangpt()
    # np.save(r'C:\Users\HP\Desktop\Python\Carbon_emission\data\NpyData\UrbanGPT_groundtruth.npy',
    #         np.stack((y_true_load_urbangpt, y_true_pv_wind_urbangpt), axis=-1))
    # np.save(r'C:\Users\HP\Desktop\Python\Carbon_emission\data\NpyData\UrbanGPT_prediction.npy',
    #         np.stack((y_pred_load_urbangpt, y_pred_pv_wind_urbangpt), axis=-1))

    # y_true_load_urbangpt = np.sum(y_true_load_urbangpt, axis=2)
    # y_true_pv_wind_urbangpt = np.sum(y_true_pv_wind_urbangpt, axis=2)
    # y_true_load = np.sum(y_true_load, axis=2)
    # y_true_pv = np.sum(y_true_pv, axis=2)
    # y_true_wind = np.sum(y_true_wind, axis=2)
    # equal_load = equal(y_true_load, y_true_load_urbangpt)
    # equal_res = equal(y_true_pv + y_true_wind, y_true_pv_wind_urbangpt)
    # print出来不相等的元素
    # for i in range(len(equal_load[0])):
    #     for j in range(len(equal_load[0][i])):
    #         if not equal_load[0][i][j]:
    #             print(y_true_load[0][i][j], y_true_load_urbangpt[0][i][j])
    # for i in equal_res:
    #     if i.all():
    #         print("All elements are equal")

    # y_true_load_mpgtn, _, y_true_pv_mpgtn, y_pred_pv_mpgtn, y_true_wind_mpgtn, _ = load_npy(
    #     r'C:\Users\HP\Desktop\Python\Carbon_emission\data\npy\power_DG_MPGTN_20241018144454\MPGTN_groundtruth.npy',
    #     r'C:\Users\HP\Desktop\Python\Carbon_emission\data\npy\power_DG_MPGTN_20241018144454\MPGTN_prediction.npy')

    #
    # y_true_load_megacrn, _, y_true_pv_megacrn, _, y_true_wind_megacrn, _ = load_npy(
    #     r'C:\Users\HP\Desktop\Python\Carbon_emission\data\npy\power_DG_MegaCRN_20241008145010\MegaCRN_groundtruth.npy',
    #     r'C:\Users\HP\Desktop\Python\Carbon_emission\data\npy\power_DG_MegaCRN_20241008145010\MegaCRN_prediction.npy')
    # #
    # y_true_load_dg, y_pred_load_dg = load_dg(r'C:\Users\HP\Desktop\Python\Carbon_emission\data\NpyData\power_DG_true.npy',
    #                                          r'C:\Users\HP\Desktop\Python\Carbon_emission\data\npy\AGCRN\power_DG_pred.npy')
    #
    # y_true_load_tgcn, _, y_true_pv_tgcn, _, y_true_wind_tgcn, _ = load_npy(
    #     r'C:\Users\HP\Desktop\Python\Carbon_emission\data\NpyData\TGCN_groundtruth.npy',
    #     r'C:\Users\HP\Desktop\Python\Carbon_emission\data\NpyData\TGCN_prediction.npy')
    # #
    # equal_load = equal(y_true_load[98], y_true_load_tgcn[98])
    # for i in range(len(equal_load)):
    #     for j in range(len(equal_load[i])):
    #         if not equal_load[i][j]:
    #             print(y_true_load[98][i][j], y_true_load_tgcn[98][i][j])
    #
    # for i in equal_load:
    #     if i.all():
    #         print("All elements are equal")
