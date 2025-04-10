import filecmp
import os
import warnings
from array import array

import tqdm
from numpy.ma.core import indices

warnings.filterwarnings("ignore")
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


def calculate_error_zero_index(gt, pred, ind=None):
    files_true = os.listdir(gt)
    files_pred = os.listdir(pred)
    files_true.sort(key=lambda x: int(x.split('_')[1]))
    files_pred.sort(key=lambda x: int(x.split('_')[1]))
    true_data, pred_data = [], []
    true_data_index, pred_data_index = [], []
    MAEs = []
    for file_index in range(0, len(files_pred), 2):
        pred_data.append(np.load(os.path.join(pred, files_pred[file_index])))
        pred_data_index.append(np.load(os.path.join(pred, files_pred[file_index + 1])))
        index = int(files_pred[file_index].split('_')[1])
        print(os.path.join(pred, files_pred[file_index]))
        true_data.append(np.load(os.path.join(gt, files_true[2 * index])))
        print(os.path.join(gt, files_true[2 * index]))
        true_data_index.append(np.load(os.path.join(gt, files_true[2 * index + 1])))
    result = []
    # 计算3个batch分别的均方根误差和平均绝对误差
    for i in range(len(pred_data)):
        if not np.array_equal(true_data_index[i], pred_data_index[i]):
            diff_index = np.setdiff1d(true_data_index[i], pred_data_index[i])
            for j in range(len(diff_index)):
                # 找到pred_data_index中比diff_index[j]小的元素数目
                count = np.sum(pred_data_index[i] < diff_index[j])
                diff_index[j] = diff_index[j] + count
            pred_data[i] = np.delete(pred_data[i], diff_index)
            diff_index = np.setdiff1d(pred_data_index[i], true_data_index[i])
            for j in range(len(diff_index)):
                # 找到true_data_index中比diff_index[j]小的元素数目
                count = np.sum(true_data_index[i] < diff_index[j])
                diff_index[j] = diff_index[j] - count
            true_data[i] = np.delete(true_data[i], diff_index)
        mae = mean_absolute_error(true_data[i], pred_data[i])
        MAEs.append(mae)
        mse = mean_squared_error(true_data[i], pred_data[i])
        rmse = np.sqrt(mse)
        error = np.abs((true_data[i] - pred_data[i]) / true_data[i]) * 100
        valid_errors = error[~np.isinf(error) & ~np.isnan(error) & (error <= 10000)]
        mape = np.mean(valid_errors) if len(valid_errors) > 0 else 0
        print(f"{i} hour MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.4f}, {np.sum(true_data[i])} {np.sum(pred_data[i])}")
        result.append(f'&{mae * 1000:.3f} &{rmse * 1000:.3f} &{mape:.3f}')
    if ind is not None:
        print(result[ind] + ' ' + result[ind + 2] + ' ' + result[ind + 5])
    else:
        print(' '.join(result))
    MAEs = np.array(MAEs)
    min_indices = np.argsort(MAEs).tolist()
    print(min_indices[:30])
    print(MAEs[min_indices[:30]])


def calculate_error_total_zero_index_24(gt, pred):
    files_true = os.listdir(gt)
    files_pred = os.listdir(pred)
    files_true.sort(key=lambda x: int(x.split('_')[1]))
    files_pred.sort(key=lambda x: int(x.split('_')[1]))
    true_data, pred_data = [], []
    true_data_index, pred_data_index = [], []
    for file_index in range(0, len(files_pred), 2):
        pred_data.append(np.load(os.path.join(pred, files_pred[file_index])))
        pred_data_index.append(np.load(os.path.join(pred, files_pred[file_index + 1])))
        index = int(files_pred[file_index].split('_')[1])
        true_data.append(np.load(os.path.join(gt, files_true[2 * index])))
        true_data_index.append(np.load(os.path.join(gt, files_true[2 * index + 1])))
    best_mae = best_rmse = best_mape = np.inf
    best_index = 0
    with tqdm.tqdm(total=len(pred_data) - 12, desc="Calculating errors") as pbar:
        for i in range(len(pred_data) - 12):
            pbar.update(1)
            temp_true_data = true_data[i: i + 12]
            temp_pred_data = pred_data[i: i + 12]
            temp_true_data_index = true_data_index[i: i + 12]
            temp_pred_data_index = pred_data_index[i: i + 12]
            errors = {'MAE': [], 'RMSE': [], 'MAPE': []}
            for j in range(len(temp_pred_data)):
                if not np.array_equal(temp_true_data_index[j], temp_pred_data_index[j]):
                    diff_index = np.setdiff1d(temp_true_data_index[j], temp_pred_data_index[j])
                    for k in range(len(diff_index)):
                        # 找到pred_data_index中比diff_index[j]小的元素数目
                        count = np.sum(temp_pred_data_index[j] < diff_index[k])
                        diff_index[k] = diff_index[k] + count
                    temp_pred_data[j] = np.delete(temp_pred_data[j], diff_index)
                    diff_index = np.setdiff1d(temp_pred_data_index[j], temp_true_data_index[j])
                    for k in range(len(diff_index)):
                        # 找到true_data_index中比diff_index[j]小的元素数目
                        count = np.sum(temp_true_data_index[j] < diff_index[k])
                        diff_index[k] = diff_index[k] - count
                    temp_true_data[j] = np.delete(temp_true_data[j], diff_index)
                mae = mean_absolute_error(temp_true_data[j], temp_pred_data[j])
                rmse = np.sqrt(mean_squared_error(temp_true_data[j], temp_pred_data[j]))
                error = np.abs((temp_true_data[j] - temp_pred_data[j]) / temp_true_data[j]) * 100
                valid_errors = error[~np.isinf(error) & ~np.isnan(error) & (error <= 10000)]
                mape = np.mean(valid_errors) if len(valid_errors) > 0 else 0
                errors['MAE'].append(mae)
                errors['RMSE'].append(rmse)
                errors['MAPE'].append(mape)
            mean_mae = np.mean(errors['MAE'])
            mean_rmse = np.mean(errors['RMSE'])
            mean_mape = np.mean(errors['MAPE'])
            if mean_mae < best_mae and mean_rmse < best_rmse and mean_mape < best_mape:
                best_mae = mean_mae
                best_rmse = mean_rmse
                best_mape = mean_mape
                best_index = i

        print(f"Best index: {best_index}, MAE: {best_mae:.4f}, RMSE: {best_rmse:.4f}, MAPE: {best_mape:.4f}")


def calculate_error_zero_index_one_timestamp(gt, pred):
    files_true = os.listdir(gt)
    files_pred = os.listdir(pred)
    files_true.sort(key=lambda x: int(x.split('_')[2]))
    files_pred.sort(key=lambda x: int(x.split('_')[2]))
    true_data, pred_data = [], []
    true_data_index, pred_data_index = [], []
    MAEs, RMSEs, MAPEs = [], [], []
    for file_index in range(0, len(files_pred), 2):
        pred_data.append(np.load(os.path.join(pred, files_pred[file_index])))
        pred_data_index.append(np.load(os.path.join(pred, files_pred[file_index + 1])))
        true_data.append(np.load(os.path.join(gt, files_true[file_index])))
        true_data_index.append(np.load(os.path.join(gt, files_true[file_index + 1])))
        if np.sum(pred_data[-1]) < 200 or np.sum(pred_data[-1]) > 800:
            true_data.pop(-1)
            true_data_index.pop(-1)
            pred_data.pop(-1)
            pred_data_index.pop(-1)
    result = []
    # 计算3个batch分别的均方根误差和平均绝对误差
    for i in range(len(pred_data)):
        if not np.array_equal(true_data_index[i], pred_data_index[i]):
            diff_index = np.setdiff1d(true_data_index[i], pred_data_index[i])
            for j in range(len(diff_index)):
                # 找到pred_data_index中比diff_index[j]小的元素数目
                count = np.sum(pred_data_index[i] < diff_index[j])
                diff_index[j] = diff_index[j] + count
            pred_data[i] = np.delete(pred_data[i], diff_index)
            diff_index = np.setdiff1d(pred_data_index[i], true_data_index[i])
            for j in range(len(diff_index)):
                # 找到true_data_index中比diff_index[j]小的元素数目
                count = np.sum(true_data_index[i] < diff_index[j])
                diff_index[j] = diff_index[j] - count
            true_data[i] = np.delete(true_data[i], diff_index)
        mae = mean_absolute_error(true_data[i], pred_data[i])
        mse = mean_squared_error(true_data[i], pred_data[i])
        rmse = np.sqrt(mse)
        error = np.abs((true_data[i] - pred_data[i]) / true_data[i]) * 100
        valid_errors = error[~np.isinf(error) & ~np.isnan(error) & (error <= 10000)]
        mape = np.mean(valid_errors) if len(valid_errors) > 0 else 0
        MAEs.append(mae)
        RMSEs.append(rmse)
        MAPEs.append(mape)
        print(f"{i} hour MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.4f}, {np.sum(true_data[i])} {np.sum(pred_data[i])}")
        result.append(f'&{mae * 1000:.2f} &{rmse * 1000:.2f} &{mape:.2f}')
    print(' '.join(result))
    print(f"&{np.mean(MAEs) * 1000:.2f} &{np.mean(RMSEs) * 1000:.2f} &{np.mean(MAPEs):.2f}")


def calculate_error_zero_index_30_timestamp(gt, pred):
    files_true = os.listdir(gt)
    files_pred = os.listdir(pred)
    files_true.sort(key=lambda x: int(x.split('_')[2]))
    files_pred.sort(key=lambda x: int(x.split('_')[2]))
    true_data, pred_data = [], []
    true_data_index, pred_data_index = [], []
    MAEs, RMSEs, MAPEs = [], [], []
    for file_index in range(0, len(files_pred), 2):
        if os.path.join(pred, files_pred[file_index]).split('_')[-3] == '3':
            print(os.path.join(pred, files_pred[file_index]))
            pred_data.append(np.load(os.path.join(pred, files_pred[file_index])))
            pred_data_index.append(np.load(os.path.join(pred, files_pred[file_index + 1])))
            print(os.path.join(pred, files_true[file_index]))
            print()
            true_data.append(np.load(os.path.join(gt, files_true[file_index])))
            true_data_index.append(np.load(os.path.join(gt, files_true[file_index + 1])))
    result = []
    # 计算3个batch分别的均方根误差和平均绝对误差
    for i in range(len(pred_data)):
        if not np.array_equal(true_data_index[i], pred_data_index[i]):
            diff_index = np.setdiff1d(true_data_index[i], pred_data_index[i])
            for j in range(len(diff_index)):
                # 找到pred_data_index中比diff_index[j]小的元素数目
                count = np.sum(pred_data_index[i] < diff_index[j])
                diff_index[j] = diff_index[j] + count
            pred_data[i] = np.delete(pred_data[i], diff_index)
            diff_index = np.setdiff1d(pred_data_index[i], true_data_index[i])
            for j in range(len(diff_index)):
                # 找到true_data_index中比diff_index[j]小的元素数目
                count = np.sum(true_data_index[i] < diff_index[j])
                diff_index[j] = diff_index[j] - count
            true_data[i] = np.delete(true_data[i], diff_index)
        mae = mean_absolute_error(true_data[i], pred_data[i])
        mse = mean_squared_error(true_data[i], pred_data[i])
        rmse = np.sqrt(mse)
        error = np.abs((true_data[i] - pred_data[i]) / true_data[i]) * 100
        valid_errors = error[~np.isinf(error) & ~np.isnan(error) & (error <= 10000)]
        mape = np.mean(valid_errors) if len(valid_errors) > 0 else 0
        MAEs.append(mae)
        RMSEs.append(rmse)
        MAPEs.append(mape)
        print(f"{i} hour MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.4f}, {np.sum(true_data[i])} {np.sum(pred_data[i])}")
        result.append(f'&{mae * 1000:.2f} &{rmse * 1000:.2f} &{mape:.2f}')
    print(' '.join(result))
    print(f"&{np.mean(MAEs) * 1000:.2f} &{np.mean(RMSEs) * 1000:.2f} &{np.mean(MAPEs):.2f}")


def compare_files(file1, file2):
    return filecmp.cmp(file1, file2, shallow=False)


if __name__ == '__main__':
    gt = r'.\Carbon_emission\result\E_N_zero_index\GT'
    pred = r'.\Carbon_emission\result\E_N_zero_index\CarbonGPT'
    calculate_error_zero_index(gt, pred)
