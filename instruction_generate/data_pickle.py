import json
import pickle
import re
import numpy as np
import pandas as pd
import argparse
from configparser import ConfigParser
from datetime import datetime, timedelta

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, choices=['power_DG'], default='power_DG', help='which dataset to run')
parser.add_argument('--month', type=str, default='power_DG_200712', help='which experiment setting (month) to run as testing data')
args = parser.parse_args()

config = ConfigParser()
config.read('params_DG.txt', encoding='UTF-8')
train_month = eval(config[args.month]['train_month'])
test_month = eval(config[args.month]['test_month'])
power_path = config[args.month]['power_path']
subroad_path = config[args.dataset]['subroad_path']
road_path = config['common']['road_path']
adj_path = config['common']['adj01_path']
num_nodes = len(np.loadtxt(subroad_path).astype(int))
N_link = config.getint('common', 'N_link')
args.num_nodes = num_nodes

def get_timestamp(time_step):
    start_time = datetime(2016, 10, 1, 0, 0)

    current_time = start_time + timedelta(hours=time_step)
    next_time = start_time + timedelta(hours=time_step + 24)

    out_ = start_time + timedelta(hours=time_step + 25)
    out_next = start_time + timedelta(hours=time_step + 30)

    input_time = current_time.strftime("%B %d, %Y, %H:%M, %A") + " to " + next_time.strftime("%B %d, %Y, %H:%M, %A")
    pre_time = out_.strftime("%B %d, %Y, %H:%M, %A") + " to " + out_next.strftime("%B %d, %Y, %H:%M, %A")

    return input_time, pre_time

def get_data(data_path, N_link, subdata_path, feature_list):
    data = pd.read_csv(data_path)[feature_list].values
    data = data.reshape(-1, N_link, data.shape[-1])
    data[data<0] = 0

    sub_idx = np.loadtxt(subdata_path).astype(int)
    data = data[:, sub_idx, :]
    return data

def get_time(data_path, N_link, subdata_path):
    df = pd.read_csv(data_path)[['timestamp']]
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['weekdaytime'] = df['timestamp'].dt.weekday * 144 + (df['timestamp'].dt.hour * 60 + df['timestamp'].dt.minute)//10
    df['weekdaytime'] = df['weekdaytime'] / df['weekdaytime'].max()
    data = df[['weekdaytime']].values
    data = data.reshape(-1, N_link, data.shape[-1])
    sub_idx = np.loadtxt(subdata_path).astype(int)
    data = data[:, sub_idx, :]
    return data

train_data = [get_data(config[month]['power_path'], N_link, subroad_path, ['power', 'pv', 'wind']) for month in train_month]
test_data = [get_data(config[month]['power_path'], N_link, subroad_path, ['power', 'pv', 'wind']) for month in test_month]
train_time = [get_time(config[month]['power_path'], N_link, subroad_path) for month in train_month]
test_time = [get_time(config[month]['power_path'], N_link, subroad_path) for month in test_month]


train_time_steps = 1464
test_time_steps = 744

total_data = np.concatenate((train_data[0], train_data[1], test_data[0]), axis=0)
train_data = total_data[:train_time_steps]
test_data = total_data[train_time_steps:]


def create_dataset(data, history_size, prediction_size):
    dataset = []
    for i in range(len(data) - history_size - prediction_size + 1):

        data_x = data[i:i + history_size]
        data_x = np.concatenate([data_x, np.ones((history_size, data.shape[1], 1))], axis=-1)
        data_x = np.expand_dims(data_x, axis=0)

        data_y = data[i + history_size:i + history_size + prediction_size]
        data_y = np.concatenate([data_y, np.ones((prediction_size, data.shape[1], 1))], axis=-1)
        data_y = np.expand_dims(data_y, axis=0)

        dataset.append({'data_x': data_x, 'data_y': data_y})

    return dataset

history_size = 12
prediction_size = 12

train = create_dataset(train_data, history_size, prediction_size)
test = create_dataset(test_data, history_size, prediction_size)

print(f'Train dataset size: {len(train)}')
print(f'Test dataset size: {len(test)}')
print(f'Example data_x shape: {train[0]["data_x"].shape}')
print(f'Example data_y shape: {train[0]["data_y"].shape}')

with open('train_pv10_withUTN.pkl', 'wb') as f:
    pickle.dump(train, f)

with open('test_pv10_withUTN.pkl', 'wb') as f:
    pickle.dump(test, f)

print("Data saved to train.pkl and test.pkl.")




