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
    next_time = start_time + timedelta(hours=time_step + 11)

    out_ = start_time + timedelta(hours=time_step + 12)
    out_next = start_time + timedelta(hours=time_step + 23)
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


print("Loading data from origin data")
list_data_dict = json.load(open('./multi_NYC.json', "r"))
first_id_value = list_data_dict[0]['id']
text_id = []

for i in range(num_nodes):
    modified_id_value = first_id_value.replace('NYCmulti', 'Color')
    parts = modified_id_value.split('_')
    parts[3] = str(i)
    parts[4] = str(i + 1)

    modified_id_value = '_'.join(parts)

    text_id.append(modified_id_value)

final_text_id = []
for j in range(train_data.shape[0] - 12 - 12 + 1):
    for k in range(num_nodes):
        parts = text_id[k].split('_')
        parts[-1] = str(j)
        modified_id_value = '_'.join(parts)
        final_text_id.append(modified_id_value)

conversations = json.load(open('./human_carbon.json', "r"))
first_conversations = conversations['value']
text_conversations = []
pattern = r'\[(.*?)\]'
matches = re.findall(pattern, first_conversations)
final_text_conversations = []

gpt = json.load(open('./gpt_data.json', "r"))

for h in range(train_data.shape[0] - 12 - 12 + 1):
    for i in range(train_data.shape[1]):

        current_conversation = first_conversations
        parts = re.findall(pattern, first_conversations)
        load_ = train_data[h:h + 12, i, 0].tolist()
        pv_ = train_data[h:h + 12, i, 1].tolist()
        wind_ = train_data[h:h + 12, i, 2].tolist()
        replaced_parts = parts.copy()
        for j in range(len(parts)):
            numbers_load = parts[0].split()
            numbers_pv = parts[1].split()
            numbers_wind = parts[2].split()
            for k in range(len(numbers_load)):
                numbers_load[k] = "%.3f" % load_[k]
                numbers_pv[k] = "%.3f" % pv_[k]
                numbers_wind[k] = "%.3f" % wind_[k]
            replaced_parts[0] = ' '.join(numbers_load)
            replaced_parts[1] = ' '.join(numbers_pv)
            replaced_parts[2] = ' '.join(numbers_wind)
            current_conversation = re.sub(matches[j], replaced_parts[j], current_conversation, 1)

        input_time, pre_time = get_timestamp(h)

        match_ = re.findall(r"'(.*?)'", current_conversation)
        cleaned_match = [s.split(', with')[0] for s in match_]

        cleaned_match[0] = input_time

        cleaned_match[1] = pre_time

        for j in range(len(match_)):
            current_conversation = current_conversation.replace(match_[j], cleaned_match[j])

        text_conversations.append(current_conversation)

human = 'human'
dict_list = [{"from": human, "value": s} for s in text_conversations]
out_list = [[y, gpt] for y in dict_list]
result_list = [{"id": f, "conversations": e} for f, e in zip(final_text_id, out_list)]

csv_data = pd.read_csv('coupling_node_1177.csv', header=None)
node_ids = csv_data.iloc[:, 0].tolist()

result_list_by_id = {item['id']: item for item in result_list}
extend_text = ", which includes both regular load and the charging load from electric vehicle charging stations. The recorded photovoltaic generation"

for node_id in node_ids:
    prefix = f"train_Color_region_{node_id}_{node_id+1}_len_"
    for i in range(len(result_list)):
        item_id = f"{prefix}{i}"
        if item_id in result_list_by_id:
            to_conversations = result_list_by_id[item_id]['conversations'][0]['value']
            conversations_str = to_conversations.replace(", the recorded photovoltaic generation", extend_text)

            result_list_by_id[item_id]['conversations'][0]['value'] = conversations_str
    print(node_id)
print('end')


with open('./train_10pv_withUTN.json', 'w') as f:
    json.dump(list(result_list_by_id.values()), f)

print()