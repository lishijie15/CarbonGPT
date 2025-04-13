import numpy as np
import pandas as pd

from concat_data import concat_data
from load_opendss import generate_opendss_files
from Carbon_new import calculate_carbon

file_path_transformer = r"..\..\data\template\LoadXfmrs.dss"
file_path_node_transform = r'..\..\data\template\1177nodes.csv'


def line_pre(file_path):
    with open(file_path, "r") as file:
        data = file.readlines()
    line_data_output = []
    line_name = []
    line_bus1 = []
    line_bus2 = []

    for line_number, line in enumerate(data):
        line_data = line.strip().split()
        print(line_data)
        if len(line_data) >= 4:
            if line_data[1].startswith("Transformer"):
                line_data[1] = line_data[1].replace("Load.", "")
                line_data[1] = line_data[1].upper()
                line_data[5] = line_data[5].replace("bus=", "").replace(".1", "").replace(".2", "").replace(".3", "")
                line_data[9] = line_data[9].replace("bus=", "").replace(".1.0", "")

                line_name.append(line_data[1])
                line_bus1.append(line_data[5])
                line_bus2.append(line_data[9])
                line_data_output.append((line_data[1], line_data[5], line_data[9]))
    return line_data_output


def data_loading(file_path):
    df1 = pd.read_csv(file_path, header=None)
    # power_list_time = df1.iloc[:, 0].tolist()
    power_list_node = df1.iloc[:, 1].tolist()
    # power_list_load = df1.iloc[:, 2].tolist()
    power_list_pv = df1.iloc[:, 3].tolist()
    power_list_wind = df1.iloc[:, 4].tolist()
    pv_out = []
    wind_out = []

    for data_number, data in enumerate(power_list_pv):
        if data != 0.5:
            pv_out.append((power_list_node[data_number], data))
    for data_number, data in enumerate(power_list_wind):
        if data != 0.5:
            wind_out.append((power_list_node[data_number], data))
    return pv_out, wind_out


def data_loading_1(file_path):
    df1 = pd.read_csv(file_path)
    df2 = pd.read_csv(r'..\..\data\template\1177nodes.csv', header=None)
    data_transform = df2.iloc[:, 0].tolist()
    pv_statistics = df1.iloc[:, 1].tolist()
    wind_statistics = df1.iloc[:, 2].tolist()
    pv_out = []
    wind_out = []

    for data_number, data in enumerate(pv_statistics):
        if data != 0:
            pv_out.append((data_transform[data_number], data))
    for data_number, data in enumerate(wind_statistics):
        if data != 0:
            wind_out.append((data_transform[data_number], data))
    return pv_out, wind_out


def data_loading_2(file_path):
    df1 = pd.read_csv(file_path)
    df2 = pd.read_csv(r'..\..\data\template\1177nodes.csv', header=None)
    data_transform = df2.iloc[:, 0].tolist()
    pv_wind_statistics = df1.iloc[:, 1].tolist()
    pv_wind_out = []

    for data_number, data in enumerate(pv_wind_statistics):
        if data != 0:
            pv_wind_out.append((data_transform[data_number], data))
    return pv_wind_out


def unique_list(l):
    unique_third_values = set()
    # Initialize an empty list to store the filtered output
    filtered_line_data_output = []

    # Iterate through the original list
    for item in l:
        # Check if the third value is already in the set
        if item[2] not in unique_third_values:
            # If not, add it to the set and the filtered list
            unique_third_values.add(item[2])
            filtered_line_data_output.append(item)
        else:
            item = list(item)
            item[2] = item[2] + "1"
            item = tuple(item)
            filtered_line_data_output.append(item)
    return filtered_line_data_output


def load_pre(file_path):
    with open(file_path, "r") as file:
        data = file.readlines()
    line_data_output = []

    for line_number, line in enumerate(data):
        line_data = line.strip().split()
        if len(line_data) >= 4:
            if line_data[1].startswith("Load."):
                line_data[1] = line_data[1].replace("Load.", "")
                line_data[1] = line_data[1].upper()
                line_data[3] = (line_data[3].replace("Bus1=SX", "L").replace("bus1=SX", "L")
                                .replace(".1.2", "").replace("A", "").replace("B", "").replace("C", ""))

                line_data[8] = line_data[8].replace("kW=", "")

                line_data_output.append((line_number - 12, line_data[1], line_data[3], line_data[8]))
    # line_data_output = unique_list(line_data_output)
    return line_data_output


def data_match(data0, data_pv_wind, node_transform_path):
    data_power = data_pv_wind
    df1 = pd.read_csv(node_transform_path, header=None)
    data_transform = df1.iloc[:, 0].tolist()
    data_mid = data0
    for k in range(len(data_transform)):
        data_mid[k] = (data0[k][0], data0[k][1], data0[k][2], data0[k][3], 0)

    for j in range(len(data_power)):
        for i in range(len(data_transform)):
            if data_power[j][0] == data_transform[i]:
                if data_mid[i][4] != 0:
                    data_mid[i] = (data0[i][0], data0[i][1], data0[i][2], data0[i][3], data_mid[i][4] + data_power[j][1])
                else:
                    data_mid[i] = (data0[i][0], data0[i][1], data0[i][2], data0[i][3], data_power[j][1])
                break
    data_pv_wind = []
    data_mid.append((1177, '213123123', 'm1125934', '0', 1000))
    pv_df = pd.read_csv(r'..\..\data\template\119.csv', header=None)
    pv_wind_index = pv_df.iloc[:, 0].tolist()
    # for m in range(len(data_mid)):
    #     if data_mid[m][4] != 0:
    #         data_pv_wind.append((data_mid[m][0], data_mid[m][2], data_mid[m][4]))
    for j in pv_wind_index:
        if data_mid[j][4] != 0:
            data_pv_wind.append((data_mid[j][0], data_mid[j][2], data_mid[j][4]))
        else:
            data_pv_wind.append((data_mid[j][0], data_mid[j][2], np.finfo(float).eps))
    return data_mid, data_pv_wind


def dss_file_make():
    return 0


def generateOpendss(opt):
    import tqdm

    for i in tqdm.tqdm(range(1, 13)):
        file_path_load = rf'{opt.file_path_load}' + f'Loads_test{i}.dss'
        file_path_max_data_loading = rf'{opt.file_path_max_data_loading}' + f'concat_data{i}.csv'
        file_path_output_dss = rf'{opt.file_path_output_dss}' + rf'Generators_{i}.dss'

        # lv_data = line_pre(file_path_transformer)
        load_data = load_pre(file_path_load)

        if opt.file_type == 'json' or opt.file_type == 'npy':
            pv_data, wind_data = data_loading_1(file_path_max_data_loading)
            final_data, final_pv_wind_data = data_match(load_data, pv_data + wind_data, file_path_node_transform)
        elif opt.file_type == 'urbangpt':
            pv_wind_data = data_loading_2(file_path_max_data_loading)
            final_data, final_pv_wind_data = data_match(load_data, pv_wind_data, file_path_node_transform)
        elif opt.file_type == 'dg':
            pv_wind_data = []
            final_data, final_pv_wind_data = data_match(load_data, pv_wind_data, file_path_node_transform)

        bus_need = len(final_pv_wind_data)

        # if bus_need == 153:
        #     final_pv_wind_data = [item for item in final_pv_wind_data if item[2] > 0.05]
        #     df = pd.DataFrame(final_pv_wind_data)
        #     df.to_csv(r'.\119.csv', index=False, header=None)

        with open(file_path_output_dss, 'w') as outfile:
            if bus_need < 2:
                outfile.write('// 12.47/480V Transformer and neutral reactor definition.\n')
                outfile.write('\n\n\n')
                outfile.write('// Generator definition\n')
                outfile.write('\n')
            else:
                for i in range(bus_need):
                    if i < 2:
                        outfile.write('// 12.47/480V Transformer and neutral reactor definition.\n')
                    outfile.write(
                        f'New "Transformer.G_{final_pv_wind_data[i][1]}"  XHL=5.75  kVA={final_pv_wind_data[i][2]}  Conns=[wye, Delta]\n')
                    outfile.write(f'~ wdg=1 bus={final_pv_wind_data[i][1]}.1.2.3.4  kV=12.47\n')
                    outfile.write(f'~ wdg=2 bus=G_{final_pv_wind_data[i][1]}        kV=0.48\n')
                    outfile.write(
                        f'New Reactor.G_{final_pv_wind_data[i][1]} Phases = 1 Bus1 = {final_pv_wind_data[i][1]}.4 R=0.001 X=0  !Neutral Reactor/Resistor\n')
                    outfile.write('\n')

                outfile.write('\n')
                outfile.write('\n')

                for i in range(bus_need):
                    if i < 2:
                        outfile.write('// Generator definition\n')
                    outfile.write(
                        f'New   "Generator.G_{final_pv_wind_data[i][1]}"  Bus1=G_{final_pv_wind_data[i][1]}  kW={final_pv_wind_data[i][2]}  PF=1  kVA={final_pv_wind_data[i][2]}  kV=0.48  Xdp=0.27  Xdpp=0.2  H=2\n')
                    outfile.write('~ Conn=Delta   ! use the interconnection transformer to achieve wye for direct connect\n')
                    outfile.write('\n')


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Concatenate data')
    parser.add_argument('--file_type', type=str, default='npy', help='type of the file')
    parser.add_argument('--pred_or_true', type=str, default='pred', help='pred or true')
    parser.add_argument('--index', type=int, default=0, help='index of the data')
    parser.add_argument('--file_path_load', type=str, default=r'..\..\result\Load\\', help='path to the data')
    parser.add_argument('--file_path_max_data_loading', type=str, default=r'..\..\result\concat_data\\')
    parser.add_argument('--file_path_output_dss', type=str, default=r'..\..\result/Generator/',
                        help='path to output the dss file')
    parser.add_argument('--path', type=str, default=r'..\..\data\JsonData\CarbonGPT', help='path to the data')
    parser.add_argument('--true_data', type=str, default=r'..\..\data\NpyData\CarbonGPT_groundtruth.npy')
    parser.add_argument('--pred_data', type=str, default=r'..\..\data\NpyData\CarbonGPT_prediction.npy')
    parser.add_argument('--path_dg_true', type=str, default=r'..\..\data\NpyData\power_DG_true.npy')
    parser.add_argument('--path_dg_pred', type=str, default=r'..\..\data\NpyData\power_DG_pred.npy')
    parser.add_argument('--path_urbangpt_true', type=str, default=r'..\..\data\NpyData\UrbanGPT_groundtruth.npy')
    parser.add_argument('--path_urbangpt_pred', type=str, default=r'..\..\data\NpyData\UrbanGPT_prediction.npy')
    parser.add_argument('--save', type=bool, default=False, help='save the data')
    parser.add_argument('--plot', type=bool, default=False, help='plot the data')
    return parser.parse_args()


if __name__ == '__main__':
    options = parse_args()
    generate_opendss_files(options)
    concat_data(options)
    generate_opendss_files(options)
    generateOpendss(options)
    calculate_carbon(options)
