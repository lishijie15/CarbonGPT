import copy
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from PF_transfer import powerFlowCalculation

plt.rc('font', family='Comic Sans MS')

input_file_voltage = './IEEE8500_EXP_VOLTAGES.CSV'
input_file_power = './IEEE8500_EXP_POWERS.CSV'
file_path_line = '../../data/Line/Lines.dss'
file_path_load_line = '../../data/template/LoadXfmrs.dss'

output_file_load = '../../data/load_list.csv'
output_file_gen = '../../data/gen_list.csv'


# df1 = pd.read_csv(input_file_power)

def all_list(input_file):
    df1 = pd.read_csv(input_file)
    end_index = df1[df1.iloc[:, 0].str.startswith('HVMV_SUB_HSB')].index[0]
    df1 = df1.iloc[1:end_index]
    node_all_list = [i for i in range(len(df1))]
    name_all_list = df1.iloc[:, 0].tolist()
    final_list = list(zip(node_all_list, name_all_list))
    return final_list


def gen_list(input_file):
    df = pd.read_csv(input_file)
    filtered_df = df[df['Element'].str.startswith('Generator') | df['Element'].str.startswith('PVSystem')]

    name_gen_list = filtered_df.iloc[:, 0].tolist()
    name_gen_list = [word.replace(" ", "") for word in name_gen_list]
    name_gen_list = [word.replace("Generator.G_", "") for word in name_gen_list]
    active_power_gen = filtered_df.iloc[:, 2].abs().tolist()
    power_gen_list = [abs(x) for x in active_power_gen]
    final_list = list(zip(name_gen_list, power_gen_list))

    selected_rows1 = df[(df.iloc[:, 0] == 'Reactor.HVMV_SUB_HSB  ') & (df.iloc[:, 1] == 1)]
    if len(selected_rows1) == 0:
        print("Warning: No rows found satisfying HVMV_SUB_HSB!")
    elif len(selected_rows1) != 1:
        print("Warning: More than one line found to satisfy HVMV_SUB_HSB!")
    list_selected_rows1 = selected_rows1.values.tolist()

    total_power_sourcebus = abs(list_selected_rows1[0][2])
    sourcebus_row = ['HVMV_SUB_HSB', total_power_sourcebus]
    final_list.append(tuple(sourcebus_row))

    return final_list


def line_pre(file_path):
    with open(file_path, "r") as file:
        data = file.readlines()
    Line_data = []
    Line_name = []
    Line_bus1 = []
    Line_bus2 = []

    for line_number, line in enumerate(data, start=1):
        line_data = line.strip().split()
        if len(line_data) >= 4:
            if line_data[1].startswith("Line"):
                line_data[1] = line_data[1].replace("Line.", "")
                line_data[1] = line_data[1].upper()
                line_data[2] = line_data[2].replace("bus1=", "").replace(".1", "").replace(".2", "").replace(".3", "")
                line_data[2] = line_data[2].upper()
                line_data[3] = line_data[3].replace("bus2=", "").replace(".1", "").replace(".2", "").replace(".3", "")
                line_data[3] = line_data[3].upper()

                Line_name.append(line_data[1])
                Line_bus1.append(line_data[2])
                Line_bus2.append(line_data[3])
                Line_data.append((line_data[1], line_data[2], line_data[3]))
    return Line_data


def load_line_pre(file_path):
    with open(file_path, "r") as file:
        data = file.readlines()
    Line_data = []
    Line_name = []
    Line_bus1 = []
    Line_bus2 = []
    for line_number, line in enumerate(data, start=1):
        line_data = line.strip().split()
        if len(line_data) >= 4:
            if line_data[1].startswith("Transformer"):
                line_data[1] = line_data[1].replace("Transformer.T5", "")
                line_data[1] = line_data[1].replace("Transformer.T", "")
                line_data[1] = line_data[1].replace("22112371533C", "2212371533C")
                line_data[1] = line_data[1].replace("22112168880C", "2212168880C")
                line_data[1] = line_data[1].replace("22112168874C", "2212168874C")
                line_data[1] = line_data[1].replace("22112168866C", "2212168866C")
                line_data[1] = line_data[1].replace("22112168870C", "2212168870C")
                line_data[1] = line_data[1].replace("22112168887C", "2212168887C")
                line_data[1] = line_data[1].replace("0107693B", "52107693B")
                line_data[1] = line_data[1].replace("0107636A", "52107636A")

                line_data[1] = line_data[1].upper()
                line_data[5] = line_data[5].replace("bus=", "").replace(".1", "").replace(".2", "").replace(".3", "")
                line_data[9] = line_data[9].replace("bus=", "").replace(".1.0", "")

                Line_name.append(line_data[1])
                Line_bus1.append(line_data[5])
                Line_bus2.append(line_data[9])
                Line_data.append((line_data[1], line_data[5], line_data[9]))
    return Line_data


def data_matching_line(Line_data, power_file):
    df1 = pd.read_csv(power_file)
    line_data1 = copy.copy(Line_data)

    filtered_df = df1[df1['Element'].str.startswith('Line.') & ~df1['Element'].str.startswith('Line.TPX')]

    power_list_name = filtered_df.iloc[:, 0].tolist()
    power_list_terminal = filtered_df.iloc[:, 1].tolist()
    power_list_active = filtered_df.iloc[:, 2].tolist()
    power_list_reactive = filtered_df.iloc[:, 3].tolist()

    power_list_name = [word.replace(" ", "") for word in power_list_name]
    power_list_name = [word.replace("Line.", "") for word in power_list_name]
    power_list_name = [word.upper() for word in power_list_name]
    power_list_total = power_list_active
    power_list = list(zip(power_list_name, power_list_terminal, power_list_total))
    power_list_forward = [tuple for tuple in power_list if tuple[1] == 1]

    for tuple2 in power_list_forward:
        for idx, tuple1 in enumerate(line_data1):
            if tuple1[0] == tuple2[0]:
                line_data1[idx] = (tuple1[0], tuple1[1], tuple1[2], tuple2[2])

    rows_with_three_elements = [row for row in line_data1 if len(row) == 3]

    line_data1 = [row for row in line_data1 if len(row) != 3]
    return line_data1


def data_matching_transformer_line(Line_data, power_file):
    Line_data1 = copy.copy(Line_data)
    df1 = pd.read_csv(power_file)
    filtered_df = df1[df1['Element'].str.startswith('Transformer.') & ~df1['Element'].str.startswith('Transformer.G')]

    power_list_name = filtered_df.iloc[:, 0].tolist()
    power_list_terminal = filtered_df.iloc[:, 1].tolist()
    power_list_active = filtered_df.iloc[:, 2].tolist()
    power_list_reactive = filtered_df.iloc[:, 3].tolist()

    power_list_name = [word.replace(" ", "") for word in power_list_name]
    power_list_name = [word.replace("Transformer.T5", "") for word in power_list_name]
    power_list_name = [word.replace("Transformer.T", "") for word in power_list_name]
    power_list_name = [word.replace("Transformer.", "") for word in power_list_name]
    power_list_name = [word.replace("22112371533C", "2212371533C") for word in power_list_name]
    power_list_name = [word.replace("22112168880C", "2212168880C") for word in power_list_name]
    power_list_name = [word.replace("22112168874C", "2212168874C") for word in power_list_name]
    power_list_name = [word.replace("22112168866C", "2212168866C") for word in power_list_name]
    power_list_name = [word.replace("22112168870C", "2212168870C") for word in power_list_name]
    power_list_name = [word.replace("22112168887C", "2212168887C") for word in power_list_name]
    power_list_name = [word.replace("0107693B", "52107693B") for word in power_list_name]
    power_list_name = [word.replace("0107636A", "52107636A") for word in power_list_name]
    power_list_name = [word.upper() for word in power_list_name]

    power_list_total = [abs(x) for x in power_list_active]

    power_list = list(zip(power_list_name, power_list_terminal, power_list_total))
    power_list_forward = [tuple for tuple in power_list if tuple[1] == 1]

    for tuple2 in power_list_forward:
        found = False
        for idx, tuple1 in enumerate(Line_data1):
            if tuple1[0] == tuple2[0]:
                Line_data1[idx] = (tuple1[0], tuple1[1], tuple1[2], tuple2[2])
                found = True
        # if not found: print(tuple2)
    return Line_data1


def matrix_assignment_line(line_list, all_list):
    dimension = len(all_list)
    p_b = np.zeros((dimension, dimension))
    # z = 1
    # k = 0
    for i, row in enumerate(line_list):
        found_second = False
        for index, info in all_list:
            if info == row[1]:
                for index2, info2 in all_list:
                    if info2 == row[2]:
                        if p_b[index][index2] == 0:
                            p_b[index][index2] = row[3]
                            found_second = True
                        else:
                            # print(row, p_b[index][index2])
                            p_b[index][index2] = p_b[index][index2] + row[3]
                            found_second = True
                        break
                break
        if not found_second:
            print('Incomplete P_B matching:', row)
        # nonzero_values_p_b = p_b[p_b != 0]
        # if len(nonzero_values_p_b) != z:
        #     k = k+1
        #     print('error', k, row)
        #     z = z-1
        # z = z + 1
    return p_b


def matrix_assignment_generator(line_list, all_list):
    matrix = np.zeros((len(line_list), len(all_list)))
    line_list = [list(item) for item in line_list]
    for i in range(len(line_list)):
        line_list[i][0] = line_list[i][0].replace('PVSystem.PV_', '').replace('HVMV_SUB_HSB', '_HVMV_SUB_LSB')
    line_list = [tuple(item) for item in line_list]

    for i, row in enumerate(line_list):
        found_second = False
        for index, info in all_list:
            if info == row[0]:
                matrix[i][index] = row[1]
                found_second = True
        if not found_second:
            print('Incomplete P_G matching:', row)
            found_second = True

    return matrix


def matrix_assignment_load(line_list, all_list):
    matrix = np.zeros((len(line_list), len(all_list)))

    for i, row in enumerate(line_list):
        found_second = False
        for index, info in all_list:
            if info == row[1]:
                matrix[i][index] = row[3]
                found_second = True
        if not found_second:
            print('Incomplete P_L matching:', row)
    return matrix


def update_matrices_and_vectors(E_N, P_N, P_B, P_G, P_L):
    # Step 1: Identify zero diagonal elements  (feasibility analysis)
    zero_diag_indices = np.where(np.diagonal(E_N) == 0)[0]
    # Step 2: Remove corresponding rows and columns from E_N
    E_N = np.delete(E_N, zero_diag_indices, axis=0)
    E_N = np.delete(E_N, zero_diag_indices, axis=1)

    # Step 3: Update P_N and P_B accordingly
    P_N = np.delete(P_N, zero_diag_indices, axis=0)
    P_N = np.delete(P_N, zero_diag_indices, axis=1)
    P_B = np.delete(P_B, zero_diag_indices, axis=0)
    P_B = np.delete(P_B, zero_diag_indices, axis=1)
    P_G = np.delete(P_G, zero_diag_indices, axis=1)
    P_L = np.delete(P_L, zero_diag_indices, axis=1)

    return E_N, P_N, P_B, P_G, P_L, zero_diag_indices


def calculate_carbon_emission_flow(P_B, P_G, P_L, E_G, load_node_index):
    Q_N = np.ones((1, P_B.shape[0]))
    P_Z = np.concatenate([P_B, P_G], axis=0)
    Q_NK = np.ones((1, P_Z.shape[0]))

    P_N = np.matmul(Q_NK, P_Z)
    P_N = np.diag(P_N.squeeze(0))
    E_N = (P_N - P_B.T)

    E_N, P_N, P_B, P_G, P_L, zero_diag_index = update_matrices_and_vectors(E_N, P_N, P_B, P_G, P_L)
    P_N_original = np.diag(P_N)

    P_Z = np.concatenate([P_B, P_G], axis=0)
    Q_NK = np.ones((1, P_Z.shape[0]))

    E_N = np.linalg.inv(E_N)
    E_N = np.matmul(E_N, P_G.T)
    E_N = np.matmul(E_N, E_G)

    R_G = np.sum(P_G, axis=1)
    R_G = R_G * E_G  # kgCO₂/h
    R_B = np.matmul(np.diag(E_N), P_B)
    R_L = np.matmul(P_L, E_N)  # kgCO₂/h
    R_N = np.matmul(Q_NK, P_Z).T
    R_N_original = np.copy(R_N)
    R_N = np.multiply(R_N, E_N)

    new_load_index = []
    for i in range(len(load_node_index)):
        if load_node_index[i] in zero_diag_index:
            pass
        else:
            count = 0
            for j in range(len(zero_diag_index)):
                if zero_diag_index[j] < load_node_index[i]:
                    count += 1
            new_load_index.append(load_node_index[i] - count)

    E_L = E_N[new_load_index]
    P_L = P_N_original[new_load_index]

    return P_Z, E_N, R_G, R_B, R_L, R_N, P_L, R_N_original, P_N_original, zero_diag_index, E_L


def Fig_plt(P_Z, E_N, R_G, R_B, R_L, R_N, P_L, P_N):
    scaler = MinMaxScaler()
    E_N_normalized = scaler.fit_transform(E_N.reshape(-1, 1)).reshape(-1)
    P_N_normalized = scaler.fit_transform(P_N.reshape(-1, 1)).reshape(-1)
    # P_N_normalized = P_N
    # E_N_normalized = E_N

    plt.figure(figsize=(24, 6))
    E_N_normalized = np.log1p(E_N_normalized)
    import matplotlib.colors as mcolors
    norm = mcolors.PowerNorm(gamma=1)
    plt.imshow(E_N_normalized.reshape(1, -1), cmap='GnBu', aspect='auto', norm=norm)
    plt.colorbar()
    plt.xlabel('Node Index', fontsize=14)
    plt.ylabel('Carbon Emission Intensity (kgCO_2 /kWh)', fontsize=14)
    plt.title('Carbon Emission Distribution Across Nodes', fontsize=16)
    plt.savefig('hot_CEI.svg', format='svg', bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(24, 6))
    P_N_normalized = np.log1p(P_N_normalized)
    plt.imshow(P_N_normalized.reshape(1, -1), cmap='GnBu', aspect='auto', norm=norm)
    plt.colorbar()
    plt.xlabel('Node Index', fontsize=14)
    plt.ylabel('Active Power (kW)', fontsize=14)
    plt.title('Active Power Distribution Across Nodes', fontsize=16)
    plt.savefig('hot_AP.svg', format='svg', bbox_inches='tight')
    plt.show()


def modify_master_file(i, original_content):
    modified_content = re.sub(r'Redirect\s+\.\./\.\./result/Generator/Generators_\d+\.dss',
                              f'Redirect  ../../result/Generator/Generators_{i}.dss',
                              original_content)
    modified_content = re.sub(r'Redirect\s+\.\./\.\./result/Load/Loads_test\d+\.dss',
                              f'Redirect  ../../result/Load/Loads_test{i}.dss',
                              modified_content)
    return modified_content


def process_data():
    final_all_list = all_list(input_file_voltage)
    final_gen_list = gen_list(input_file_power)
    Line_Data = line_pre(file_path_line)
    load_line_data = load_line_pre(file_path_load_line)

    final_all_line = data_matching_line(Line_Data, input_file_power)
    final_data_transformer_line = data_matching_transformer_line(load_line_data, input_file_power)

    P_B = matrix_assignment_line(final_all_line, final_all_list)  # line
    P_G = matrix_assignment_generator(final_gen_list, final_all_list)  # generator
    P_L = matrix_assignment_load(final_data_transformer_line, final_all_list)  # Transformer
    print(np.sum(P_G))
    print(np.sum(P_L))
    E_G = 0.0515 * np.ones((P_G.shape[0]))
    E_G[31] = E_G[5] = 0.0336
    E_G[-3:] = [0.98883, 0, 0.98883]

    return P_B, P_G, P_L, E_G


def calculate_carbon(opt):
    with open('./Master.dss', 'r') as file:
        original_content = file.read()

    load_node_index = pd.read_csv(r'.\data\template\1177nodes.csv', header=None).iloc[:, 0].tolist()

    for i in range(1, 13):
        modified_content = modify_master_file(i, original_content)
        with open('./Master.dss', 'w') as file:
            file.write(modified_content)

        powerFlowCalculation()
        P_B, P_G, P_L, E_G = process_data()

        P_Z, E_N, R_G, R_B, R_L, R_N, P_L, R_N_original, P_N, zero_diag_zero, E_L = calculate_carbon_emission_flow(P_B, P_G, P_L, E_G,
                                                                                                                   load_node_index)
        print(np.sum(E_N))

        model_names = {'json': 'CarbonGPT', 'npy': opt.true_data.split('\\')[-1].split('_')[0], 'dg': 'AGCRN', 'urbangpt': 'UrbanGPT'}
        if opt.save:
            np.save(f'../../{model_names[opt.file_type]}_{opt.index}_{i}_{opt.pred_or_true}_EN.npy', E_N)
            np.save(f'../../{model_names[opt.file_type]}_{opt.index}_{i}_{opt.pred_or_true}_zero_diag_index.npy', zero_diag_zero)
            # np.save(f'../../{model_names[opt.file_type]}_{opt.index}_{opt.pred_or_true}_EL.npy', E_L)
            # np.save(f'../../{model_names[opt.file_type]}_{opt.index}_{opt.pred_or_true}_PN.npy', P_N)
            # np.save(f'../../{model_names[opt.file_type]}_{opt.index}_{opt.pred_or_true}_PL.npy', P_L)
            # np.save(f'../../{model_names[opt.file_type]}_{opt.index}_{opt.pred_or_true}_RL.npy', R_L)
        if opt.plot:
            Fig_plt(P_Z, E_N, R_G, R_B, R_L, R_N_original, P_L, P_N)


if __name__ == "__main__":
    from generate_opendss import parse_args

    opt = parse_args()
    calculate_carbon(opt)
