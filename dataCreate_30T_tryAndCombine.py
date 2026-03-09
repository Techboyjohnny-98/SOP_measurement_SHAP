# Author: Junran Chen
# Date: 2025-June-27
# Function: Create training data for XGBoost. This
import numpy as np
import pandas as pd
import glob
import os

alldata_Y = []
alldata_X = []
SOP_test3 = []
SOP_test8 = []
# # ------Samsung 30T
# # ------Extract features from SOP measurement data (Old data)
folder_path = '../Source_data/Samsung30T_old/'
file_list = glob.glob(os.path.join(folder_path, '*.CSV'))

# Iterate and load all CSV files
for file in file_list:
    if (file == '../Source_data/Samsung30T_old/SOP_30T_Oct_18_-10degC_10s_Channel_5_Wb_1.CSV'
            or file == '../Source_data/Samsung30T_old/SOP_30T_Sep_26_-20degC_2s_Channel_5_Wb_1.CSV'
            or file == '../Source_data/Samsung30T_old/SOP_30T_Oct_18_-20degC_30s_Channel_5_Wb_1.CSV'
            or file == '../Source_data/Samsung30T_old/SOP_30T_Sep_26_-20degC_10s_Channel_5_Wb_1.CSV'
            or file == '../Source_data/Samsung30T_old/SOP_30T_Oct_17_-10degC_2s_Channel_5_Wb_1.CSV'
            or file == '../Source_data/Samsung30T_old/SOP_30T_Oct_18_-10degC_30s_Channel_5_Wb_1.CSV'
            or file == '../Source_data/Samsung30T_old\\SOP_30T_Oct_18_-10degC_10s_Channel_5_Wb_1.CSV'
            or file == '../Source_data/Samsung30T_old\\SOP_30T_Sep_26_-20degC_2s_Channel_5_Wb_1.CSV'
            or file == '../Source_data/Samsung30T_old\\SOP_30T_Oct_18_-20degC_30s_Channel_5_Wb_1.CSV'
            or file == '../Source_data/Samsung30T_old\\SOP_30T_Sep_26_-20degC_10s_Channel_5_Wb_1.CSV'
            or file == '../Source_data/Samsung30T_old\\SOP_30T_Oct_17_-10degC_2s_Channel_5_Wb_1.CSV'
            or file == '../Source_data/Samsung30T_old\\SOP_30T_Oct_18_-10degC_30s_Channel_5_Wb_1.CSV'):
        SOP_index = 59
        select_index = 58
    else:
        SOP_index = 52
        select_index = 51
    df = pd.read_csv(file)
    last_indices = []
    index_dis_SOP = df.index[df['Step_Index'] == SOP_index].tolist()
    last_points = []
    for i in range(1, len(index_dis_SOP)):
        if index_dis_SOP[i] - index_dis_SOP[i - 1] > 1:
            last_points.append(index_dis_SOP[i - 1])
    last_points.append(index_dis_SOP[-1])
    index_dis_SOP = last_points
    # Add SOP measurement results as Label Y.
    # Try converting to numeric to detect any errors
    df['Power(W)_numeric'] = pd.to_numeric(df['Power(W)'], errors='coerce')
    non_numeric_mask = df['Power(W)_numeric'].isna() & df['Power(W)'].notna()

    if non_numeric_mask.any():
        print(f"[WARNING] Non-numeric entries in 'Power(W)' for file: {file}")
        print(df.loc[non_numeric_mask, ['Power(W)']])
    SOP_disch = -df.loc[index_dis_SOP, 'Power(W)'].reset_index(drop=True)
    matches = df[df['Step_Index'] == select_index]
    if not matches.empty:
        SOP_51_index = matches.index
    # Extract the previous 50-second data.
    index_sequence_Last_point = []
    for step in SOP_51_index:
        time_threshold = df.loc[step, 'Test_Time(s)'] - 2
        matches = df[df['Test_Time(s)'] > time_threshold]
        if not matches.empty:
            first_index = matches.index[0]
            index_sequence_Last_point.append(first_index)
    # Extract SOC, Pulse length, temperature, voltage
    print(file)
    SOC = df.loc[index_sequence_Last_point, 'SOC'].reset_index(drop=True)
    pulseLength = df.loc[index_dis_SOP, 'Step_Time(s)'].reset_index(drop=True)
    temperature = df.loc[index_sequence_Last_point, 'Aux_Temperature_5(C)'].reset_index(drop=True)
    voltage = df.loc[index_sequence_Last_point, 'Voltage(V)'].reset_index(drop=True)
    current = df.loc[index_sequence_Last_point, 'Current(A)'].reset_index(drop=True)
    # Average current - 5 second
    avg_currents_5 = []
    for idx in index_sequence_Last_point:
        current_time = df.loc[idx, 'Test_Time(s)']
        time_window_start = current_time - 5
        # Get rows in the past seconds
        mask = (df['Test_Time(s)'] >= time_window_start) & (df['Test_Time(s)'] <= current_time)
        current_mean = df.loc[mask, 'Current(A)'].abs().mean()
        avg_currents_5.append(current_mean)
    # Average current - 10 second
    avg_currents_10 = []
    for idx in index_sequence_Last_point:
        current_time = df.loc[idx, 'Test_Time(s)']
        time_window_start = current_time - 10
        # Get rows in the past seconds
        mask = (df['Test_Time(s)'] >= time_window_start) & (df['Test_Time(s)'] <= current_time)
        current_mean = df.loc[mask, 'Current(A)'].abs().mean()
        avg_currents_10.append(current_mean)
    # Average current - 20 second
    avg_currents_20 = []
    for idx in index_sequence_Last_point:
        current_time = df.loc[idx, 'Test_Time(s)']
        time_window_start = current_time - 20
        # Get rows in the past seconds
        mask = (df['Test_Time(s)'] >= time_window_start) & (df['Test_Time(s)'] <= current_time)
        current_mean = df.loc[mask, 'Current(A)'].abs().mean()
        avg_currents_20.append(current_mean)
    # Average current - 50 second
    avg_currents_50 = []
    for idx in index_sequence_Last_point:
        current_time = df.loc[idx, 'Test_Time(s)']
        time_window_start = current_time - 50
        # Get rows in the past seconds
        mask = (df['Test_Time(s)'] >= time_window_start) & (df['Test_Time(s)'] <= current_time)
        current_mean = df.loc[mask, 'Current(A)'].abs().mean()
        avg_currents_50.append(current_mean)

    data_X = pd.DataFrame({
        'SOC': SOC,
        'PulseLength(s)': pulseLength,
        'Temperature(C)': temperature,
        'Voltage(V)': voltage,
        'Current(A)': current,
        'AvgCurrent_5s': avg_currents_5,
        'AvgCurrent_10s': avg_currents_10,
        'AvgCurrent_20s': avg_currents_20,
        'AvgCurrent_50s': avg_currents_50
    })
    alldata_X.append(data_X)
    alldata_Y.append(SOP_disch)

# folder_path = '../Source_data/Test#3/'
# file_list = glob.glob(os.path.join(folder_path, '*.csv'))
#
# for file in file_list:
#     if file == '../Source_data/Test#3/JC_SOP_30T_10s_25degC_Sep30_Channel_1_Wb_1.csv':
#         SOP_upRange = 79
#         SOP_downRange = 65
#         selectSOPtoApply = 64
#     elif file == '../Source_data/Test#3/JC_SOP_30T_2s_Sep29_Channel_1_Wb_1.csv':
#         SOP_upRange = 79
#         SOP_downRange = 65
#         selectSOPtoApply = 64
#     elif file == '../Source_data/Test#3/JC_SOP_30T_30s_25degC_Sep27_Channel_1_Wb_1.csv':
#         SOP_upRange = 77
#         SOP_downRange = 65
#         selectSOPtoApply = 64
#     else:
#         SOP_upRange = 80
#         SOP_downRange = 66
#         selectSOPtoApply = 65
#     df = pd.read_csv(file)
#     last_indices = []
#     for step in range(SOP_downRange, SOP_upRange):
#         matches = df[df['Step_Index'] == step]
#         if not matches.empty:
#             last_index = matches.index[-1]  # Get the last row's index, which means real SOP
#             last_indices.append(last_index)
#     index_dis_SOP = last_indices
#     # Add SOP measurement results as Label Y.
#     SOP_disch = -df.loc[index_dis_SOP, 'Power(W)'].reset_index(drop=True)
#     # Find the index that before applying.
#     findFirst_1 = index_dis_SOP[0] - 50
#     matches = df[df['Step_Index'] == selectSOPtoApply]  # 65 - selectSOPtoApply
#     if not matches.empty:
#         SOP_65_index = matches[matches.index > findFirst_1].index
#     # Extract the previous 50-second data.
#     index_sequence_Last_point = []
#     for step in SOP_65_index:
#         time_threshold = df.loc[step, 'Test_Time(s)'] - 2
#         matches = df[df['Test_Time(s)'] > time_threshold]
#         if not matches.empty:
#             first_index = matches.index[0]
#             index_sequence_Last_point.append(first_index)
#     # print(index_sequence_Last_point)
#     # Extract SOC, Pulse length, temperature, voltage
#     SOC = df.loc[index_sequence_Last_point, 'SOC'].reset_index(drop=True)
#     pulseLength = df.loc[index_dis_SOP, 'Step_Time(s)'].reset_index(drop=True)
#     temperature = df.loc[index_sequence_Last_point, 'Aux_Temperature_1(thermocouple1(C))'].reset_index(drop=True)
#     voltage = df.loc[index_sequence_Last_point, 'Voltage(V)'].reset_index(drop=True)
#     current = df.loc[index_sequence_Last_point, 'Current(A)'].reset_index(drop=True)
#     # Average current - 5 second
#     avg_currents_5 = []
#     for idx in index_sequence_Last_point:
#         current_time = df.loc[idx, 'Test_Time(s)']
#         time_window_start = current_time - 5
#         # Get rows in the past seconds
#         mask = (df['Test_Time(s)'] >= time_window_start) & (df['Test_Time(s)'] <= current_time)
#         current_mean = df.loc[mask, 'Current(A)'].abs().mean()
#         avg_currents_5.append(current_mean)
#     # Average current - 10 second
#     avg_currents_10 = []
#     for idx in index_sequence_Last_point:
#         current_time = df.loc[idx, 'Test_Time(s)']
#         time_window_start = current_time - 10
#         # Get rows in the past seconds
#         mask = (df['Test_Time(s)'] >= time_window_start) & (df['Test_Time(s)'] <= current_time)
#         current_mean = df.loc[mask, 'Current(A)'].abs().mean()
#         avg_currents_10.append(current_mean)
#     # Average current - 20 second
#     avg_currents_20 = []
#     for idx in index_sequence_Last_point:
#         current_time = df.loc[idx, 'Test_Time(s)']
#         time_window_start = current_time - 20
#         # Get rows in the past seconds
#         mask = (df['Test_Time(s)'] >= time_window_start) & (df['Test_Time(s)'] <= current_time)
#         current_mean = df.loc[mask, 'Current(A)'].abs().mean()
#         avg_currents_20.append(current_mean)
#     # Average current - 50 second
#     avg_currents_50 = []
#     for idx in index_sequence_Last_point:
#         current_time = df.loc[idx, 'Test_Time(s)']
#         time_window_start = current_time - 50
#         # Get rows in the past seconds
#         mask = (df['Test_Time(s)'] >= time_window_start) & (df['Test_Time(s)'] <= current_time)
#         current_mean = df.loc[mask, 'Current(A)'].abs().mean()
#         avg_currents_50.append(current_mean)
#
#     data_X = pd.DataFrame({
#         'SOC': SOC,
#         'PulseLength(s)': pulseLength,
#         'Temperature(C)': temperature,
#         'Voltage(V)': voltage,
#         'Current(A)': current,
#         'AvgCurrent_5s': avg_currents_5,
#         'AvgCurrent_10s': avg_currents_10,
#         'AvgCurrent_20s': avg_currents_20,
#         'AvgCurrent_50s': avg_currents_50
#     })
#     alldata_X.append(data_X)
#     alldata_Y.append(SOP_disch)
#     SOP_test3.append(SOP_disch)
#
# # ------Extract features from SOP measurement data (Test #8)
# folder_path = '../Source_data/Test#8/'
# file_list = glob.glob(os.path.join(folder_path, '*.csv'))
#
#
# for file in file_list:
#     df = pd.read_csv(file)
#     last_indices = []
#     index_dis_CCCV = df.index[df['Step_Index'] == 36].tolist()
#     last_points = []
#     for i in range(1, len(index_dis_CCCV)):
#         if index_dis_CCCV[i] - index_dis_CCCV[i - 1] > 1:
#             last_points.append(index_dis_CCCV[i - 1])
#     last_points.append(index_dis_CCCV[-1])
#     index_dis_CCCV = last_points
#     # Add SOP measurement results as Label Y.
#     SOP_disch = -df.loc[index_dis_CCCV, 'Power(W)'].reset_index(drop=True)
#     # Find the index that before applying.
#     matches = df[df['Step_Index'] == 35]
#     if not matches.empty:
#         SOP_35_index = matches.index
#     # Extract the previous 50-second data.
#     index_sequence_Last_point = []
#     for step in SOP_35_index:
#         time_threshold = df.loc[step, 'Test_Time(s)'] - 2
#         matches = df[df['Test_Time(s)'] > time_threshold]
#         if not matches.empty:
#             first_index = matches.index[0]
#             index_sequence_Last_point.append(first_index)
#     # Extract SOC, Pulse length, temperature, voltage
#     SOC = df.loc[index_sequence_Last_point, 'SOC'].reset_index(drop=True)
#     pulseLength = df.loc[index_dis_CCCV, 'Step_Time(s)'].reset_index(drop=True)
#     temperature = df.loc[index_sequence_Last_point, 'Aux_Temperature_6(thermocouple6(C))'].reset_index(drop=True)
#     voltage = df.loc[index_sequence_Last_point, 'Voltage(V)'].reset_index(drop=True)
#     current = df.loc[index_sequence_Last_point, 'Current(A)'].reset_index(drop=True)
#     # Average current - 5 second
#     avg_currents_5 = []
#     for idx in index_sequence_Last_point:
#         current_time = df.loc[idx, 'Test_Time(s)']
#         time_window_start = current_time - 5
#         # Get rows in the past seconds
#         mask = (df['Test_Time(s)'] >= time_window_start) & (df['Test_Time(s)'] <= current_time)
#         current_mean = df.loc[mask, 'Current(A)'].abs().mean()
#         avg_currents_5.append(current_mean)
#     # Average current - 10 second
#     avg_currents_10 = []
#     for idx in index_sequence_Last_point:
#         current_time = df.loc[idx, 'Test_Time(s)']
#         time_window_start = current_time - 10
#         # Get rows in the past seconds
#         mask = (df['Test_Time(s)'] >= time_window_start) & (df['Test_Time(s)'] <= current_time)
#         current_mean = df.loc[mask, 'Current(A)'].abs().mean()
#         avg_currents_10.append(current_mean)
#     # Average current - 20 second
#     avg_currents_20 = []
#     for idx in index_sequence_Last_point:
#         current_time = df.loc[idx, 'Test_Time(s)']
#         time_window_start = current_time - 20
#         # Get rows in the past seconds
#         mask = (df['Test_Time(s)'] >= time_window_start) & (df['Test_Time(s)'] <= current_time)
#         current_mean = df.loc[mask, 'Current(A)'].abs().mean()
#         avg_currents_20.append(current_mean)
#     # Average current - 50 second
#     avg_currents_50 = []
#     for idx in index_sequence_Last_point:
#         current_time = df.loc[idx, 'Test_Time(s)']
#         time_window_start = current_time - 50
#         # Get rows in the past seconds
#         mask = (df['Test_Time(s)'] >= time_window_start) & (df['Test_Time(s)'] <= current_time)
#         current_mean = df.loc[mask, 'Current(A)'].abs().mean()
#         avg_currents_50.append(current_mean)
#
#     data_X = pd.DataFrame({
#         'SOC': SOC,
#         'PulseLength(s)': pulseLength,
#         'Temperature(C)': temperature,
#         'Voltage(V)': voltage,
#         'Current(A)': current,
#         'AvgCurrent_5s': avg_currents_5,
#         'AvgCurrent_10s': avg_currents_10,
#         'AvgCurrent_20s': avg_currents_20,
#         'AvgCurrent_50s': avg_currents_50
#     })
#     alldata_X.append(data_X)
#     alldata_Y.append(SOP_disch)
#     SOP_test8.append(SOP_disch)

# --------------------Combine all data together
# --------------------Combine all data together
data_X = pd.concat(alldata_X, ignore_index=True)
data_Y = pd.concat(alldata_Y, ignore_index=True)

# Combine features and target into one DataFrame
data_full = data_X.copy()
data_full['SOP(W)'] = data_Y  # add target column to the end
# Save to CSV
output_path = 'SOP_ML_dataset_30T_tryAndCombine.csv'
data_full.to_csv(output_path, index=False)
# Final check
# print("Input shape:", data_X.shape)
# print("Target shape:", data_Y.shape)
# print(data_X.head())
# print(data_Y.head())

