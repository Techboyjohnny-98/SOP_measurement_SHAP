# Author: Junran Chen
# Date: 2025-June-27
# Function: Create training data for XGBoost.
import numpy as np
import pandas as pd
import glob
import os

# ------A123
# ------Extract features from SOP measurement data (Test #3)
folder_path = '../Source_data/Test#4/'
file_list = glob.glob(os.path.join(folder_path, '*.CSV'))

# Iterate and load all CSV files
alldata_Y = []
alldata_X = []
for file in file_list:
    df = pd.read_csv(file)
    last_indices = []
    index_dis_SOP = df.index[df['Step_Index'] == 52].tolist()
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
    matches = df[df['Step_Index'] == 51]
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

# --------------------Combine all data together
data_X = pd.concat(alldata_X, ignore_index=True)
data_Y = pd.concat(alldata_Y, ignore_index=True)

# Combine features and target into one DataFrame
data_full = data_X.copy()
data_full['SOP(W)'] = data_Y  # add target column to the end
# Save to CSV
output_path = 'SOP_ML_dataset_A123.csv'
data_full.to_csv(output_path, index=False)
# Final check
# print("Input shape:", data_X.shape)
# print("Target shape:", data_Y.shape)
# print(data_X.head())
# print(data_Y.head())

