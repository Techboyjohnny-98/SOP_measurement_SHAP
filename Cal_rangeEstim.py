# Author: Junran Chen
# Date: 2025-July-01
# Function: Calculate usable energy and range estimation from US06 drive cycle profiels.

import pandas as pd
import os
import boto3
import numpy as np

# Settings
bucket_name = 'sopagingsamng30t'
s3_key = 'Raw/US06/JC_aging_chanel17_US06_Cycle1340_Sep15_Channel_17_Wb_1.CSV'
local_path = './cache/JC_aging_chanel17_US06_Cycle1340_Sep15_Channel_17_Wb_1.csv'

# Create local cache folder if it doesn't exist
os.makedirs(os.path.dirname(local_path), exist_ok=True)

# Download if not already cached
if not os.path.exists(local_path):
    print("Downloading from S3...")
    s3 = boto3.client('s3')
    s3.download_file(bucket_name, s3_key, local_path)
else:
    print("Using cached file.")

# Load with pandas
sourceData = pd.read_csv(local_path)

Ah = sourceData[(sourceData['Step_Index'] == 186) & (sourceData['TC_Counter1'] == 1)]['Discharge_Capacity(Ah)']
Energy = sourceData[(sourceData['Step_Index'] == 186) & (sourceData['TC_Counter1'] == 1)]['Discharge_Energy(Wh)']

# Calculate from linear_interpolation.py
Ah_new = 2.13292414

Energy_new = np.interp(Ah_new, Ah, Energy)

print(Energy_new)









