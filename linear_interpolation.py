# Author: Junran Chen
# Date: 2025-June-31
# Function: Interpolate SOC for Fig.5

import numpy as np

# original data points
x_0 = np.array([85.272, 85.512, 85.245, 83.525, 81.501, 78.948, 75.089,
              61.822, 43.256, 26.856, 20.566, 15.654, 11.269, 9.1985])
x_10 = np.array([93.653, 93.925, 93.411, 91.681, 89.493, 86.68, 83.338,
                79.337, 63.154, 36.81, 26.316, 18.29, 11.663, 8.1414])
x_25 = np.array([102.96, 102.86, 100.41, 100.41, 98.471, 95.775, 92.781,
                 89.868, 85.949, 77.981, 64.648, 42.748, 26.938, 18.658])
x_40 = np.array([107.37, 106.69, 106.24, 104.41, 102.29, 99.492, 96.69,
                 94.157, 91.241, 85.68, 81.557, 69.593, 43.891, 25.226])
y = np.array([1, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5,
              0.4, 0.3, 0.2, 0.15, 0.1, 0.05, 0.02])
x = x_40
# sort the data in ascending order
sorted_indices = np.argsort(x)
x_sorted = x[sorted_indices]
y_sorted = y[sorted_indices]

# points to interpolate
x_new = np.array([52.08])

# interpolation
y_new = np.interp(x_new, x_sorted, y_sorted)
print(y_new)
