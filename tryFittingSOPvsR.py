# Author: Junran Chen
# Date: 2025-July-01
# Function: try to find out which function fit SOP VS. R most.

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, r2_score

# SOP at 5% SOC
# x = ([58.89994, 63.36407, 67.42635, 69.32333, 79.27548, 86.09484, 89.86018, 93.26649, 96.64219, 100.84216, 107.66807,
#      110.95081, 113.04215, 117.11217, 127.05812])
# y_true  = ([51.54815, 45.71093, 39.53533, 38.21901, 31.824, 29.229, 27.937, 26.379, 24.729, 23.312,
#            20.304, 19.262, 18.663, 19.418, 17.391])

# x = ([13.04166, 13.16413, 14.01863, 14.84186, 15.67333, 16.51667, 17.57, 18.56333, 19.75667, 21.31667, 22.58333,
#       23.96667, 25.61, 27.18333, 28.63333])
# y_true = ([100.46905, 100.24807, 99.45059, 98.78125, 97.873, 97.017, 95.998, 95.015, 93.719, 92.205, 90.627, 88.939,
#            87.419, 85.736, 84.167])

x = ([13.94988, 15.47327, 18.35861, 20.34817, 23.93333, 26.88667, 29.94667, 32.78898, 37.63364, 38.94605, 47.81788,
     52.32696, 49.46213, 52.94503, 55.63575])
y_true = ([90.80862, 88.92388, 85.43715, 84.20198, 79.827, 76.966, 70.827, 65.154, 56.738, 53.803, 43.405, 38.257,
          39.625, 37.009, 35.311])

x = np.array(x)
y_true = np.array(y_true)


# -------------------------------
# Candidate Functions
# -------------------------------
def linear(x, a, b):
    return a * x + b


def quadratic(x, a, b, c):
    return a * x ** 2 + b * x + c


def exponential(x, a, b):
    return a * np.exp(b * x)


def logarithmic(x, a, b):
    return a * np.log(x + 1e-6) + b  # add small value to avoid log(0)


def power_law(x, a, b):
    return a * np.power(x, b)


def Oneoverx(x, a, b):
    return a + b / x


# -------------------------------
# Try Fitting Each Model
# -------------------------------
models = {
    'Linear': linear,
    # 'Quadratic': quadratic,
    # 'Exponential': exponential,
    'Logarithmic': logarithmic,
    'Power': Oneoverx,
}

results = {}

for name, func in models.items():
    try:
        popt, _ = curve_fit(func, x, y_true, maxfev=10000)
        y_pred = func(x, *popt)
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        results[name] = {'mse': mse, 'r2': r2, 'params': popt}
    except Exception as e:
        results[name] = {'error': str(e)}

# -------------------------------
# Show Results
# -------------------------------
best_fit = max(results.items(), key=lambda x: x[1]['r2'] if 'r2' in x[1] else -np.inf)
print("Best fit:", best_fit[0])
print("Details:", best_fit[1])

# Optional: Plot all fits
plt.scatter(x, y_true, label='Data', color='black')
for name, result in results.items():
    if 'params' in result:
        y_pred = models[name](x, *result['params'])
        plt.plot(x, y_pred, label=f"{name} (R²={result['r2']:.2f})")
plt.legend()
plt.title("Function Fitting Comparison")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
