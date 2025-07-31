# Author: Junran Chen
# Date: 2025-July-01
# Function: try to find out which function fit SOP VS. R most.

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, r2_score

x = np.array([58.89994, 63.36407, 69.32333, 67.42635, 79.27548, 86.09484, 89.86018, 93.26649, 100.84216, 96.64219, 117.11217,
     127.05812, 107.66807, 110.95081, 113.04215])
y_true  = np.array([51.54815, 45.71093, 38.21901, 39.53533, 31.824, 29.229, 27.937, 26.379, 23.312, 24.729, 19.418, 17.391,
     20.304, 19.262, 18.663])


# -------------------------------
# Candidate Functions
# -------------------------------
def linear(x, a, b):
    return a * x + b

def quadratic(x, a, b, c):
    return a * x**2 + b * x + c

def exponential(x, a, b):
    return a * np.exp(b * x)

def logarithmic(x, a, b):
    return a * np.log(x + 1e-6) + b  # add small value to avoid log(0)

def power_law(x, a, b):
    # return a * np.power(x, b)
    return a/x + b

# -------------------------------
# Try Fitting Each Model
# -------------------------------
models = {
    'Linear': linear,
    'Quadratic': quadratic,
    'Exponential': exponential,
    'Logarithmic': logarithmic,
    'Power': power_law,
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