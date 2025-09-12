# Author: Junran Chen
# Date: 2025-July-01
# Function: try to find out which function fit SOP VS. cycle.

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, r2_score

x = np.array([1, 136, 264, 389, 495, 610, 728, 841, 961, 1067, 1187, 1299, 1418, 1541, 1656, 1769, 1885, 1994])
y_true  = np.array([95.74006, 95.30375, 94.17623, 93.23949, 91.83067, 90.467, 88.872, 87.36467, 84.89567,
                    81.86267, 76.849, 72.88933, 71.818, 68.98667, 65.03533, 62.153, 59.4033, 51.48627])


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
    return a/(x**2) + b

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