# Author: Junran Chen
# Date: 2025-Sep-23
# Function: Train XGBoost model on SOP measurement dataset, tuned with Cross-validation

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import shap
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from itertools import product


# -------------------------------
# Step 1: Load the Dataset
# -------------------------------
data = pd.read_csv("SOP_ML_dataset_Samsung30T.csv")
# data = pd.read_csv("SOP_ML_dataset_aging.csv")
y = data["SOP(W)"]
# Separate features and target
X_origin = data.drop(columns=["SOP(W)"])
# Normalize inputs
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X_origin), columns=X_origin.columns)

# -------------------------------
# Search space (small but expressive)
# -------------------------------
# kf = KFold(n_splits=5, shuffle=True, random_state=42)
# search_space = {
#     "learning_rate":   [0.03, 0.05, 0.1],
#     "max_depth":       [4, 6, 8],
#     "min_child_weight":[2, 4, 6, 8],
#     "gamma":           [0.0, 0.5, 1.0],
#     "subsample":       [0.6, 0.7, 0.8, 0.9],
#     "colsample_bytree":[0.6, 0.7, 0.8, 0.9],
#     "reg_alpha":       [0.0, 0.5, 1.0],
#     "reg_lambda":      [1, 10, 20, 30],
# }
# # How many random combos to try
# N_ITER = 200
# rng = np.random.default_rng(42)

# -------------For SHAP only
kf = KFold(n_splits=5, shuffle=True, random_state=42)
search_space = {
    "learning_rate":   [0.1],
    "max_depth":       [6],
    "min_child_weight":[4],
    "gamma":           [1],
    "subsample":       [0.7],
    "colsample_bytree":[0.8],
    "reg_alpha":       [1],
    "reg_lambda":      [10],
}
# How many random combos to try
N_ITER = 1
rng = np.random.default_rng(42)
#----------------------------

# Utility: sample parameter sets uniformly from lists above
keys = list(search_space.keys())
choices = [search_space[k] for k in keys]
all_indices = [np.arange(len(v)) for v in choices]
flat_index_grid = np.array(list(product(*all_indices)))
sampled_rows = rng.choice(len(flat_index_grid), size=min(N_ITER, len(flat_index_grid)), replace=False)
param_sets = []
for idx in sampled_rows:
    row = flat_index_grid[idx]
    params = {k: choices[i][row[i]] for i, k in enumerate(keys)}
    param_sets.append(params)

# -------------------------------
# CV evaluator for a single param set
# -------------------------------
def cv_evaluate(params, X, y, kf):
    base = dict(
        n_estimators=2000,       # large; early stopping picks best
        tree_method="hist",
        random_state=42,
        eval_metric="rmse",
        early_stopping_rounds=50,
        # user params below
        **params
    )

    rmses, r2s, best_iters = [], [], []
    for tr_idx, va_idx in kf.split(X, y):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        model = xgb.XGBRegressor(**base)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            verbose=False
        )

        # Use the best iteration found on this fold
        y_pred = model.predict(X_va, iteration_range=(0, model.best_iteration + 1))
        rmses.append(np.sqrt(np.mean((y_va - y_pred) ** 2)))
        r2s.append(r2_score(y_va, y_pred))
        best_iters.append(model.best_iteration + 1)

    return {
        "rmse_mean": float(np.mean(rmses)),
        "rmse_std": float(np.std(rmses)),
        "r2_mean": float(np.mean(r2s)),
        "r2_std": float(np.std(r2s)),
        "best_iter_mean": int(np.round(np.mean(best_iters))),
        "best_iter_median": int(np.median(best_iters)),
    }

# -------------------------------
# Run randomized search
# -------------------------------
results = []
print(f"Evaluating {len(param_sets)} parameter sets...")
for i, params in enumerate(param_sets, start=1):
    stats = cv_evaluate(params, X, y, kf)
    results.append({**params, **stats})
    print(f"[{i:02d}/{len(param_sets)}] RMSE={stats['rmse_mean']:.3f}±{stats['rmse_std']:.3f} | "
          f"R²={stats['r2_mean']:.4f}±{stats['r2_std']:.4f} | "
          f"best_n≈{stats['best_iter_mean']}")

# Rank by RMSE (lower is better); tie-break by higher R²
results_sorted = sorted(results, key=lambda d: (d["rmse_mean"], -d["r2_mean"]))

best = results_sorted[0]
print("\nBest params (by mean CV RMSE):")
for k in keys:
    print(f"  {k}: {best[k]}")
print(f"  CV RMSE: {best['rmse_mean']:.3f} ± {best['rmse_std']:.3f}")
print(f"  CV R²:   {best['r2_mean']:.4f} ± {best['r2_std']:.4f}")
print(f"  Avg best_n_estimators: {best['best_iter_mean']} (median {best['best_iter_median']})")


# -------------------------------
# Refit final model on ALL data
# -------------------------------
RANDOM_STATE = 42
final_params = {k: best[k] for k in keys}
final_n = max(50, best["best_iter_median"])  # median is robust to outliers
final_model = xgb.XGBRegressor(
    n_estimators=final_n,
    tree_method="hist",
    random_state=RANDOM_STATE,
    eval_metric="rmse",
    early_stopping_rounds=50,   # harmless; will likely meet at final_n anyway
    **final_params
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

final_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

y_pred = final_model.predict(X_test)
# rmse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
r2 = r2_score(y_test, y_pred)

print(f"Test RMSE: {rmse:.2f} W")
print(f"Test R² Score: {r2:.4f}")

print(f"\nFinal model trained with n_estimators={final_n}")
final_model.save_model("XGBoost_SOP_model_5FoldCV_Tuned.json")

# --------SHAP analysis----------------
explainer = shap.TreeExplainer(final_model)
shap_values = explainer.shap_values(X)

shap.summary_plot(shap_values, X)
# plt.savefig("SHAP_bee_swarm.svg", format="svg", bbox_inches="tight")

explainer = shap.Explainer(final_model, X)
shap_values = explainer(X)

# Summary plot: feature importance
shap.summary_plot(shap_values, X, plot_type="bar")
# -------------------------------
# Step 6: Calculate heatmap
# -------------------------------
shap_df = pd.DataFrame(shap_values.values, columns=X_origin.columns)
combined_df = X_origin.copy()
combined_df["True_SOP(W)"] = y.values
for col in shap_df.columns:
    combined_df[f"SHAP_{col}"] = shap_df[col]
# Sort by SOC in descending order
combined_df_sorted = combined_df.sort_values(by="SOC", ascending=False).reset_index(drop=True)
# Save sorted data
combined_df_sorted.to_csv("Sorted_SOC_SHAP_Samsung30T.csv", index=False)


