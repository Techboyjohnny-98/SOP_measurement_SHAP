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
from matplotlib.backends.backend_pdf import PdfPages
import os


# -------------------------------
# Step 1: Load the Dataset
# -------------------------------
data = pd.read_csv("SOP_ML_dataset_30T_oldData.csv")
# data = pd.read_csv("SOP_ML_dataset_aging.csv")
y = data["SOP(W)"]
# Separate features and target
X_origin = data.drop(columns=["SOP(W)"])
# Normalize inputs
X = X_origin
# scaler = StandardScaler()
# X = pd.DataFrame(scaler.fit_transform(X_origin), columns=X_origin.columns)

# -------------------------------
# Step 2: Use Cross-validation to tune model hyper-parameters
# -------------------------------
# kf = KFold(n_splits=5, shuffle=True, random_state=42)
# search_space = {
#     "learning_rate":   [0.03, 0.05, 0.1],
#     "max_depth":       [2, 4, 6, 8],
#     "min_child_weight":[2, 4, 6, 8],
#     "gamma":           [0.0, 0.5, 1.0],
#     "subsample":       [0.6, 0.7, 0.8, 0.9],
#     "colsample_bytree":[0.6, 0.7, 0.8, 0.9],
#     "reg_alpha":       [0.0, 0.3, 0.5, 0.8, 1.0],
#     "reg_lambda":      [1, 10, 20, 30],
# }
# # How many random combos to try
# N_ITER = 500
# rng = np.random.default_rng(42)

# # -------------For SHAP only
kf = KFold(n_splits=5, shuffle=True, random_state=42)
search_space = {
    "learning_rate":   [0.1],
    "max_depth":       [5],
    "min_child_weight":[8],
    "gamma":           [0.5],
    "subsample":       [0.6],
    "colsample_bytree":[0.9],
    "reg_alpha":       [0.8],
    "reg_lambda":      [1],
}
# How many random combos to try
N_ITER = 1
rng = np.random.default_rng(42)
# #----------------------------

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
# Step 3: Retrain a XGBoost model.
# -------------------------------
RANDOM_STATE = 12
final_params = {k: best[k] for k in keys}
final_n = max(50, best["best_iter_median"])  # median is robust to outliers
kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

fold_metrics = []
all_shap = []
all_meta = []
all_y_pred_tr = []
all_y_tr = []
all_y_pred_va = []
all_y_va = []
fold_trainingRMSE = []

for fold, (tr_idx, va_idx) in enumerate(kf.split(X, y), start=1):
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

    model = xgb.XGBRegressor(
        n_estimators=final_n,
        tree_method="hist",
        random_state=RANDOM_STATE,
        eval_metric="rmse",
        **final_params
    )
    model.fit(
        X_tr, y_tr,
        verbose=False
    )

    # Evaluate
    y_pred = model.predict(X_va)
    y_pred_tr = model.predict(X_tr)
    rmse = np.sqrt(np.mean((y_va - y_pred) ** 2))
    rmse_tr = np.sqrt(np.mean((y_tr - y_pred_tr) ** 2))
    r2 = r2_score(y_va, y_pred)
    fold_metrics.append((rmse, r2, rmse_tr))
    print(f"Fold {fold}: Testing RMSE={rmse:.2f} W | Training RMSE={rmse_tr:.2f} W  | R²={r2:.4f}")

    # Save predictions and actual values together
    results_df = pd.DataFrame({
        "y_actual": y_va,
        "y_predicted": y_pred
    })
    results_df.to_csv(f"fold_{fold}_results.csv", index=False)
    print(f"Saved predictions and actual values to fold_{fold}_results.csv")

    # Combine all ys together
    y_pred_tr = pd.DataFrame(y_pred_tr)
    y_true = pd.DataFrame(y_tr)
    y_pred = pd.DataFrame(y_pred)
    y_va = pd.DataFrame(y_va)
    all_y_pred_tr.append(y_pred_tr)
    all_y_tr.append(y_tr)
    all_y_pred_va.append(y_pred)
    all_y_va.append(y_va)

# -------------------------------
# Step 4: SHAP analysis.
# -------------------------------
    # SHAP on entire dataset.
    explainer = shap.TreeExplainer(model, X, feature_perturbation="interventional")
    shap_vals = explainer.shap_values(X)

    shap_df = pd.DataFrame(shap_vals, columns=X_origin.columns, index=X.index)
    meta_df = X_origin.copy()
    meta_df["True_SOP(W)"] = y.values

    # Store per-fold SHAP + original features
    combined = pd.concat([meta_df, shap_df.add_prefix("SHAP_")], axis=1)
    combined["fold"] = fold
    all_shap.append(combined)
    shap.summary_plot(shap_vals, X, plot_type="bar")

    # -------------------------------------SHAP value at different SOC values at each random fold
    rows_soc100 = combined[(combined["SOC"] <= 1.1) & (combined["SOC"] >= 0.99)]
    rows_soc95 = combined[(combined["SOC"] <= 0.96) & (combined["SOC"] >= 0.93)]
    rows_soc90 = combined[(combined["SOC"] <= 0.91) & (combined["SOC"] >= 0.89)]
    rows_soc80 = combined[(combined["SOC"] <= 0.81) & (combined["SOC"] >= 0.79)]
    rows_soc70 = combined[(combined["SOC"] <= 0.71) & (combined["SOC"] >= 0.69)]
    rows_soc60 = combined[(combined["SOC"] <= 0.61) & (combined["SOC"] >= 0.59)]
    rows_soc50 = combined[(combined["SOC"] <= 0.51) & (combined["SOC"] >= 0.49)]
    rows_soc40 = combined[(combined["SOC"] <= 0.41) & (combined["SOC"] >= 0.39)]
    rows_soc30 = combined[(combined["SOC"] <= 0.31) & (combined["SOC"] >= 0.29)]
    rows_soc20 = combined[(combined["SOC"] <= 0.21) & (combined["SOC"] >= 0.19)]
    rows_soc15 = combined[(combined["SOC"] <= 0.16) & (combined["SOC"] >= 0.14)]
    rows_soc10 = combined[(combined["SOC"] <= 0.11) & (combined["SOC"] >= 0.09)]
    rows_soc5 = combined[(combined["SOC"] <= 0.06) & (combined["SOC"] >= 0.03)]
    rows_soc2 = combined[(combined["SOC"] <= 0.03) & (combined["SOC"] >= 0.01)]
    mean_values_soc100 = rows_soc100.abs().mean(numeric_only=True)  # avoid errors if non-numeric columns exist
    mean_values_soc95 = rows_soc95.abs().mean(numeric_only=True)
    mean_values_soc90 = rows_soc90.abs().mean(numeric_only=True)
    mean_values_soc80 = rows_soc80.abs().mean(numeric_only=True)
    mean_values_soc70 = rows_soc70.abs().mean(numeric_only=True)
    mean_values_soc60 = rows_soc60.abs().mean(numeric_only=True)
    mean_values_soc50 = rows_soc50.abs().mean(numeric_only=True)
    mean_values_soc40 = rows_soc40.abs().mean(numeric_only=True)
    mean_values_soc30 = rows_soc30.abs().mean(numeric_only=True)
    mean_values_soc20 = rows_soc20.abs().mean(numeric_only=True)
    mean_values_soc15 = rows_soc15.abs().mean(numeric_only=True)
    mean_values_soc10 = rows_soc10.abs().mean(numeric_only=True)
    mean_values_soc5  = rows_soc5.abs().mean(numeric_only=True)
    mean_values_soc2  = rows_soc2.abs().mean(numeric_only=True)
    # Put all mean Series into a list with keys
    mean_list = [
        ("SOC_100", mean_values_soc100),
        ("SOC_95", mean_values_soc95),
        ("SOC_90", mean_values_soc90),
        ("SOC_80", mean_values_soc80),
        ("SOC_70", mean_values_soc70),
        ("SOC_60", mean_values_soc60),
        ("SOC_50", mean_values_soc50),
        ("SOC_40", mean_values_soc40),
        ("SOC_30", mean_values_soc30),
        ("SOC_20", mean_values_soc20),
        ("SOC_15", mean_values_soc15),
        ("SOC_10", mean_values_soc10),
        ("SOC_5", mean_values_soc5),
        ("SOC_2", mean_values_soc2),
    ]
    # Convert into DataFrame
    summary_df = pd.DataFrame({name: vals for name, vals in mean_list}).T
    summary_df.index.name = "SOC_level"
    # print(summary_df)

    # Save to CSV
    summary_df.to_csv(f"SOC_means_summary_fold{fold}.csv")


mean_rmse = np.mean([m[0] for m in fold_metrics])
std_rmse  = np.std([m[0] for m in fold_metrics])
mean_r2   = np.mean([m[1] for m in fold_metrics])
std_r2    = np.std([m[1] for m in fold_metrics])
mean_rmse_tr   = np.mean([m[2] for m in fold_metrics])
std_rmse_tr    = np.std([m[2] for m in fold_metrics])

print("\nCV summary (5-fold):")
print(f"  RMSE: {mean_rmse:.2f} ± {std_rmse:.2f} W")
print(f"  Training RMSE: {mean_rmse_tr:.2f} ± {std_rmse_tr:.2f} W")
print(f"  R²:   {mean_r2:.4f} ± {std_r2:.4f}")

all_shap_df = pd.concat(all_shap, axis=0)

# Sort by SOC (if available) in descending order
if "SOC" in all_shap_df.columns:
    all_shap_df_sorted = all_shap_df.sort_values(by="SOC", ascending=False).reset_index(drop=True)
else:
    all_shap_df_sorted = all_shap_df.reset_index(drop=True)

# -------------------------------------SHAP value at different SOC values at each random state
rows_soc100 = all_shap_df_sorted[(all_shap_df_sorted["SOC"] <= 1.1) & (all_shap_df_sorted["SOC"] >= 0.99)]
rows_soc95 = all_shap_df_sorted[(all_shap_df_sorted["SOC"] <= 0.96) & (all_shap_df_sorted["SOC"] >= 0.93)]
rows_soc90 = all_shap_df_sorted[(all_shap_df_sorted["SOC"] <= 0.91) & (all_shap_df_sorted["SOC"] >= 0.89)]
rows_soc80 = all_shap_df_sorted[(all_shap_df_sorted["SOC"] <= 0.81) & (all_shap_df_sorted["SOC"] >= 0.79)]
rows_soc70 = all_shap_df_sorted[(all_shap_df_sorted["SOC"] <= 0.71) & (all_shap_df_sorted["SOC"] >= 0.69)]
rows_soc60 = all_shap_df_sorted[(all_shap_df_sorted["SOC"] <= 0.61) & (all_shap_df_sorted["SOC"] >= 0.59)]
rows_soc50 = all_shap_df_sorted[(all_shap_df_sorted["SOC"] <= 0.51) & (all_shap_df_sorted["SOC"] >= 0.49)]
rows_soc40 = all_shap_df_sorted[(all_shap_df_sorted["SOC"] <= 0.41) & (all_shap_df_sorted["SOC"] >= 0.39)]
rows_soc30 = all_shap_df_sorted[(all_shap_df_sorted["SOC"] <= 0.31) & (all_shap_df_sorted["SOC"] >= 0.29)]
rows_soc20 = all_shap_df_sorted[(all_shap_df_sorted["SOC"] <= 0.21) & (all_shap_df_sorted["SOC"] >= 0.19)]
rows_soc15 = all_shap_df_sorted[(all_shap_df_sorted["SOC"] <= 0.16) & (all_shap_df_sorted["SOC"] >= 0.14)]
rows_soc10 = all_shap_df_sorted[(all_shap_df_sorted["SOC"] <= 0.11) & (all_shap_df_sorted["SOC"] >= 0.09)]
rows_soc5 = all_shap_df_sorted[(all_shap_df_sorted["SOC"] <= 0.06) & (all_shap_df_sorted["SOC"] >= 0.03)]
rows_soc2 = all_shap_df_sorted[(all_shap_df_sorted["SOC"] <= 0.03) & (all_shap_df_sorted["SOC"] >= 0.01)]
mean_values_soc100 = rows_soc100.abs().mean(numeric_only=True)  # avoid errors if non-numeric columns exist
mean_values_soc95 = rows_soc95.abs().mean(numeric_only=True)
mean_values_soc90 = rows_soc90.abs().mean(numeric_only=True)
mean_values_soc80 = rows_soc80.abs().mean(numeric_only=True)
mean_values_soc70 = rows_soc70.abs().mean(numeric_only=True)
mean_values_soc60 = rows_soc60.abs().mean(numeric_only=True)
mean_values_soc50 = rows_soc50.abs().mean(numeric_only=True)
mean_values_soc40 = rows_soc40.abs().mean(numeric_only=True)
mean_values_soc30 = rows_soc30.abs().mean(numeric_only=True)
mean_values_soc20 = rows_soc20.abs().mean(numeric_only=True)
mean_values_soc15 = rows_soc15.abs().mean(numeric_only=True)
mean_values_soc10 = rows_soc10.abs().mean(numeric_only=True)
mean_values_soc5 = rows_soc5.abs().mean(numeric_only=True)
mean_values_soc2 = rows_soc2.abs().mean(numeric_only=True)
# Put all mean Series into a list with keys
mean_list = [
    ("SOC_100", mean_values_soc100),
    ("SOC_95", mean_values_soc95),
    ("SOC_90", mean_values_soc90),
    ("SOC_80", mean_values_soc80),
    ("SOC_70", mean_values_soc70),
    ("SOC_60", mean_values_soc60),
    ("SOC_50", mean_values_soc50),
    ("SOC_40", mean_values_soc40),
    ("SOC_30", mean_values_soc30),
    ("SOC_20", mean_values_soc20),
    ("SOC_15", mean_values_soc15),
    ("SOC_10", mean_values_soc10),
    ("SOC_5", mean_values_soc5),
    ("SOC_2", mean_values_soc2),
]
# Convert into DataFrame
summary_df = pd.DataFrame({name: vals for name, vals in mean_list}).T
summary_df.index.name = "SOC_level"
# print(summary_df)

# Save to CSV
summary_df.to_csv(f"SOC_means_summary_random{RANDOM_STATE}.csv")

all_shap_df_sorted.to_csv("CV_SHAP_Samsung30T.csv", index=False)
print("Saved CV-SHAP results to CV_SHAP_Samsung30T.csv")

# Save all predictions as .CSV files.
all_y_pred_tr_df = pd.concat(all_y_pred_tr, axis=0)
all_y_tr_df = pd.concat(all_y_tr, axis=0)
all_y_pred_va_df = pd.concat(all_y_pred_va, axis=0)
all_y_va_df = pd.concat(all_y_va, axis=0)
df_pred_tr = pd.DataFrame({
    "Meas (train)": np.asarray(all_y_tr_df).ravel(),
    "Estim (train)": np.asarray(all_y_pred_tr_df).ravel(),
    "split": "train"
})

df_pred_va = pd.DataFrame({
    "Meas (test)": np.asarray(all_y_va_df).ravel(),
    "Estim (test)": np.asarray(all_y_pred_va_df).ravel(),
    "split": "validation"
})

# combine vertically
combined = pd.concat([df_pred_tr, df_pred_va], axis=0, ignore_index=True)
combined.to_csv("CV_predictions_all.csv", index=False)


# -------------------------------
# Step 5: SHAP plot for each feature
# -------------------------------

shap_only = all_shap_df[[c for c in all_shap_df.columns if c.startswith("SHAP_")]]
X_val_all = all_shap_df[X_origin.columns]

# Beeswarm
plt.figure()
shap.summary_plot(shap_only.values, X_val_all, show=False)
plt.tight_layout()
plt.savefig("CV_SHAP_beeswarm.png", dpi=200)
# plt.close()

# Bar
plt.figure()
shap.summary_plot(shap_only.values, X_val_all, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig("CV_SHAP_bar.png", dpi=200)
# plt.close()

print("Saved SHAP summary plots (beeswarm + bar).")

# -------------------------------
# Step 6: SHAP analysis for separate SOC ranges
# -------------------------------


