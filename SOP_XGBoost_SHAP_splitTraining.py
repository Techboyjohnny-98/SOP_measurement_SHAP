# Author: Junran Chen
# Date: 2025-June-27
# Function: Train XGBoost model on SOP measurement dataset

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import shap
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# -------------------------------
# Step 1: Load the Dataset
# -------------------------------
data = pd.read_csv("SOP_ML_dataset_Samsung30T.csv") # Scaled SOP for ID3 + ID8

# Separate features and target
X_origin = data.drop(columns=["SOP(W)"])
# Normalize inputs
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X_origin), columns=X_origin.columns)
y = data["SOP(W)"]

# -------------------------------
# Step 2: Split the Dataset and check if there is any mistakes
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("NaN in y_train:", np.isnan(y_train).any())
print("Inf in y_train:", np.isinf(y_train).any())
# Find columns with NaN values
# Find rows with any NaN values
nan_rows = data[data.isna().any(axis=1)]

# Print the row indices and corresponding rows
if not nan_rows.empty:
    print(f"Found {len(nan_rows)} row(s) containing NaN values:")
    print(nan_rows)
else:
    print("No NaN values found in any row.")
# -------------------------------
# Step 3: Train small XGBoost Models that predict SOP without current information,
# separate for each data sets.
# -------------------------------
model_ID3 = xgb.XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model_ID8 = xgb.XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
# Select rows based on Test ID.
train_ID3_mask = X_origin.loc[X_train.index, "Test#"] == 3
X_train_ID3 = X_train.loc[train_ID3_mask]
Y_train_ID3 = y_train.loc[train_ID3_mask]
X_train_ID3_drpCurrent = X_train_ID3.drop(columns=["Current(A)", "AvgCurrent_5s", "AvgCurrent_10s",
                                           "AvgCurrent_20s", "AvgCurrent_50s", "Test#"])
train_ID8_mask = X_origin.loc[X_train.index, "Test#"] == 8
X_train_ID8 = X_train.loc[train_ID8_mask]
Y_train_ID8 = y_train.loc[train_ID8_mask]
X_train_ID8_drpCurrent = X_train_ID8.drop(columns=["Current(A)", "AvgCurrent_5s", "AvgCurrent_10s",
                                           "AvgCurrent_20s", "AvgCurrent_50s", "Test#"])

model_ID3.fit(X_train_ID3_drpCurrent, Y_train_ID3)
model_ID8.fit(X_train_ID8_drpCurrent, Y_train_ID8)

# -------------------------------
# Step 4: Evaluate the separate Models
# -------------------------------
Test_ID3_mask = X_origin.loc[X_test.index, "Test#"] == 3
X_test_ID3 = X_test.loc[Test_ID3_mask]
Y_test_ID3 = y_test.loc[Test_ID3_mask]
X_test_ID3_drpCurrent = X_test_ID3.drop(columns=["Current(A)", "AvgCurrent_5s", "AvgCurrent_10s",
                                           "AvgCurrent_20s", "AvgCurrent_50s", "Test#"])
Test_ID8_mask = X_origin.loc[X_test.index, "Test#"] == 8
X_test_ID8 = X_test.loc[Test_ID8_mask]
Y_test_ID8 = y_test.loc[Test_ID8_mask]
X_test_ID8_drpCurrent = X_test_ID8.drop(columns=["Current(A)", "AvgCurrent_5s", "AvgCurrent_10s",
                                           "AvgCurrent_20s", "AvgCurrent_50s", "Test#"])

# Evaluate Test ID 3
y_pred = model_ID3.predict(X_test_ID3_drpCurrent)
rmse = np.sqrt(np.mean((Y_test_ID3 - y_pred) ** 2))
r2 = r2_score(Y_test_ID3, y_pred)

print(f"ID3 - Test RMSE: {rmse:.2f} W")
print(f"ID3 - Test R² Score: {r2:.4f}")

y_pred = model_ID3.predict(X_train_ID3_drpCurrent)
rmse = np.sqrt(np.mean((Y_train_ID3 - y_pred) ** 2))
r2 = r2_score(Y_train_ID3, y_pred)

print(f"ID3 - Training RMSE: {rmse:.2f} W")
print(f"ID3 - Training R² Score: {r2:.4f}")

# Evaluate Test ID 8
y_pred = model_ID8.predict(X_test_ID8_drpCurrent)
rmse = np.sqrt(np.mean((Y_test_ID8 - y_pred) ** 2))
r2 = r2_score(Y_test_ID8, y_pred)

print(f"ID8 - Test RMSE: {rmse:.2f} W")
print(f"ID8 - Test R² Score: {r2:.4f}")

y_pred = model_ID8.predict(X_train_ID8_drpCurrent)
rmse = np.sqrt(np.mean((Y_train_ID8 - y_pred) ** 2))
r2 = r2_score(Y_train_ID8, y_pred)

print(f"ID8 - Training RMSE: {rmse:.2f} W")
print(f"ID8 - Training R² Score: {r2:.4f}")

# -------------------------------
# Step 5: Train the final model.
# -------------------------------
# Build baseline SOP using model_ID3 or model_ID8 based on Test#
X_train_dropID = X_train.drop(columns=["Test#"])
baseline_train = pd.Series(index=X_train.index, dtype=float)
baseline_train.loc[train_ID3_mask] = model_ID3.predict(X_train_ID3_drpCurrent)
baseline_train.loc[train_ID8_mask] = model_ID8.predict(X_train_ID8_drpCurrent)
if baseline_train.isna().any():
    missing = X_origin.loc[X_train.index, "Test#"][baseline_train.isna()].unique()
    raise ValueError(f"Missing baseline for Test# values: {missing}")
y_train_delta = y_train - baseline_train
model = xgb.XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model.fit(X_train_dropID, y_train_delta)

# -------------------------------
# Step 6: Evaluate the Final Model (Delta SOP)
# -------------------------------
X_test_dropID = X_test.drop(columns=["Test#"])
baseline_test = pd.Series(index=X_test.index, dtype=float)
baseline_test.loc[Test_ID3_mask] = model_ID3.predict(X_test_ID3_drpCurrent)
baseline_test.loc[Test_ID8_mask] = model_ID8.predict(X_test_ID8_drpCurrent)
if baseline_test.isna().any():
    missing = X_origin.loc[X_test.index, "Test#"][baseline_test.isna()].unique()
    raise ValueError(f"Missing baseline for Test# values: {missing}")
y_test_delta = y_test - baseline_test
y_pred_delta = model.predict(X_test_dropID)
rmse = np.sqrt(np.mean((y_test_delta - y_pred_delta) ** 2))
r2 = r2_score(y_test_delta, y_pred_delta)
print(f"Delta Test RMSE: {rmse:.2f} W")
print(f"Delta Test R2 Score: {r2:.4f}")
y_pred_delta = model.predict(X_train_dropID)
rmse = np.sqrt(np.mean((y_train_delta - y_pred_delta) ** 2))
r2 = r2_score(y_train_delta, y_pred_delta)
print(f"Delta Training RMSE: {rmse:.2f} W")
print(f"Delta Training R2 Score: {r2:.4f}")
# -------------------------------
# Step 5: SHAP Analysis
# -------------------------------
X_dropID = X.drop(columns=["Test#"])
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_dropID)

shap.summary_plot(shap_values, X_dropID)
# plt.savefig("SHAP_bee_swarm.svg", format="svg", bbox_inches="tight")

explainer = shap.Explainer(model, X_dropID)
shap_values = explainer(X_dropID)

# Summary plot: feature importance
shap.summary_plot(shap_values, X_dropID, plot_type="bar")


# Optional: Detailed force plot for one prediction
# shap.initjs()
# shap.force_plot(explainer.expected_value, shap_values[0], X_test.iloc[0], matplotlib=True)

# -------------------------------
# Optional: Save Model
# -------------------------------
model.save_model("XGBoost_SOP_model.json")

# -------------------------------
# Step 6: Calculate heatmap
# -------------------------------
X_origin_dropID = X_origin.drop(columns=["Test#"])
shap_df = pd.DataFrame(shap_values.values, columns=X_origin_dropID.columns)
combined_df = X_origin_dropID.copy()
combined_df["True_SOP(W)"] = y.values
for col in shap_df.columns:
    combined_df[f"SHAP_{col}"] = shap_df[col]
# Sort by SOC in descending order
combined_df_sorted = combined_df.sort_values(by="SOC", ascending=False).reset_index(drop=True)
# Save sorted data
combined_df_sorted.to_csv("Sorted_SOC_SHAP_Samsung30T.csv", index=False)

# -------------------------------
# Step 7: Get mean SHAP values for each SOC range
# -------------------------------




