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
# data = pd.read_csv("SOP_ML_dataset_Samsung30T.csv")
data = pd.read_csv("SOP_ML_dataset_aging.csv")

# Separate features and target
X_origin = data.drop(columns=["SOP(W)"])
# Normalize inputs
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X_origin), columns=X_origin.columns)
y = data["SOP(W)"]

# -------------------------------
# Step 2: Split the Dataset
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------------
# Step 3: Train XGBoost Model
# -------------------------------
model = xgb.XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

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
model.fit(X_train, y_train)

# -------------------------------
# Step 4: Evaluate the Model
# -------------------------------
y_pred = model.predict(X_test)
# rmse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
r2 = r2_score(y_test, y_pred)

print(f"Test RMSE: {rmse:.2f} W")
print(f"Test R² Score: {r2:.4f}")

# -------------------------------
# Step 5: SHAP Analysis
# -------------------------------
explainer = shap.Explainer(model, X)
shap_values = explainer(X)

# Summary plot: feature importance
shap.summary_plot(shap_values, X_test, plot_type="bar")

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
shap_df = pd.DataFrame(shap_values.values, columns=X_origin.columns)
combined_df = X_origin.copy()
combined_df["True_SOP(W)"] = y.values
for col in shap_df.columns:
    combined_df[f"SHAP_{col}"] = shap_df[col]
# Sort by SOC in descending order
combined_df_sorted = combined_df.sort_values(by="SOC", ascending=False).reset_index(drop=True)
# Save sorted data
combined_df_sorted.to_csv("Sorted_SOC_SHAP_aging.csv", index=False)

# -------------------------------
# Step 7: Get mean SHAP values for each SOC range
# -------------------------------

