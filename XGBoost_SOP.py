# Author: Junran Chen
# Date: 2025-June-27
# Function: Train XGBoost model on SOP measurement dataset

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import shap
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Step 1: Load the Dataset
# -------------------------------
data = pd.read_csv("SOP_ML_dataset.csv")

# Separate features and target
X = data.drop(columns=["SOP(W)"])
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
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

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
# Step 6: Save SHAP Values and Test Data
# -------------------------------

# Convert SHAP values to DataFrame (same shape as X_test)
shap_df = pd.DataFrame(shap_values.values, columns=X_test.columns)

# Save X_test, y_test, and SHAP values
X_test.reset_index(drop=True, inplace=True)
y_test_df = pd.DataFrame(y_test.values, columns=["True_SOP(W)"])

X_test.to_csv("X_test.csv", index=False)
y_test_df.to_csv("Y_test.csv", index=False)
shap_df.to_csv("SHAP_values.csv", index=False)

print("X_test, Y_test, and SHAP values saved to CSV.")
