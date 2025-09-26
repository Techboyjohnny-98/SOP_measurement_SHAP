# Author: Junran Chen
# Date: 2025-Sep-23
# Function: Correlation analysis (Pearson & Spearman) vs SOP target

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# -------------------------------
# Step 1: Load dataset
# -------------------------------
data = pd.read_csv("SOP_ML_dataset_30T_oldData.csv")

target = "SOP(W)"
assert target in data.columns, f"'{target}' not found in columns: {list(data.columns)}"

# Use numeric columns only (corr() ignores non-numeric, but this makes it explicit)
num_cols = data.select_dtypes(include="number").columns
data_num = data[num_cols]

# -------------------------------
# Step 2: Compute full matrices
# -------------------------------
pearson_mat  = data_num.corr(method="pearson")
spearman_mat = data_num.corr(method="spearman")

# Also keep the single-column view vs target if you want it
pearson_vs_target  = pearson_mat[target].sort_values(ascending=False)
spearman_vs_target = spearman_mat[target].sort_values(ascending=False)

print("\nCorrelation of features with SOP(W):")
corr_table = pd.DataFrame({"Pearson": pearson_vs_target, "Spearman": spearman_vs_target})
print(corr_table)

# Save tables
corr_table.to_csv("SOP_correlation_table.csv")
pearson_mat.to_csv("Pearson_correlation_matrix.csv")
spearman_mat.to_csv("Spearman_correlation_matrix.csv")
print("Saved SOP_correlation_table.csv, Pearson_correlation_matrix.csv, Spearman_correlation_matrix.csv")

# -------------------------------
# Step 3 (optional): Reorder variables by |corr| with the target, for nicer heatmaps
# -------------------------------
order = pearson_mat[target].abs().sort_values(ascending=False).index
pearson_ordered  = pearson_mat.loc[order, order]
spearman_ordered = spearman_mat.loc[order, order]

# -------------------------------
# Step 4: Heatmap visualization (full matrices)
# -------------------------------
plt.figure(figsize=(8,6))
sns.heatmap(pearson_ordered, cmap="coolwarm", center=0, annot=False, square=True)
plt.title("Pearson Correlation Heatmap (ordered by |corr with SOP|)")
plt.tight_layout()
plt.savefig("Pearson_heatmap.png", dpi=200)

plt.figure(figsize=(8,6))
sns.heatmap(spearman_ordered, cmap="coolwarm", center=0, annot=False, square=True)
plt.title("Spearman Correlation Heatmap (ordered by |corr with SOP|)")
plt.tight_layout()
plt.savefig("Spearman_heatmap.png", dpi=200)

print("Saved Pearson_heatmap.png and Spearman_heatmap.png")
