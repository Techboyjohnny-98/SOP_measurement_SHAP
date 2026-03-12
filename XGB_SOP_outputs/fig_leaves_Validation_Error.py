import numpy as np
import pandas as pd
from pathlib import Path

# =========================
# User settings
# =========================
# input_csv = "Tuning_results_test_CC.csv"
# input_csv = "Tuning_results_test_US06.csv"
input_csv = "Final_tuning_results_all_data.csv"
n_bins = 15   # change if needed

# =========================
# Load data
# =========================
df = pd.read_csv(input_csv)

required_cols = ["n_leaves", "val_rmse"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in {input_csv}")

df = df.copy()
df = df[df["n_leaves"] > 0].reset_index(drop=True)

# =========================
# Create linear bins
# =========================
xmin = df["n_leaves"].min()
xmax = df["n_leaves"].max()

bin_edges = np.linspace(xmin, xmax, n_bins + 1)

df["bin_id"] = pd.cut(
    df["n_leaves"],
    bins=bin_edges,
    labels=False,
    include_lowest=True
)

# Make bin_id start from 1
df["bin_id"] = df["bin_id"] + 1

# =========================
# Compute per-bin summary
# =========================
bin_summary = (
    df.groupby("bin_id", as_index=False)
      .agg(
          bin_mid=("n_leaves", "median"),
          bin_median=("val_rmse", "median"),
          bin_min=("val_rmse", "min"),
          bin_count=("val_rmse", "size")
      )
)

bin_summary["bin_left"] = [bin_edges[i - 1] for i in bin_summary["bin_id"]]
bin_summary["bin_right"] = [bin_edges[i] for i in bin_summary["bin_id"]]

# =========================
# Merge back into original file
# =========================
df_with_bins = df.merge(bin_summary, on="bin_id", how="left")

# =========================
# Save outputs
# =========================
input_path = Path(input_csv)
stem = input_path.stem

out_augmented = input_path.with_name(f"{stem}_with_bins.csv")
out_summary = input_path.with_name(f"{stem}_bin_summary.csv")

df_with_bins.to_csv(out_augmented, index=False)
bin_summary.to_csv(out_summary, index=False)

print(f"Saved augmented file: {out_augmented}")
print(f"Saved bin summary file: {out_summary}")
print("\nPreview of bin summary:")
print(bin_summary.head(10))