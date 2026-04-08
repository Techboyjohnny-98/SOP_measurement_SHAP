# Author: Junran Chen
# Standalone Spearman correlation analysis for SOP aging dataset
# Purpose:
#   1) Load CC and US06 datasets independently
#   2) Compute Spearman correlation matrices and p-value matrices
#   3) Save lower-triangle heatmaps
#   4) Save bar plots of feature correlation with SOP
#   5) Save outputs for CC, US06, and combined datasets

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr


# ============================================================
# 0. User settings
# ============================================================
OUTPUT_DIR = Path("Spearman_outputs")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

FEATURES_FOR_CORRELATION = ["SOC", "Temperature(C)", "Capacity", "R_50%", "SOP(W)"]

# ============================================================
# 1. Load data
# ============================================================
data_CC = pd.read_csv("SOP_ML_dataset_aging_CC.csv")
data_US06 = pd.read_csv("SOP_ML_dataset_aging_US06.csv")
data_all = pd.concat([data_CC, data_US06], axis=0).reset_index(drop=True)


# ============================================================
# 2. Helper functions
# ============================================================
def validate_columns(df: pd.DataFrame, required_cols: list[str], dataset_name: str):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {dataset_name}: {missing}")


def compute_spearman_matrices(df: pd.DataFrame, cols: list[str]):
    """
    Compute Spearman correlation and p-value matrices.
    """
    corr_matrix = pd.DataFrame(index=cols, columns=cols, dtype=float)
    pval_matrix = pd.DataFrame(index=cols, columns=cols, dtype=float)

    for c1 in cols:
        for c2 in cols:
            rho, pval = spearmanr(df[c1], df[c2], nan_policy="omit")
            corr_matrix.loc[c1, c2] = rho
            pval_matrix.loc[c1, c2] = pval

    return corr_matrix, pval_matrix


def save_lower_triangle_heatmap(corr_matrix: pd.DataFrame, title: str, save_path: Path):
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(
        corr_matrix.astype(float),
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "Spearman correlation"},
        ax=ax,
    )
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()



def save_target_barplot(corr_matrix: pd.DataFrame, target_col: str, title: str, save_path: Path):
    target_corr = corr_matrix[target_col].drop(target_col).sort_values(key=np.abs, ascending=False)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.bar(target_corr.index, target_corr.values)
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_ylabel(f"Spearman correlation with {target_col}")
    ax.set_title(title)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()



def save_spearman_outputs(df: pd.DataFrame, dataset_name: str, cols: list[str], out_dir: Path):
    validate_columns(df, cols, dataset_name)
    sub_df = df[cols].copy()

    corr_matrix, pval_matrix = compute_spearman_matrices(sub_df, cols)

    corr_matrix.to_csv(out_dir / f"Spearman_corr_{dataset_name}.csv")
    pval_matrix.to_csv(out_dir / f"Spearman_pval_{dataset_name}.csv")

    save_lower_triangle_heatmap(
        corr_matrix=corr_matrix,
        title=f"Spearman correlation ({dataset_name})",
        save_path=out_dir / f"Spearman_heatmap_{dataset_name}.png",
    )

    save_target_barplot(
        corr_matrix=corr_matrix,
        target_col="SOP(W)",
        title=f"Feature correlation with SOP ({dataset_name})",
        save_path=out_dir / f"Spearman_targetbar_{dataset_name}.png",
    )

    summary = corr_matrix["SOP(W)"].drop("SOP(W)").sort_values(key=np.abs, ascending=False)
    summary.to_csv(out_dir / f"Spearman_targetcorr_{dataset_name}.csv", header=["spearman_with_SOP"])

    return corr_matrix, pval_matrix, summary


# ============================================================
# 3. Run analysis
# ============================================================
for dataset_name, df in {
    "CC": data_CC,
    "US06": data_US06,
    "Combined": data_all,
}.items():
    corr_matrix, pval_matrix, summary = save_spearman_outputs(
        df=df,
        dataset_name=dataset_name,
        cols=FEATURES_FOR_CORRELATION,
        out_dir=OUTPUT_DIR,
    )

    print("=" * 70)
    print(f"Dataset: {dataset_name}")
    print("Spearman correlation with SOP(W):")
    print(summary)
    print("Saved outputs to:", OUTPUT_DIR.resolve())

print("\nDone.")
print("All Spearman analysis outputs are saved in:")
print(OUTPUT_DIR.resolve())
