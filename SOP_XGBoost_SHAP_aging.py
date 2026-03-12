# Author: Junran Chen
# Updated for paper-ready SOP XGBoost workflow
# Purpose:
#   1) Leave-one-cell-out evaluation
#   2) Chronological validation split within training pool
#   3) Hyperparameter tuning with fixed seed + early stopping
#   4) Robustness check across multiple training seeds
#   5) Final SHAP analysis on full dataset
#   6) Export SOC-binned SHAP summaries
#   7) Use early stopping only during tuning
#   8) Use fixed n_estimators for final evaluation and SHAP

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import xgboost as xgb

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import ParameterGrid


# ============================================================
# 0. User settings
# ============================================================
TARGET_COL = "SOP(W)"
SOC_COL = "SOC"

# If your file already has an aging-order column, put its name here.
# Example: AGING_ORDER_COL = "Aging_cycle"
# If None, the script will assume the current row order is already chronological.
AGING_ORDER_COL = None

# Validation fraction within each training cell/pool
VAL_FRAC = 0.20

# Fixed seed for hyperparameter tuning
TUNE_SEED = 42

# Seeds used to evaluate stochastic training robustness
ROBUSTNESS_SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

# Fixed seed for the final SHAP model
FINAL_SHAP_SEED = 42

# Output folder
OUTPUT_DIR = Path("XGB_SOP_outputs")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Whether to run SHAP ranking stability across seeds
RUN_SHAP_STABILITY = True

# SOC bin width for mean SHAP export
SOC_BIN_WIDTH_PERCENT = 5

# Early stopping / tuning settings
TUNING_N_ESTIMATORS_CEILING = 1000
EARLY_STOPPING_ROUNDS = 50

# ------------------------------------------------------------
# Hyperparameter grid
# Keep this modest because the dataset is not large
# ------------------------------------------------------------
PARAM_GRID = {
    "learning_rate": [0.03, 0.05, 0.1],
    "max_depth": [2, 3, 4],
    "min_child_weight": [3, 5, 8],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.75, 1.0],
    "reg_lambda": [1, 5, 10, 20],
    "gamma": [0, 0.1, 0.3, 1.0]
}


# ============================================================
# 1. Load data
# ============================================================
data_CC = pd.read_csv("SOP_ML_dataset_aging_CC.csv")
data_US06 = pd.read_csv("SOP_ML_dataset_aging_US06.csv")


# ============================================================
# 2. Helper functions
# ============================================================
CELL_COL = "__cell_id__"
ORDER_COL = "__aging_order__"


def add_cell_and_order_columns(df: pd.DataFrame, cell_label: str) -> pd.DataFrame:
    df = df.copy()
    df[CELL_COL] = cell_label

    if AGING_ORDER_COL is not None:
        if AGING_ORDER_COL not in df.columns:
            raise ValueError(
                f"AGING_ORDER_COL='{AGING_ORDER_COL}' not found in dataframe columns."
            )
        df[ORDER_COL] = df[AGING_ORDER_COL].values
    else:
        # Assume current row order is already chronological
        df[ORDER_COL] = np.arange(len(df))

    df = df.sort_values(ORDER_COL).reset_index(drop=True)
    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    excluded = {TARGET_COL, CELL_COL, ORDER_COL}
    if AGING_ORDER_COL is not None:
        excluded.add(AGING_ORDER_COL)

    feature_cols = [c for c in df.columns if c not in excluded]
    return feature_cols


def chronological_split_single_cell(
    df: pd.DataFrame,
    val_frac: float = 0.20
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a single-cell dataframe chronologically.
    If AGING_ORDER_COL is given, split by unique aging stages/checkpoints.
    Otherwise split by row order.
    """
    df = df.sort_values(ORDER_COL).reset_index(drop=True)

    if AGING_ORDER_COL is not None:
        unique_steps = pd.Series(df[ORDER_COL].unique()).sort_values().tolist()
        n_val_steps = max(1, int(np.ceil(len(unique_steps) * val_frac)))
        val_steps = set(unique_steps[-n_val_steps:])

        val_df = df[df[ORDER_COL].isin(val_steps)].copy()
        train_df = df[~df[ORDER_COL].isin(val_steps)].copy()
    else:
        split_idx = int(np.floor(len(df) * (1 - val_frac)))
        split_idx = max(1, min(split_idx, len(df) - 1))

        train_df = df.iloc[:split_idx].copy()
        val_df = df.iloc[split_idx:].copy()

    return train_df, val_df


def chronological_split_training_pool(
    df_pool: pd.DataFrame,
    val_frac: float = 0.20
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split each training cell chronologically, then combine.
    This is generic and still works when the training pool has only one cell.
    """
    train_parts = []
    val_parts = []

    for cell_name in df_pool[CELL_COL].unique():
        df_cell = df_pool[df_pool[CELL_COL] == cell_name].copy()
        tr, va = chronological_split_single_cell(df_cell, val_frac=val_frac)
        train_parts.append(tr)
        val_parts.append(va)

    train_df = pd.concat(train_parts, axis=0).sort_values([CELL_COL, ORDER_COL]).reset_index(drop=True)
    val_df = pd.concat(val_parts, axis=0).sort_values([CELL_COL, ORDER_COL]).reset_index(drop=True)
    return train_df, val_df


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def build_model(
    params: dict,
    seed: int,
    n_estimators: int = 300,
    use_early_stopping: bool = False,
    early_stopping_rounds: int = 50
) -> xgb.XGBRegressor:
    extra = {}
    if use_early_stopping:
        extra["early_stopping_rounds"] = early_stopping_rounds

    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=seed,
        n_jobs=-1,
        tree_method="hist",
        n_estimators=n_estimators,
        **extra,
        **params
    )
    return model


def tune_hyperparameters(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    param_grid: dict,
    seed: int,
    tuning_n_estimators_ceiling: int = 1000,
    early_stopping_rounds: int = 50
) -> tuple[dict, int, pd.DataFrame]:
    """
    Hyperparameter tuning using a fixed random seed and early stopping.
    Returns:
        best_params
        best_n_estimators
        tuning_df
    """
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_val = val_df[feature_cols]
    y_val = val_df[target_col]

    records = []
    best_params = None
    best_n_estimators = None
    best_rmse = np.inf

    for params in ParameterGrid(param_grid):
        model = build_model(
            params=params,
            seed=seed,
            n_estimators=tuning_n_estimators_ceiling,
            use_early_stopping=True,
            early_stopping_rounds=early_stopping_rounds
        )

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        booster = model.get_booster()
        tree_df = booster.trees_to_dataframe()

        n_leaves = int((tree_df["Feature"] == "Leaf").sum())
        n_splits = int((tree_df["Feature"] != "Leaf").sum())
        n_nodes = int(len(tree_df))

        y_val_pred = model.predict(X_val)
        val_rmse = rmse(y_val, y_val_pred)
        val_r2 = r2_score(y_val, y_val_pred)

        chosen_n_estimators = int(model.best_iteration) + 1

        rec = dict(params)
        rec["val_rmse"] = val_rmse
        rec["val_r2"] = val_r2
        rec["best_iteration"] = int(model.best_iteration)
        rec["best_n_estimators"] = chosen_n_estimators
        rec["n_leaves"] = n_leaves
        rec["n_splits"] = n_splits
        rec["n_nodes"] = n_nodes
        records.append(rec)

        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_params = dict(params)
            best_n_estimators = chosen_n_estimators

    tuning_df = pd.DataFrame(records).sort_values("val_rmse").reset_index(drop=True)
    return best_params, best_n_estimators, tuning_df


def evaluate_across_seeds(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    best_params: dict,
    best_n_estimators: int,
    seeds: list,
    fold_name: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Retrain on train+val with fixed n_estimators and no early stopping,
    then evaluate on the held-out test cell.

    Returns
    -------
    metrics_df : pd.DataFrame
        Per-seed summary metrics.
    predictions_df : pd.DataFrame
        Row-wise true/predicted values for train and test sets.
    """
    train_full = (
        pd.concat([train_df, val_df], axis=0)
        .sort_values([CELL_COL, ORDER_COL])
        .reset_index(drop=True)
    )

    X_train_full = train_full[feature_cols]
    y_train_full = train_full[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    metrics_results = []
    prediction_rows = []

    for seed in seeds:
        model = build_model(
            params=best_params,
            seed=seed,
            n_estimators=best_n_estimators,
            use_early_stopping=False
        )

        model.fit(X_train_full, y_train_full, verbose=False)

        y_train_pred = model.predict(X_train_full)
        y_test_pred = model.predict(X_test)

        # Summary metrics
        metrics_results.append({
            "fold": fold_name,
            "seed": seed,
            "n_estimators": best_n_estimators,
            "train_rmse": rmse(y_train_full, y_train_pred),
            "train_r2": r2_score(y_train_full, y_train_pred),
            "test_rmse": rmse(y_test, y_test_pred),
            "test_r2": r2_score(y_test, y_test_pred)
        })

        # Row-wise train predictions
        train_pred_df = train_full.copy().reset_index(drop=True)
        train_pred_df["fold"] = fold_name
        train_pred_df["seed"] = seed
        train_pred_df["split"] = "train"
        train_pred_df["y_true"] = y_train_full.values
        train_pred_df["y_pred"] = y_train_pred

        # Row-wise test predictions
        test_pred_df = test_df.copy().reset_index(drop=True)
        test_pred_df["fold"] = fold_name
        test_pred_df["seed"] = seed
        test_pred_df["split"] = "test"
        test_pred_df["y_true"] = y_test.values
        test_pred_df["y_pred"] = y_test_pred

        prediction_rows.append(train_pred_df)
        prediction_rows.append(test_pred_df)

    metrics_df = pd.DataFrame(metrics_results)
    predictions_df = pd.concat(prediction_rows, axis=0).reset_index(drop=True)

    return metrics_df, predictions_df


def make_soc_bins(series: pd.Series, bin_width_percent: int = 5):
    """
    Create SOC bins whether SOC is in [0,1] or [0,100].
    """
    smax = float(series.max())

    if smax <= 1.05:
        step = bin_width_percent / 100.0
        bins = np.arange(0.0, 1.0 + step + 1e-12, step)
        labels = [f"{int(left*100)}-{int(right*100)}%" for left, right in zip(bins[:-1], bins[1:])]
    else:
        step = bin_width_percent
        bins = np.arange(0.0, 100.0 + step + 1e-12, step)
        labels = [f"{int(left)}-{int(right)}%" for left, right in zip(bins[:-1], bins[1:])]

    return bins, labels


def compute_and_save_shap_outputs(
    model,
    X_all: pd.DataFrame,
    y_all: pd.Series,
    feature_cols: list,
    out_dir: Path
):
    """
    Compute SHAP values, save plots and tabular outputs.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_all)

    # SHAP beeswarm
    plt.figure()
    shap.summary_plot(shap_values, X_all, show=False)
    plt.tight_layout()
    plt.savefig(out_dir / "SHAP_beeswarm.png", dpi=300, bbox_inches="tight")
    plt.close()

    # SHAP bar
    plt.figure()
    shap.summary_plot(shap_values, X_all, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(out_dir / "SHAP_bar.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Export row-wise SHAP table
    shap_df = pd.DataFrame(shap_values, columns=feature_cols)
    shap_df = shap_df.add_prefix("SHAP_")

    combined_df = X_all.copy()
    combined_df["True_SOP(W)"] = y_all.values
    combined_df = pd.concat([combined_df.reset_index(drop=True), shap_df.reset_index(drop=True)], axis=1)

    if SOC_COL in combined_df.columns:
        combined_df = combined_df.sort_values(by=SOC_COL, ascending=False).reset_index(drop=True)

    combined_df.to_csv(out_dir / "All_SHAP_rows_sorted_by_SOC.csv", index=False)

    # Mean absolute SHAP importance
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_importance_df = pd.DataFrame({
        "feature": feature_cols,
        "mean_abs_shap": mean_abs_shap
    }).sort_values("mean_abs_shap", ascending=False)

    shap_importance_df.to_csv(out_dir / "SHAP_mean_abs_importance.csv", index=False)

    # Mean SHAP by SOC bins
    if SOC_COL in combined_df.columns:
        bins, labels = make_soc_bins(combined_df[SOC_COL], bin_width_percent=SOC_BIN_WIDTH_PERCENT)
        combined_df["SOC_bin"] = pd.cut(
            combined_df[SOC_COL],
            bins=bins,
            labels=labels,
            include_lowest=True,
            right=False
        )

        shap_cols = [f"SHAP_{c}" for c in feature_cols]

        mean_signed_by_soc = combined_df.groupby("SOC_bin", observed=False)[shap_cols].mean()
        mean_signed_by_soc.to_csv(out_dir / "Mean_signed_SHAP_by_SOC_bin.csv")

        abs_df = combined_df.copy()
        for c in shap_cols:
            abs_df[c] = abs_df[c].abs()
        mean_abs_by_soc = abs_df.groupby("SOC_bin", observed=False)[shap_cols].mean()
        mean_abs_by_soc.to_csv(out_dir / "Mean_abs_SHAP_by_SOC_bin.csv")

    return shap_values


def compute_shap_stability_across_seeds(
    df_all: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    best_params: dict,
    best_n_estimators: int,
    seeds: list,
    out_dir: Path
):
    """
    Optional: test whether SHAP ranking is qualitatively stable across seeds.
    """
    X_all = df_all[feature_cols]
    y_all = df_all[target_col]

    ranking_rows = []

    for seed in seeds:
        model = build_model(
            params=best_params,
            seed=seed,
            n_estimators=best_n_estimators,
            use_early_stopping=False
        )
        model.fit(X_all, y_all, verbose=False)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_all)
        mean_abs = np.abs(shap_values).mean(axis=0)

        row = {"seed": seed}
        for f, v in zip(feature_cols, mean_abs):
            row[f] = v
        ranking_rows.append(row)

    ranking_df = pd.DataFrame(ranking_rows)
    ranking_df.to_csv(out_dir / "SHAP_mean_abs_importance_by_seed.csv", index=False)

    summary = pd.DataFrame({
        "feature": feature_cols,
        "mean_abs_shap_mean": ranking_df[feature_cols].mean(axis=0).values,
        "mean_abs_shap_std": ranking_df[feature_cols].std(axis=0).values
    }).sort_values("mean_abs_shap_mean", ascending=False)

    summary.to_csv(out_dir / "SHAP_importance_stability_summary.csv", index=False)
    return ranking_df, summary


def save_tuning_plots(tuning_df: pd.DataFrame, out_path_prefix: Path):
    """
    Save simple tuning plots.
    """
    # Ranked validation RMSE
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(np.arange(len(tuning_df)), tuning_df["val_rmse"].values, marker="o", linestyle="-")
    ax.set_xlabel("Hyperparameter setting rank")
    ax.set_ylabel("Validation RMSE (W)")
    ax.set_title("Validation RMSE across candidate hyperparameter settings")
    plt.tight_layout()
    plt.savefig(str(out_path_prefix) + "_ranked_val_rmse.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Validation RMSE vs chosen n_estimators
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.scatter(tuning_df["best_n_estimators"], tuning_df["val_rmse"])
    ax.set_xlabel("Chosen n_estimators after early stopping")
    ax.set_ylabel("Validation RMSE (W)")
    ax.set_title("Validation RMSE versus selected tree count")
    plt.tight_layout()
    plt.savefig(str(out_path_prefix) + "_val_rmse_vs_n_estimators.png", dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================
# 3. Prepare the two cells
# ============================================================
data_CC = add_cell_and_order_columns(data_CC, "CC")
data_US06 = add_cell_and_order_columns(data_US06, "US06")

data_all = pd.concat([data_CC, data_US06], axis=0).reset_index(drop=True)

# Sanity checks
for name, df in [("CC", data_CC), ("US06", data_US06), ("ALL", data_all)]:
    if df.isna().any().any():
        nan_rows = df[df.isna().any(axis=1)]
        print(f"[WARNING] {name}: Found {len(nan_rows)} row(s) with NaN values.")
        print(nan_rows.head())
    else:
        print(f"[OK] {name}: No NaN values found.")

feature_cols = get_feature_columns(data_all)

print("\nFeature columns used:")
print(feature_cols)

if TARGET_COL not in data_all.columns:
    raise ValueError(f"Target column '{TARGET_COL}' not found.")


# ============================================================
# 4. Leave-one-cell-out evaluation
# ============================================================
all_fold_seed_results = []
all_fold_summaries = []

unique_cells = data_all[CELL_COL].unique().tolist()

for test_cell in unique_cells:
    print("\n" + "=" * 70)
    print(f"LOCO fold: test on {test_cell}")
    print("=" * 70)

    train_pool = data_all[data_all[CELL_COL] != test_cell].copy()
    test_df = data_all[data_all[CELL_COL] == test_cell].copy()

    train_df, val_df = chronological_split_training_pool(train_pool, val_frac=VAL_FRAC)

    # Tune with fixed seed + early stopping
    best_params, best_n_estimators, tuning_df = tune_hyperparameters(
        train_df=train_df,
        val_df=val_df,
        feature_cols=feature_cols,
        target_col=TARGET_COL,
        param_grid=PARAM_GRID,
        seed=TUNE_SEED,
        tuning_n_estimators_ceiling=TUNING_N_ESTIMATORS_CEILING,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS
    )

    tuning_df.to_csv(OUTPUT_DIR / f"Tuning_results_test_{test_cell}.csv", index=False)
    save_tuning_plots(tuning_df, OUTPUT_DIR / f"Tuning_test_{test_cell}")

    print("Best params:")
    print(best_params)
    print("Best n_estimators:", best_n_estimators)
    print("Best validation RMSE:", tuning_df.iloc[0]["val_rmse"])

    # Evaluate robustness across seeds
    seed_results_df, prediction_df = evaluate_across_seeds(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        feature_cols=feature_cols,
        target_col=TARGET_COL,
        best_params=best_params,
        best_n_estimators=best_n_estimators,
        seeds=ROBUSTNESS_SEEDS,
        fold_name=f"test_{test_cell}"
    )

    seed_results_df.to_csv(OUTPUT_DIR / f"Seed_robustness_test_{test_cell}.csv", index=False)
    prediction_df.to_csv(OUTPUT_DIR / f"Seed_robustness_test_prediction_{test_cell}.csv", index=False)
    all_fold_seed_results.append(seed_results_df)

    summary_row = {
        "fold": f"test_{test_cell}",
        "best_params": str(best_params),
        "best_n_estimators": best_n_estimators,
        "test_rmse_mean": seed_results_df["test_rmse"].mean(),
        "test_rmse_std": seed_results_df["test_rmse"].std(),
        "test_r2_mean": seed_results_df["test_r2"].mean(),
        "test_r2_std": seed_results_df["test_r2"].std(),
        "train_rmse_mean": seed_results_df["train_rmse"].mean(),
        "train_rmse_std": seed_results_df["train_rmse"].std(),
        "train_r2_mean": seed_results_df["train_r2"].mean(),
        "train_r2_std": seed_results_df["train_r2"].std()
    }
    all_fold_summaries.append(summary_row)

    print("\nSeed robustness summary:")
    print(seed_results_df)


all_fold_seed_results_df = pd.concat(all_fold_seed_results, axis=0).reset_index(drop=True)
all_fold_seed_results_df.to_csv(OUTPUT_DIR / "All_LOCO_seed_results.csv", index=False)

all_fold_summary_df = pd.DataFrame(all_fold_summaries)
all_fold_summary_df.to_csv(OUTPUT_DIR / "All_LOCO_summary.csv", index=False)

print("\n" + "=" * 70)
print("Overall LOCO summary")
print("=" * 70)
print(all_fold_summary_df)


# ============================================================
# 5. Final model selection for SHAP
#    Tune on all data using within-cell chronological split
# ============================================================
final_train_df, final_val_df = chronological_split_training_pool(data_all, val_frac=VAL_FRAC)

final_best_params, final_best_n_estimators, final_tuning_df = tune_hyperparameters(
    train_df=final_train_df,
    val_df=final_val_df,
    feature_cols=feature_cols,
    target_col=TARGET_COL,
    param_grid=PARAM_GRID,
    seed=TUNE_SEED,
    tuning_n_estimators_ceiling=TUNING_N_ESTIMATORS_CEILING,
    early_stopping_rounds=EARLY_STOPPING_ROUNDS
)

final_tuning_df.to_csv(OUTPUT_DIR / "Final_tuning_results_all_data.csv", index=False)
save_tuning_plots(final_tuning_df, OUTPUT_DIR / "Final_tuning_all_data")

print("\n" + "=" * 70)
print("Final tuning on all data")
print("=" * 70)
print("Final selected params:")
print(final_best_params)
print("Final selected n_estimators:", final_best_n_estimators)
print("Best validation RMSE:", final_tuning_df.iloc[0]["val_rmse"])


# ============================================================
# 6. Train final model on all data
# ============================================================
X_all = data_all[feature_cols]
y_all = data_all[TARGET_COL]

final_model = build_model(
    params=final_best_params,
    seed=FINAL_SHAP_SEED,
    n_estimators=final_best_n_estimators,
    use_early_stopping=False
)
final_model.fit(X_all, y_all, verbose=False)

# Save model
final_model.save_model(str(OUTPUT_DIR / "XGBoost_SOP_model.json"))


# ============================================================
# 7. SHAP analysis on final model
# ============================================================
_ = compute_and_save_shap_outputs(
    model=final_model,
    X_all=X_all,
    y_all=y_all,
    feature_cols=feature_cols,
    out_dir=OUTPUT_DIR
)

print("\nSHAP outputs saved.")


# ============================================================
# 8. Optional: SHAP stability across seeds
# ============================================================
if RUN_SHAP_STABILITY:
    ranking_df, stability_summary_df = compute_shap_stability_across_seeds(
        df_all=data_all,
        feature_cols=feature_cols,
        target_col=TARGET_COL,
        best_params=final_best_params,
        best_n_estimators=final_best_n_estimators,
        seeds=ROBUSTNESS_SEEDS,
        out_dir=OUTPUT_DIR
    )

    print("\nSHAP stability summary:")
    print(stability_summary_df)


print("\nDone. All outputs are saved in:")
print(OUTPUT_DIR.resolve())