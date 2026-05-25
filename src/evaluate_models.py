import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

# =========================================
# BASE DIRECTORY
# =========================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
REPORT_DIR = BASE_DIR / "reports"

REPORT_DIR.mkdir(exist_ok=True)

# =========================================
# LOAD RESULTS
# =========================================

baseline = pd.read_csv(
    DATA_DIR / "baseline_results.csv"
)

boosting = pd.read_csv(
    DATA_DIR / "boosting_results.csv"
)

stacking = pd.read_csv(
    DATA_DIR / "stacking_results.csv"
)

# =========================================
# STANDARDIZE COLUMN NAMES
# =========================================

baseline = baseline.rename(columns={
    "CV RMSE Mean": "CV_RMSE",
    "CV RMSE Std": "CV_STD"
})

boosting = boosting.rename(columns={
    "CV RMSE": "CV_RMSE",
    "CV Std": "CV_STD"
})

stacking = stacking.rename(columns={
    "CV RMSE": "CV_RMSE"
})

# stacking không có std thật
stacking["CV_STD"] = np.nan

# =========================================
# ADD GROUP LABEL
# =========================================

baseline["Group"] = "Baseline"

boosting["Group"] = "Boosting"

stacking["Group"] = "Stacking"

# =========================================
# COMBINE RESULTS
# =========================================

all_results = pd.concat(
    [baseline, boosting, stacking],
    ignore_index=True
)

# =========================================
# SORT BY RMSE
# =========================================

all_results = all_results.sort_values(
    by="CV_RMSE"
)

# =========================================
# PRINT RESULTS
# =========================================

print("\n===== MODEL COMPARISON =====\n")

print(all_results)

# =========================================
# SAVE FINAL TABLE
# =========================================

all_results.to_csv(
    REPORT_DIR / "final_model_comparison.csv",
    index=False
)

# =========================================
# RMSE COMPARISON PLOT
# =========================================

plt.figure(figsize=(12,6))

sns.barplot(
    data=all_results,
    x="Model",
    y="CV_RMSE",
    hue="Group"
)

plt.title("Model CV RMSE Comparison")

plt.xlabel("Model")

plt.ylabel("CV RMSE")

plt.xticks(rotation=15)

plt.tight_layout()

plt.savefig(
    REPORT_DIR / "model_rmse_comparison.png"
)

plt.show()

# =========================================
# REMOVE STACKING FOR STABILITY
# =========================================

stability_df = all_results.dropna(
    subset=["CV_STD"]
)

# =========================================
# STABILITY PLOT
# =========================================

plt.figure(figsize=(12,6))

sns.barplot(
    data=stability_df,
    x="Model",
    y="CV_STD",
    hue="Group"
)

plt.title("Model Stability Comparison")

plt.xlabel("Model")

plt.ylabel("CV RMSE Std")

plt.xticks(rotation=15)

plt.tight_layout()

plt.savefig(
    REPORT_DIR / "model_std_comparison.png"
)

plt.show()

# =========================================
# BEST MODEL
# =========================================

best_model = all_results.iloc[0]

print("\n===== BEST MODEL =====\n")

print(best_model)

# =========================================
# SAVE BEST MODEL
# =========================================

best_model.to_frame().to_csv(
    REPORT_DIR / "best_model_summary.csv"
)

print("\nDONE!")
print("Evaluation results saved to reports/")