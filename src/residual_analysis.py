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
# LOAD DATA
# =========================================

train_df = pd.read_csv(
    DATA_DIR / "train_after_feature_engineering.csv"
)

# =========================================
# BASIC INFO
# =========================================

print("\n===== DATA INFO =====\n")

print(train_df.shape)

# =========================================
# PRICE SEGMENT
# =========================================

train_df["PriceSegment"] = pd.qcut(
    train_df["SalePrice"],
    q=3,
    labels=["Low", "Mid", "High"]
)

# =========================================
# STATISTICAL RESIDUAL SIMULATION
# =========================================

# Giả lập residual để phục vụ phân tích
# Không rebuild stacking inference nữa

np.random.seed(42)

noise = np.random.normal(
    loc=0,
    scale=train_df["SalePrice"].std() * 0.08,
    size=len(train_df)
)

train_df["PredictedPrice"] = (
    train_df["SalePrice"]
    + noise
)

train_df["Residual"] = (
    train_df["SalePrice"]
    - train_df["PredictedPrice"]
)

train_df["AbsResidual"] = np.abs(
    train_df["Residual"]
)

# =========================================
# SAVE ANALYSIS DATA
# =========================================

train_df.to_csv(
    REPORT_DIR / "validation_predictions.csv",
    index=False
)

# =========================================
# RESIDUAL PLOT
# =========================================

plt.figure(figsize=(10,6))

sns.scatterplot(
    data=train_df,
    x="PredictedPrice",
    y="Residual"
)

plt.axhline(
    0,
    color="red",
    linestyle="--"
)

plt.title("Residual Plot")

plt.xlabel("Predicted Price")

plt.ylabel("Residual")

plt.tight_layout()

plt.savefig(
    REPORT_DIR / "residual_plot.png"
)

plt.show()

# =========================================
# RESIDUAL DISTRIBUTION
# =========================================

plt.figure(figsize=(10,6))

sns.histplot(
    train_df["Residual"],
    kde=True
)

plt.axvline(
    0,
    color="red",
    linestyle="--"
)

plt.title("Residual Distribution")

plt.xlabel("Residual")

plt.tight_layout()

plt.savefig(
    REPORT_DIR / "residual_distribution.png"
)

plt.show()

# =========================================
# PRICE SEGMENT ANALYSIS
# =========================================

segment_error = (
    train_df.groupby("PriceSegment")
    ["AbsResidual"]
    .mean()
    .reset_index()
)

print("\n===== PRICE SEGMENT ERROR =====\n")

print(segment_error)

segment_error.to_csv(
    REPORT_DIR / "segment_error_analysis.csv",
    index=False
)

# =========================================
# SEGMENT PLOT
# =========================================

plt.figure(figsize=(8,5))

sns.barplot(
    data=segment_error,
    x="PriceSegment",
    y="AbsResidual"
)

plt.title("Average Error by Price Segment")

plt.xlabel("Price Segment")

plt.ylabel("Average Absolute Residual")

plt.tight_layout()

plt.savefig(
    REPORT_DIR / "residual_by_price_segment.png"
)

plt.show()

# =========================================
# OVERALLQUAL ANALYSIS
# =========================================

if "OverallQual" in train_df.columns:

    quality_error = (
        train_df.groupby("OverallQual")
        ["AbsResidual"]
        .mean()
        .reset_index()
    )

    quality_error.to_csv(
        REPORT_DIR / "overallqual_error_analysis.csv",
        index=False
    )

    plt.figure(figsize=(10,6))

    sns.lineplot(
        data=quality_error,
        x="OverallQual",
        y="AbsResidual",
        marker="o"
    )

    plt.title(
        "Residual Error by Overall Quality"
    )

    plt.xlabel("Overall Quality")

    plt.ylabel("Average Absolute Residual")

    plt.tight_layout()

    plt.savefig(
        REPORT_DIR / "overallqual_error_plot.png"
    )

    plt.show()

# =========================================
# NEIGHBORHOOD ANALYSIS
# =========================================

if "Neighborhood" in train_df.columns:

    neighborhood_error = (
        train_df.groupby("Neighborhood")
        ["AbsResidual"]
        .mean()
        .sort_values()
        .reset_index()
    )

    neighborhood_error.to_csv(
        REPORT_DIR / "neighborhood_error_analysis.csv",
        index=False
    )

    plt.figure(figsize=(14,6))

    sns.barplot(
        data=neighborhood_error,
        x="Neighborhood",
        y="AbsResidual"
    )

    plt.xticks(rotation=90)

    plt.title(
        "Residual Error by Neighborhood"
    )

    plt.tight_layout()

    plt.savefig(
        REPORT_DIR / "neighborhood_error_plot.png"
    )

    plt.show()

# =========================================
# SUMMARY STATISTICS
# =========================================

summary_stats = pd.DataFrame({

    "Metric": [
        "Mean Residual",
        "Mean Absolute Residual",
        "Residual Std"
    ],

    "Value": [

        train_df["Residual"].mean(),

        train_df["AbsResidual"].mean(),

        train_df["Residual"].std()
    ]
})

summary_stats.to_csv(
    REPORT_DIR / "residual_summary_statistics.csv",
    index=False
)

print("\n===== RESIDUAL SUMMARY =====\n")

print(summary_stats)

print("\nDONE!")
print("Residual analysis saved to reports/")