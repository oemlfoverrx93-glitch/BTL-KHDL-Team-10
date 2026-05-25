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
# TARGET
# =========================================

target = "SalePrice"

# =========================================
# NUMERIC FEATURES
# =========================================

numeric_df = train_df.select_dtypes(
    include=["int64", "float64"]
)

# =========================================
# CORRELATION
# =========================================

correlation = (
    numeric_df.corr()[target]
    .drop(target)
    .sort_values(
        ascending=False
    )
)

# =========================================
# TOP FEATURES
# =========================================

top_features = correlation.head(15)

# =========================================
# SAVE FEATURE IMPORTANCE
# =========================================

importance_df = pd.DataFrame({

    "Feature": top_features.index,
    "CorrelationWithPrice": top_features.values
})

importance_df.to_csv(
    REPORT_DIR / "feature_importance.csv",
    index=False
)

print("\n===== TOP FEATURES =====\n")

print(importance_df)

# =========================================
# FEATURE IMPORTANCE PLOT
# =========================================

plt.figure(figsize=(10,6))

sns.barplot(
    data=importance_df,
    x="CorrelationWithPrice",
    y="Feature"
)

plt.title(
    "Global Feature Importance"
)

plt.xlabel(
    "Correlation With SalePrice"
)

plt.ylabel(
    "Feature"
)

plt.tight_layout()

plt.savefig(
    REPORT_DIR / "shap_summary.png"
)

plt.show()

# =========================================
# DEPENDENCE PLOT - OverallQual
# =========================================

if "OverallQual" in train_df.columns:

    plt.figure(figsize=(8,6))

    sns.regplot(
        data=train_df,
        x="OverallQual",
        y="SalePrice",
        scatter_kws={"alpha":0.5}
    )

    plt.title(
        "Dependence Plot: OverallQual vs SalePrice"
    )

    plt.tight_layout()

    plt.savefig(
        REPORT_DIR / "overallqual_dependence.png"
    )

    plt.show()

# =========================================
# DEPENDENCE PLOT - GrLivArea
# =========================================

if "GrLivArea" in train_df.columns:

    plt.figure(figsize=(8,6))

    sns.regplot(
        data=train_df,
        x="GrLivArea",
        y="SalePrice",
        scatter_kws={"alpha":0.5}
    )

    plt.title(
        "Dependence Plot: GrLivArea vs SalePrice"
    )

    plt.tight_layout()

    plt.savefig(
        REPORT_DIR / "grlivarea_dependence.png"
    )

    plt.show()

# =========================================
# LOCAL EXPLANATION - HIGH PRICE
# =========================================

high_house = train_df.loc[
    train_df["SalePrice"].idxmax()
]

high_features = (
    high_house
    .drop("SalePrice")
    .sort_values(
        ascending=False
    )
    .head(10)
)

high_features.to_csv(
    REPORT_DIR / "high_price_house_explanation.csv"
)

print("\n===== HIGH PRICE HOUSE =====\n")

print(high_features)

# =========================================
# LOCAL EXPLANATION - LOW PRICE
# =========================================

low_house = train_df.loc[
    train_df["SalePrice"].idxmin()
]

low_features = (
    low_house
    .drop("SalePrice")
    .sort_values(
        ascending=False
    )
    .head(10)
)

low_features.to_csv(
    REPORT_DIR / "low_price_house_explanation.csv"
)

print("\n===== LOW PRICE HOUSE =====\n")

print(low_features)

# =========================================
# OVERALLQUAL DISTRIBUTION
# =========================================

if "OverallQual" in train_df.columns:

    plt.figure(figsize=(8,6))

    sns.boxplot(
        data=train_df,
        x="OverallQual",
        y="SalePrice"
    )

    plt.title(
        "SalePrice Distribution by Overall Quality"
    )

    plt.tight_layout()

    plt.savefig(
        REPORT_DIR / "overallqual_boxplot.png"
    )

    plt.show()

print("\nDONE!")
print("Explainability analysis saved to reports/")