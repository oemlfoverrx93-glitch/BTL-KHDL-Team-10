import argparse
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from pandas.api.types import is_numeric_dtype
from scipy.stats import jarque_bera, kurtosis, pearsonr, probplot, spearmanr
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sqlalchemy import create_engine


sns.set_theme(style="whitegrid")


FEATURE_GROUPS: Dict[str, List[str]] = {
    "size_area": ["LotArea", "GrLivArea", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF"],
    "quality_condition": ["OverallQual", "OverallCond", "KitchenQual", "ExterQual", "HeatingQC"],
    "house_age": ["YearBuilt", "YearRemodAdd", "GarageYrBlt"],
    "location": ["Neighborhood", "Condition1", "Street"],
    "utilities": [
        "GarageCars",
        "GarageArea",
        "FullBath",
        "HalfBath",
        "BedroomAbvGr",
        "TotRmsAbvGrd",
        "Fireplaces",
        "PoolArea",
    ],
    "transaction": ["SaleType", "SaleCondition", "YrSold", "MoSold"],
}


FEATURE_TYPES: Dict[str, List[str]] = {
    "Numerical Continuous": [
        "LotArea", "GrLivArea", "TotalBsmtSF", "GarageArea", "MasVnrArea", "LotFrontage"
    ],
    "Numerical Count": [
        "FullBath", "HalfBath", "BedroomAbvGr", "TotRmsAbvGrd", "GarageCars", "Fireplaces"
    ],
    "Ordinal Categorical": [
        "ExterQual", "ExterCond", "KitchenQual", "HeatingQC", "BsmtQual", "BsmtCond", "GarageQual", "GarageCond", "FireplaceQu"
    ],
    "Nominal Categorical": [
        "Neighborhood", "MSZoning", "BldgType", "HouseStyle", "RoofStyle", "Exterior1st", "Exterior2nd", "SaleType", "SaleCondition"
    ],
    "Binary": [
        "CentralAir", "Street"
    ],
    "Special Coded Category": [
        "MSSubClass", "MoSold", "YrSold"
    ],
    "Target": [
        "SalePrice"
    ],
    "ID": [
        "Id"
    ]
}

PROCESSING_GUIDE: Dict[str, str] = {
    "Numerical Continuous": "Pearson correlation, StandardScaler, dùng trực tiếp cho regression",
    "Numerical Count": "Spearman correlation, có thể scaling nhẹ hoặc giữ nguyên",
    "Ordinal Categorical": "OrdinalEncoder (Ex=5, Gd=4, TA=3, Fa=2, Po=1, None=0)",
    "Nominal Categorical": "One-Hot Encoding (pd.get_dummies / OneHotEncoder)",
    "Binary": "LabelEncoder hoặc map thủ công (Y=1, N=0; Pave=1, Grvl=0)",
    "Special Coded Category": "Chuyển sang str rồi One-Hot Encoding (KHÔNG dùng như số)",
    "Target": "Log1p transform để giảm skewness",
    "ID": "Loại bỏ khỏi tập đặc trưng (drop trước khi train)",
}


NA_SEMANTIC_MAP: Dict[str, str] = {
    "PoolQC": "No Pool",
    "Alley": "No alley access",
    "Fence": "No Fence",
    "FireplaceQu": "No Fireplace",
    "GarageType": "No Garage",
    "GarageFinish": "No Garage",
    "GarageQual": "No Garage",
    "GarageCond": "No Garage",
    "BsmtQual": "No Basement",
    "BsmtCond": "No Basement",
    "BsmtExposure": "No Basement",
    "BsmtFinType1": "No Basement",
    "BsmtFinType2": "No Basement",
    "MiscFeature": "No Misc Feature",
}


NO_FEATURE_COLS = [
    "PoolQC",
    "MiscFeature",
    "Alley",
    "Fence",
    "FireplaceQu",
    "GarageType",
    "GarageFinish",
    "GarageQual",
    "GarageCond",
    "BsmtQual",
    "BsmtCond",
    "BsmtExposure",
    "BsmtFinType1",
    "BsmtFinType2",
]


ZERO_FILL_COLS = [
    "GarageArea",
    "GarageCars",
    "BsmtFinSF1",
    "BsmtFinSF2",
    "BsmtUnfSF",
    "TotalBsmtSF",
    "BsmtFullBath",
    "BsmtHalfBath",
]


DOMAIN_MEDIAN_COLS = ["LotFrontage"]
DOMAIN_MODE_COLS = ["Electrical"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run domain-aware EDA, outlier sensitivity, and residual analysis for Ames Housing."
    )
    parser.add_argument(
        "--db-url",
        default=os.getenv("AMES_DB_URL"),
        help=(
            "SQLAlchemy database URL. If omitted, script uses CSV fallback. "
            "Example: mysql+pymysql://user:pass@localhost:3306/ames_housing"
        ),
    )
    parser.add_argument(
        "--table",
        default="ames_train",
        help="SQL train table to read when --db-url is provided.",
    )
    parser.add_argument(
        "--query",
        default=None,
        help="Optional SQL query for train data. Overrides --table.",
    )
    parser.add_argument(
        "--test-table",
        default="ames_test",
        help="SQL test table to read when --db-url is provided.",
    )
    parser.add_argument(
        "--test-query",
        default=None,
        help="Optional SQL query for test data. Overrides --test-table.",
    )
    parser.add_argument(
        "--csv-path",
        default="data/raw/train.csv",
        help="Fallback CSV path for train data.",
    )
    parser.add_argument(
        "--test-csv-path",
        default="data/raw/test.csv",
        help="Fallback CSV path for test data.",
    )
    parser.add_argument(
        "--data-description-path",
        default="data/raw/data_description.txt",
        help="Path to Ames data_description.txt for domain reference table.",
    )
    parser.add_argument(
        "--output-dir",
        default="reports",
        help="Directory to save figures, tables, and summary.",
    )
    parser.add_argument(
        "--keep-old-output",
        action="store_true",
        help="Keep existing files in output directory instead of cleaning old outputs first.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Validation split size for model analysis.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--outlier-grliv-threshold",
        type=float,
        default=4000.0,
        help="Domain threshold for GrLivArea outlier rule.",
    )
    parser.add_argument(
        "--outlier-saleprice-threshold",
        type=float,
        default=300000.0,
        help="Domain threshold for SalePrice outlier rule.",
    )
    parser.add_argument(
        "--drop-domain-outliers",
        action="store_true",
        help="If set, downstream EDA/model analysis uses outlier-removed train set.",
    )
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Disable Plotly interactive HTML outputs.",
    )
    return parser.parse_args()


def ensure_dirs(base_output: Path) -> Tuple[Path, Path, Path, Path]:
    figures_dir = base_output / "figures"
    tables_dir = base_output / "tables"
    interactive_dir = base_output / "interactive"
    post_eda_dir = base_output / "eda_after_preprocessing"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    interactive_dir.mkdir(parents=True, exist_ok=True)
    post_eda_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir, tables_dir, interactive_dir, post_eda_dir


def clean_output_dir(base_output: Path, keep_old_output: bool) -> None:
    if keep_old_output:
        return
    if not base_output.exists():
        return
    for child in base_output.iterdir():
        try:
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
        except PermissionError as exc:
            print(f"[WARN] Could not remove old output item {child}: {exc}")


def load_single_dataset(
    db_url: str | None,
    table: str,
    query: str | None,
    csv_path: Path,
    label: str,
) -> Tuple[pd.DataFrame, str]:
    if db_url:
        try:
            engine = create_engine(db_url)
            sql = query if query else f"SELECT * FROM {table}"
            df = pd.read_sql(sql, engine)
            return df, f"SQL:{label} ({sql})"
        except Exception as exc:
            print(f"[WARN] SQL load failed for {label}: {exc}")
            print(f"[INFO] Falling back to CSV for {label}: {csv_path}")

    if not csv_path.exists():
        raise FileNotFoundError(f"Missing fallback CSV for {label}: {csv_path}")

    df = pd.read_csv(csv_path)
    return df, f"CSV:{label} ({csv_path})"


def load_train_test(args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    train_df, train_source = load_single_dataset(
        db_url=args.db_url,
        table=args.table,
        query=args.query,
        csv_path=Path(args.csv_path),
        label="train",
    )
    test_df, test_source = load_single_dataset(
        db_url=args.db_url,
        table=args.test_table,
        query=args.test_query,
        csv_path=Path(args.test_csv_path),
        label="test",
    )
    return train_df, test_df, f"{train_source}; {test_source}"


def build_missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        return pd.DataFrame(columns=["Column", "MissingCount", "MissingPercent"])
    missing_percent = (missing / len(df)) * 100
    summary = pd.DataFrame(
        {
            "Column": missing.index,
            "MissingCount": missing.values,
            "MissingPercent": missing_percent.values,
        }
    )
    return summary


def plot_missing_percent(
    missing_df: pd.DataFrame, output_path: Path, title: str, top_n: int = 30
) -> None:
    plt.figure(figsize=(14, 6))
    if missing_df.empty:
        plt.text(0.5, 0.5, "No missing values", ha="center", va="center", fontsize=14)
        plt.axis("off")
    else:
        plot_df = missing_df.head(top_n)
        sns.barplot(data=plot_df, x="Column", y="MissingPercent")
        plt.xticks(rotation=90)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if {"TotalBsmtSF", "1stFlrSF", "2ndFlrSF"}.issubset(out.columns):
        out["TotalSF"] = out["TotalBsmtSF"] + out["1stFlrSF"] + out["2ndFlrSF"]
    if {"YrSold", "YearBuilt"}.issubset(out.columns):
        out["HouseAge"] = out["YrSold"] - out["YearBuilt"]
    if {"YrSold", "YearRemodAdd"}.issubset(out.columns):
        out["RemodAge"] = out["YrSold"] - out["YearRemodAdd"]
    if {"FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath"}.issubset(out.columns):
        out["TotalBath"] = (
            out["FullBath"] + 0.5 * out["HalfBath"] + out["BsmtFullBath"] + 0.5 * out["BsmtHalfBath"]
        )
    return out


def _safe_mode(series: pd.Series, default_value: object) -> object:
    mode_values = series.mode(dropna=True)
    if not mode_values.empty:
        return mode_values.iloc[0]
    return default_value


def _fill_in_both(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    col: str,
    fill_value: object,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if col in train_df.columns:
        train_df[col] = train_df[col].fillna(fill_value)
    if col in test_df.columns:
        test_df[col] = test_df[col].fillna(fill_value)
    return train_df, test_df


def apply_domain_missing_strategy(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    data_description_path: Path,
    figures_dir: Path,
    tables_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = train_df.copy()
    test = test_df.copy()

    missing_train_before = build_missing_summary(train)
    missing_test_before = build_missing_summary(test)
    missing_train_before.to_csv(tables_dir / "missing_train_before.csv", index=False)
    missing_test_before.to_csv(tables_dir / "missing_test_before.csv", index=False)
    missing_train_before.to_csv(tables_dir / "missing_summary_before.csv", index=False)
    plot_missing_percent(
        missing_train_before,
        figures_dir / "01_missing_train_before.png",
        "Missing Value Percentage (Train, Before)",
    )

    strategy_rows: List[Dict[str, object]] = []

    for col in NO_FEATURE_COLS:
        if col in train.columns or col in test.columns:
            train, test = _fill_in_both(train, test, col, "None")
            strategy_rows.append({"Column": col, "Strategy": "fillna('None')", "Reason": "No feature"})

    for col in ZERO_FILL_COLS:
        if col in train.columns or col in test.columns:
            train, test = _fill_in_both(train, test, col, 0)
            strategy_rows.append({"Column": col, "Strategy": "fillna(0)", "Reason": "No numeric feature"})

    for col in DOMAIN_MEDIAN_COLS:
        if col in train.columns:
            median_value = float(train[col].median())
            train, test = _fill_in_both(train, test, col, median_value)
            strategy_rows.append(
                {
                    "Column": col,
                    "Strategy": f"fillna(median={median_value:.4f})",
                    "Reason": "True missing numeric",
                }
            )

    for col in DOMAIN_MODE_COLS:
        if col in train.columns:
            mode_value = _safe_mode(train[col], "SBrkr")
            train, test = _fill_in_both(train, test, col, mode_value)
            strategy_rows.append(
                {
                    "Column": col,
                    "Strategy": f"fillna(mode={mode_value})",
                    "Reason": "True missing categorical",
                }
            )

    remaining_cols = [c for c in train.columns if train[c].isnull().any()]
    for col in remaining_cols:
        if is_numeric_dtype(train[col]):
            median_value = train[col].median()
            fill_value = 0.0 if pd.isna(median_value) else float(median_value)
            train, test = _fill_in_both(train, test, col, fill_value)
            strategy_rows.append(
                {
                    "Column": col,
                    "Strategy": f"fallback fillna(median={fill_value:.4f})",
                    "Reason": "Remaining numeric missing",
                }
            )
        else:
            fill_value = _safe_mode(train[col], "None")
            train, test = _fill_in_both(train, test, col, fill_value)
            strategy_rows.append(
                {
                    "Column": col,
                    "Strategy": f"fallback fillna(mode={fill_value})",
                    "Reason": "Remaining categorical missing",
                }
            )

    remaining_test_cols = [c for c in test.columns if test[c].isnull().any()]
    for col in remaining_test_cols:
        if is_numeric_dtype(test[col]):
            median_value = test[col].median()
            fill_value = 0.0 if pd.isna(median_value) else float(median_value)
            test[col] = test[col].fillna(fill_value)
            strategy_rows.append(
                {
                    "Column": col,
                    "Strategy": f"test-only fillna(median={fill_value:.4f})",
                    "Reason": "Column not available in train",
                }
            )
        else:
            fill_value = _safe_mode(test[col], "None")
            test[col] = test[col].fillna(fill_value)
            strategy_rows.append(
                {
                    "Column": col,
                    "Strategy": f"test-only fillna(mode={fill_value})",
                    "Reason": "Column not available in train",
                }
            )

    missing_train_after = build_missing_summary(train)
    missing_test_after = build_missing_summary(test)
    missing_train_after.to_csv(tables_dir / "missing_train_after.csv", index=False)
    missing_test_after.to_csv(tables_dir / "missing_test_after.csv", index=False)
    missing_train_after.to_csv(tables_dir / "missing_summary_after.csv", index=False)
    plot_missing_percent(
        missing_train_after,
        figures_dir / "02_missing_train_after.png",
        "Missing Value Percentage (Train, After)",
    )

    strategy_df = pd.DataFrame(strategy_rows).drop_duplicates(subset=["Column", "Strategy"])
    strategy_df.to_csv(tables_dir / "missing_strategy_applied.csv", index=False)

    save_feature_groups(train, tables_dir)
    save_feature_types(train, tables_dir)
    save_na_semantics_reference(train, data_description_path, tables_dir)

    return train, test


def save_feature_groups(train_df: pd.DataFrame, tables_dir: Path) -> None:
    rows: List[Dict[str, str]] = []
    for group_name, cols in FEATURE_GROUPS.items():
        for col in cols:
            rows.append(
                {
                    "Group": group_name,
                    "Feature": col,
                    "InTrainColumns": col in train_df.columns,
                }
            )
    pd.DataFrame(rows).to_csv(tables_dir / "feature_groups.csv", index=False)


def save_feature_types(train_df: pd.DataFrame, tables_dir: Path) -> None:
    rows: List[Dict[str, str]] = []
    for ftype, features in FEATURE_TYPES.items():
        guide = PROCESSING_GUIDE.get(ftype, "")
        for feat in features:
            rows.append(
                {
                    "Feature": feat,
                    "Type": ftype,
                    "Processing": guide,
                    "InTrain": str(feat in train_df.columns),
                }
            )
    pd.DataFrame(rows).to_csv(tables_dir / "feature_type_classification.csv", index=False)


def save_na_semantics_reference(
    train_df: pd.DataFrame, data_description_path: Path, tables_dir: Path
) -> None:
    description_text = ""
    if data_description_path.exists():
        description_text = data_description_path.read_text(encoding="utf-8", errors="ignore")

    rows: List[Dict[str, object]] = []
    for col, meaning in NA_SEMANTIC_MAP.items():
        rows.append(
            {
                "Column": col,
                "NA_Meaning": meaning,
                "InTrainColumns": col in train_df.columns,
                "MentionedInDataDescription": col in description_text,
            }
        )
    pd.DataFrame(rows).to_csv(tables_dir / "na_semantics_reference.csv", index=False)


def detect_domain_outliers(
    df: pd.DataFrame, grliv_threshold: float, saleprice_threshold: float
) -> pd.Series:
    if "GrLivArea" not in df.columns or "SalePrice" not in df.columns:
        return pd.Series(False, index=df.index)
    return (df["GrLivArea"] > grliv_threshold) & (df["SalePrice"] < saleprice_threshold)


def save_outlier_scatter(
    df: pd.DataFrame, outlier_mask: pd.Series, figures_dir: Path, title_suffix: str = ""
) -> None:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=df["GrLivArea"], y=df["SalePrice"], alpha=0.7, label="Inlier")
    if outlier_mask.any():
        outlier_df = df.loc[outlier_mask]
        sns.scatterplot(
            x=outlier_df["GrLivArea"],
            y=outlier_df["SalePrice"],
            color="red",
            s=80,
            label="Domain outlier",
        )
    plt.title(f"GrLivArea vs SalePrice{title_suffix}")
    plt.tight_layout()
    plt.savefig(figures_dir / "03_outlier_grlivarea_saleprice.png", dpi=150)
    plt.savefig(figures_dir / "outlier_grlivarea_saleprice.png", dpi=150)
    plt.close()


def prepare_for_model(df: pd.DataFrame, target_col: str = "SalePrice") -> pd.DataFrame:
    model_df = df.copy()
    for col in model_df.columns:
        if col == target_col:
            continue
        if not is_numeric_dtype(model_df[col]):
            model_df[col] = model_df[col].fillna("Missing").astype(str)
            model_df[col] = LabelEncoder().fit_transform(model_df[col])
        else:
            if model_df[col].isna().any():
                model_df[col] = model_df[col].fillna(model_df[col].median())
    return model_df


def run_outlier_sensitivity_analysis(
    train_df: pd.DataFrame,
    outlier_mask: pd.Series,
    tables_dir: Path,
    figures_dir: Path,
    test_size: float,
    random_state: int,
) -> pd.DataFrame:
    baseline_features = [c for c in ["OverallQual", "GrLivArea", "GarageCars"] if c in train_df.columns]
    if len(baseline_features) < 2:
        sensitivity_df = pd.DataFrame(
            columns=["Scenario", "Rows", "MAE", "RMSE", "R2", "OutlierCount"]
        )
        sensitivity_df.to_csv(tables_dir / "outlier_sensitivity_metrics.csv", index=False)
        return sensitivity_df

    scenarios = {
        "with_outliers": train_df.copy(),
        "without_outliers": train_df.loc[~outlier_mask].copy(),
    }
    metrics_rows: List[Dict[str, float | str | int]] = []

    for scenario_name, scenario_df in scenarios.items():
        X = scenario_df[baseline_features]
        y = scenario_df["SalePrice"]
        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        model = LinearRegression()
        model.fit(X_train, y_train)
        preds = model.predict(X_valid)
        mae = mean_absolute_error(y_valid, preds)
        rmse = np.sqrt(mean_squared_error(y_valid, preds))
        r2 = r2_score(y_valid, preds)
        metrics_rows.append(
            {
                "Scenario": scenario_name,
                "Rows": int(scenario_df.shape[0]),
                "MAE": float(mae),
                "RMSE": float(rmse),
                "R2": float(r2),
                "OutlierCount": int(outlier_mask.sum() if scenario_name == "with_outliers" else 0),
            }
        )

    sensitivity_df = pd.DataFrame(metrics_rows)
    sensitivity_df.to_csv(tables_dir / "outlier_sensitivity_metrics.csv", index=False)

    plt.figure(figsize=(8, 5))
    sns.barplot(data=sensitivity_df, x="Scenario", y="RMSE")
    plt.title("Outlier Sensitivity Analysis (RMSE)")
    plt.tight_layout()
    plt.savefig(figures_dir / "04_outlier_sensitivity_rmse.png", dpi=150)
    plt.close()

    return sensitivity_df


def save_saleprice_distributions(
    df: pd.DataFrame, figures_dir: Path, tables_dir: Path
) -> Dict[str, float]:
    price = df["SalePrice"].dropna()
    log_price = np.log1p(price)

    plt.figure(figsize=(10, 6))
    sns.histplot(price, kde=True)
    plt.title("SalePrice Distribution")
    plt.tight_layout()
    plt.savefig(figures_dir / "05_saleprice_distribution.png", dpi=150)
    plt.savefig(figures_dir / "saleprice_before_log.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.histplot(log_price, kde=True, color="teal")
    plt.title("Log Transformed SalePrice")
    plt.tight_layout()
    plt.savefig(figures_dir / "06_saleprice_log_distribution.png", dpi=150)
    plt.savefig(figures_dir / "saleprice_after_log.png", dpi=150)
    plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    probplot(price, dist="norm", plot=ax[0])
    ax[0].set_title("QQ Plot: SalePrice")
    probplot(log_price, dist="norm", plot=ax[1])
    ax[1].set_title("QQ Plot: log1p(SalePrice)")
    plt.tight_layout()
    plt.savefig(figures_dir / "07_saleprice_qqplots.png", dpi=150)
    plt.close()

    raw_jb = jarque_bera(price)
    log_jb = jarque_bera(log_price)

    diagnostics = pd.DataFrame(
        [
            {
                "TargetVersion": "SalePrice",
                "Skewness": float(price.skew()),
                "ExcessKurtosis": float(kurtosis(price, fisher=True, bias=False)),
                "JarqueBeraStat": float(raw_jb.statistic),
                "JarqueBeraPValue": float(raw_jb.pvalue),
            },
            {
                "TargetVersion": "log1p(SalePrice)",
                "Skewness": float(log_price.skew()),
                "ExcessKurtosis": float(kurtosis(log_price, fisher=True, bias=False)),
                "JarqueBeraStat": float(log_jb.statistic),
                "JarqueBeraPValue": float(log_jb.pvalue),
            },
        ]
    )
    diagnostics.to_csv(tables_dir / "saleprice_distribution_diagnostics.csv", index=False)

    return {
        "saleprice_skew": float(price.skew()),
        "saleprice_log_skew": float(log_price.skew()),
        "saleprice_kurtosis": float(kurtosis(price, fisher=True, bias=False)),
        "saleprice_log_kurtosis": float(kurtosis(log_price, fisher=True, bias=False)),
        "saleprice_jb_pvalue": float(raw_jb.pvalue),
        "saleprice_log_jb_pvalue": float(log_jb.pvalue),
    }


def save_scatter_plots(df: pd.DataFrame, figures_dir: Path, tables_dir: Path) -> None:
    important_features = ["OverallQual", "GrLivArea", "GarageArea", "TotalBsmtSF"]
    diagnostic_rows: List[Dict[str, float | str]] = []
    for idx, col in enumerate(important_features, start=1):
        if col not in df.columns:
            continue
        plt.figure(figsize=(8, 5))
        sns.scatterplot(x=df[col], y=df["SalePrice"], alpha=0.7)
        sns.regplot(
            x=df[col],
            y=df["SalePrice"],
            scatter=False,
            order=2,
            ci=None,
            line_kws={"color": "red", "lw": 1.5},
        )
        plt.title(f"{col} vs SalePrice")
        plt.tight_layout()
        plt.savefig(figures_dir / f"07_scatter_{idx}_{col}.png", dpi=150)
        if col == "GrLivArea":
            plt.savefig(figures_dir / "grlivarea_vs_saleprice.png", dpi=150)
        plt.close()

        valid = df[[col, "SalePrice"]].dropna()
        if len(valid) > 3:
            pearson_corr, pearson_p = pearsonr(valid[col], valid["SalePrice"])
            spearman_corr, spearman_p = spearmanr(valid[col], valid["SalePrice"])
            x_bins = pd.qcut(valid[col], q=5, duplicates="drop")
            var_by_bin = valid.groupby(x_bins, observed=False)["SalePrice"].var().dropna()
            var_ratio = (
                float(var_by_bin.max() / var_by_bin.min())
                if len(var_by_bin) > 1 and (var_by_bin.min() > 0)
                else np.nan
            )
            diagnostic_rows.append(
                {
                    "Feature": col,
                    "PearsonCorr": float(pearson_corr),
                    "PearsonPValue": float(pearson_p),
                    "SpearmanCorr": float(spearman_corr),
                    "SpearmanPValue": float(spearman_p),
                    "MonotonicMinusLinear": float(abs(spearman_corr) - abs(pearson_corr)),
                    "SalePriceVarRatio_Q5": var_ratio,
                }
            )

    if diagnostic_rows:
        pd.DataFrame(diagnostic_rows).to_csv(
            tables_dir / "scatter_relationship_diagnostics.csv", index=False
        )


def compute_pearson(
    df: pd.DataFrame, figures_dir: Path, tables_dir: Path
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    numeric_df = df.select_dtypes(include=np.number)
    corr_matrix = numeric_df.corr(numeric_only=True)
    corr_matrix.to_csv(tables_dir / "pearson_full_matrix.csv")

    top_corr = (
        corr_matrix["SalePrice"]
        .dropna()
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"index": "Feature", "SalePrice": "PearsonCorr"})
    )
    top_corr.to_csv(tables_dir / "pearson_saleprice_ranking.csv", index=False)

    top_features = top_corr["Feature"].head(15).tolist()
    if top_features:
        plt.figure(figsize=(12, 10))
        sns.heatmap(df[top_features].corr(numeric_only=True), annot=True, cmap="coolwarm")
        plt.title("Top Correlated Features (Pearson)")
        plt.tight_layout()
        plt.savefig(figures_dir / "08_top_correlation_heatmap.png", dpi=150)
        plt.close()

    return top_corr, corr_matrix


def compute_multicollinearity_pairs(
    corr_matrix: pd.DataFrame, tables_dir: Path, threshold: float = 0.8
) -> pd.DataFrame:
    abs_corr = corr_matrix.abs()
    rows: List[Dict[str, float | str]] = []
    cols = list(abs_corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = abs_corr.iloc[i, j]
            if np.isfinite(val) and val >= threshold:
                rows.append(
                    {
                        "FeatureA": cols[i],
                        "FeatureB": cols[j],
                        "AbsPearsonCorr": float(val),
                    }
                )
    if rows:
        pair_df = pd.DataFrame(rows).sort_values("AbsPearsonCorr", ascending=False)
    else:
        pair_df = pd.DataFrame(columns=["FeatureA", "FeatureB", "AbsPearsonCorr"])
    pair_df.to_csv(tables_dir / "multicollinearity_pairs.csv", index=False)
    return pair_df


def compute_spearman(df: pd.DataFrame, tables_dir: Path) -> pd.DataFrame:
    numeric_df = df.select_dtypes(include=np.number).copy()
    numeric_cols = [c for c in numeric_df.columns if c != "SalePrice"]
    rows: List[Tuple[str, float]] = []
    for col in numeric_cols:
        corr, _ = spearmanr(numeric_df[col], numeric_df["SalePrice"], nan_policy="omit")
        rows.append((col, float(corr)))
    spearman_df = pd.DataFrame(rows, columns=["Feature", "SpearmanCorr"]).sort_values(
        "SpearmanCorr", ascending=False
    )
    spearman_df.to_csv(tables_dir / "spearman_saleprice_ranking.csv", index=False)
    return spearman_df


def compute_mutual_info(df: pd.DataFrame, tables_dir: Path) -> pd.DataFrame:
    mi_df = prepare_for_model(df, target_col="SalePrice")
    X = mi_df.drop(columns=["SalePrice"])
    y = mi_df["SalePrice"]
    mi_scores = mutual_info_regression(X, y, random_state=42)
    mi_result = (
        pd.DataFrame({"Feature": X.columns, "MI_Score": mi_scores})
        .sort_values("MI_Score", ascending=False)
        .reset_index(drop=True)
    )
    mi_result.to_csv(tables_dir / "mutual_information_ranking.csv", index=False)
    return mi_result


def save_segment_plots(df: pd.DataFrame, figures_dir: Path, tables_dir: Path) -> None:
    segment_specs = [
        ("Neighborhood", "09_segment_neighborhood.png", 90),
        ("OverallQual", "10_segment_overallqual.png", 0),
        ("HouseStyle", "11_segment_housestyle.png", 45),
        ("SaleCondition", "12_segment_salecondition.png", 45),
    ]
    for col, file_name, rotate in segment_specs:
        if col not in df.columns:
            continue
        plot_df = df[[col, "SalePrice"]].dropna().copy()
        if plot_df.empty:
            continue
        order = plot_df.groupby(col)["SalePrice"].median().sort_values(ascending=False).index
        plt.figure(figsize=(14, 6))
        sns.boxplot(x=col, y="SalePrice", data=plot_df, order=order)
        plt.xticks(rotation=rotate)
        plt.title(f"SalePrice by {col}")
        plt.tight_layout()
        plt.savefig(figures_dir / file_name, dpi=150)
        if col == "OverallQual":
            plt.savefig(figures_dir / "overallqual_vs_saleprice.png", dpi=150)
        plt.close()

    if "OverallQual" in df.columns:
        overallqual_stats = (
            df.groupby("OverallQual")["SalePrice"]
            .agg(
                count="size",
                median="median",
                q1=lambda s: s.quantile(0.25),
                q3=lambda s: s.quantile(0.75),
                mean="mean",
                std="std",
            )
            .reset_index()
            .sort_values("OverallQual")
        )
        overallqual_stats["IQR"] = overallqual_stats["q3"] - overallqual_stats["q1"]
        overallqual_stats.to_csv(tables_dir / "overallqual_saleprice_stats.csv", index=False)

    if "Neighborhood" in df.columns:
        neighborhood_stats = (
            df.groupby("Neighborhood")["SalePrice"]
            .agg(count="size", median="median", mean="mean", std="std")
            .sort_values("median", ascending=False)
            .reset_index()
        )
        neighborhood_stats.to_csv(tables_dir / "neighborhood_saleprice_stats.csv", index=False)


def residual_analysis(
    df: pd.DataFrame,
    figures_dir: Path,
    tables_dir: Path,
    test_size: float,
    random_state: int,
) -> Tuple[Dict[str, float], pd.DataFrame, pd.DataFrame]:
    model_df = prepare_for_model(df, target_col="SalePrice")
    X = model_df.drop(columns=["SalePrice"])
    y = model_df["SalePrice"]

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    residuals = y_valid - preds

    coef_df = pd.DataFrame(
        {
            "Feature": X.columns,
            "Coefficient": model.coef_,
            "AbsCoefficient": np.abs(model.coef_),
        }
    ).sort_values("AbsCoefficient", ascending=False)
    coef_df.to_csv(tables_dir / "linear_model_coefficients.csv", index=False)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=preds, y=residuals, alpha=0.7)
    plt.axhline(0, color="red")
    plt.title("Residual Plot")
    plt.xlabel("Predicted Price")
    plt.ylabel("Residual")
    plt.tight_layout()
    plt.savefig(figures_dir / "13_residual_plot.png", dpi=150)
    plt.close()

    error_df = pd.DataFrame(index=X_valid.index)
    error_df["Actual"] = y_valid
    error_df["Predicted"] = preds
    error_df["Residual"] = residuals
    error_df["AbsResidual"] = np.abs(residuals)

    for col in ["OverallQual", "Neighborhood", "GrLivArea"]:
        if col in df.columns:
            error_df[col] = df.loc[error_df.index, col]

    if "OverallQual" in error_df.columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x="OverallQual", y="Residual", data=error_df)
        plt.title("Residuals by OverallQual")
        plt.tight_layout()
        plt.savefig(figures_dir / "14_residual_by_overallqual.png", dpi=150)
        plt.close()
        (
            error_df.groupby("OverallQual")
            .agg(count=("Residual", "size"), mean_residual=("Residual", "mean"), mae=("AbsResidual", "mean"))
            .sort_values("mae", ascending=False)
            .to_csv(tables_dir / "residual_by_overallqual.csv")
        )

    if "Neighborhood" in error_df.columns:
        (
            error_df.groupby("Neighborhood")
            .agg(count=("Residual", "size"), mean_residual=("Residual", "mean"), mae=("AbsResidual", "mean"))
            .sort_values("mae", ascending=False)
            .to_csv(tables_dir / "residual_by_neighborhood.csv")
        )

    error_df["PriceSegment"] = pd.qcut(
        error_df["Actual"],
        q=[0.0, 0.5, 0.9, 1.0],
        labels=["Mainstream_50pct", "UpperMid_40pct", "Luxury_10pct"],
        duplicates="drop",
    )
    (
        error_df.groupby("PriceSegment", observed=False)
        .agg(count=("Residual", "size"), mean_residual=("Residual", "mean"), mae=("AbsResidual", "mean"))
        .to_csv(tables_dir / "residual_by_price_segment.csv")
    )

    pearson_het, pearson_het_p = pearsonr(error_df["Predicted"], error_df["AbsResidual"])
    spearman_het, spearman_het_p = spearmanr(error_df["Predicted"], error_df["AbsResidual"])
    pd.DataFrame(
        [
            {
                "Metric": "corr(abs_residual, predicted)",
                "Pearson": float(pearson_het),
                "PearsonPValue": float(pearson_het_p),
                "Spearman": float(spearman_het),
                "SpearmanPValue": float(spearman_het_p),
            }
        ]
    ).to_csv(tables_dir / "heteroscedasticity_proxy.csv", index=False)

    mae = mean_absolute_error(y_valid, preds)
    rmse = np.sqrt(mean_squared_error(y_valid, preds))
    r2 = r2_score(y_valid, preds)
    pd.DataFrame({"metric": ["MAE", "RMSE", "R2"], "value": [mae, rmse, r2]}).to_csv(
        tables_dir / "residual_metrics.csv", index=False
    )
    error_df.to_csv(tables_dir / "validation_predictions_residuals.csv", index=True)
    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
        "AbsResidualPredictedPearson": float(pearson_het),
        "AbsResidualPredictedPearsonPValue": float(pearson_het_p),
        "AbsResidualPredictedSpearman": float(spearman_het),
        "AbsResidualPredictedSpearmanPValue": float(spearman_het_p),
    }, error_df, coef_df


def save_interactive_plots(
    df: pd.DataFrame,
    corr_matrix: pd.DataFrame,
    error_df: pd.DataFrame,
    outlier_mask: pd.Series,
    interactive_dir: Path,
) -> None:
    html_opts = {"include_plotlyjs": "directory", "full_html": True}

    fig = px.histogram(df, x="SalePrice", nbins=50, title="SalePrice Distribution (Interactive)")
    fig.write_html(interactive_dir / "01_saleprice_distribution_interactive.html", **html_opts)
    fig = px.histogram(
        df.assign(SalePriceLog=np.log1p(df["SalePrice"])),
        x="SalePriceLog",
        nbins=50,
        title="log1p(SalePrice) Distribution (Interactive)",
    )
    fig.write_html(interactive_dir / "01b_saleprice_log_distribution_interactive.html", **html_opts)

    if "GrLivArea" in df.columns:
        outlier_plot_df = df[["GrLivArea", "SalePrice"]].copy()
        outlier_plot_df["OutlierFlag"] = np.where(outlier_mask, "Domain outlier", "Inlier")
        fig = px.scatter(
            outlier_plot_df,
            x="GrLivArea",
            y="SalePrice",
            color="OutlierFlag",
            title="Domain Outlier Check: GrLivArea vs SalePrice (Interactive)",
        )
        fig.write_html(interactive_dir / "02_outlier_scatter_interactive.html", **html_opts)

    scatter_features = ["OverallQual", "GrLivArea", "GarageArea", "TotalBsmtSF"]
    hover_candidates = ["Neighborhood", "GarageCars", "OverallQual", "LotArea"]
    hover_cols = [c for c in hover_candidates if c in df.columns]
    for idx, col in enumerate(scatter_features, start=1):
        if col not in df.columns:
            continue
        fig = px.scatter(
            df,
            x=col,
            y="SalePrice",
            color="OverallQual" if "OverallQual" in df.columns and col != "OverallQual" else None,
            hover_data=hover_cols,
            title=f"{col} vs SalePrice (Interactive)",
        )
        fig.update_traces(marker={"size": 7, "opacity": 0.75})
        fig.write_html(interactive_dir / f"03_scatter_{idx}_{col}_interactive.html", **html_opts)

    fig = px.imshow(
        corr_matrix,
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
        text_auto=".2f",
        title="Correlation Heatmap (Interactive)",
        aspect="auto",
    )
    fig.update_layout(width=1200, height=1200)
    fig.write_html(interactive_dir / "04_pearson_heatmap_interactive.html", **html_opts)

    segment_specs = [
        ("Neighborhood", -45),
        ("OverallQual", 0),
        ("HouseStyle", -35),
        ("SaleCondition", -35),
    ]
    for col, angle in segment_specs:
        if col not in df.columns:
            continue
        plot_df = df[[col, "SalePrice"]].dropna().copy()
        if plot_df.empty:
            continue
        order = plot_df.groupby(col)["SalePrice"].median().sort_values(ascending=False).index
        fig = px.box(
            plot_df,
            x=col,
            y="SalePrice",
            points="outliers",
            title=f"SalePrice by {col} (Interactive)",
            category_orders={col: list(order)},
        )
        fig.update_layout(xaxis_tickangle=angle)
        fig.write_html(interactive_dir / f"05_segment_{col}_interactive.html", **html_opts)

    hover_residual = [c for c in ["Actual", "Residual", "Neighborhood", "OverallQual"] if c in error_df.columns]
    fig = px.scatter(
        error_df,
        x="Predicted",
        y="Residual",
        color="OverallQual" if "OverallQual" in error_df.columns else None,
        hover_data=hover_residual,
        title="Residual Plot (Interactive)",
    )
    fig.add_hline(y=0, line_color="red")
    fig.write_html(interactive_dir / "06_residual_scatter_interactive.html", **html_opts)

    if "Neighborhood" in error_df.columns:
        order = error_df.groupby("Neighborhood")["AbsResidual"].mean().sort_values(ascending=False).index
        fig = px.box(
            error_df,
            x="Neighborhood",
            y="Residual",
            points="outliers",
            title="Residuals by Neighborhood (Interactive)",
            category_orders={"Neighborhood": list(order)},
        )
        fig.update_layout(xaxis_tickangle=-45)
        fig.write_html(interactive_dir / "07_residual_by_neighborhood_interactive.html", **html_opts)

    if "PriceSegment" in error_df.columns:
        fig = px.box(
            error_df,
            x="PriceSegment",
            y="Residual",
            points="outliers",
            title="Residuals by Price Segment (Interactive)",
        )
        fig.write_html(interactive_dir / "08_residual_by_price_segment_interactive.html", **html_opts)


def save_post_preprocessing_evidence(
    train_before_outlier: pd.DataFrame,
    train_after_processing: pd.DataFrame,
    outlier_mask: pd.Series,
    error_df: pd.DataFrame,
    post_eda_dir: Path,
) -> None:
    if {"GrLivArea", "SalePrice"}.issubset(train_before_outlier.columns):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        sns.scatterplot(
            x=train_before_outlier["GrLivArea"],
            y=train_before_outlier["SalePrice"],
            alpha=0.7,
            ax=axes[0],
            color="#4C78A8",
        )
        if outlier_mask.any():
            flagged = train_before_outlier.loc[outlier_mask]
            sns.scatterplot(
                x=flagged["GrLivArea"],
                y=flagged["SalePrice"],
                color="red",
                s=80,
                ax=axes[0],
                label="Domain outlier",
            )
        axes[0].set_title("Before Outlier Handling")
        axes[0].set_xlabel("GrLivArea")
        axes[0].set_ylabel("SalePrice")

        sns.scatterplot(
            x=train_after_processing["GrLivArea"],
            y=train_after_processing["SalePrice"],
            alpha=0.7,
            ax=axes[1],
            color="#4C78A8",
        )
        sns.regplot(
            x=train_after_processing["GrLivArea"],
            y=train_after_processing["SalePrice"],
            scatter=False,
            order=2,
            ci=None,
            line_kws={"color": "red", "lw": 1.5},
            ax=axes[1],
        )
        axes[1].set_title("After Outlier Handling")
        axes[1].set_xlabel("GrLivArea")
        axes[1].set_ylabel("SalePrice")
        plt.tight_layout()
        plt.savefig(post_eda_dir / "after_outlier_scatter.png", dpi=150)
        plt.close()

    if "SalePrice" in train_before_outlier.columns and "SalePrice" in train_after_processing.columns:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        sns.histplot(train_before_outlier["SalePrice"], kde=True, ax=axes[0], color="#4C78A8")
        axes[0].set_title("SalePrice Before Transformation")
        axes[0].set_xlabel("SalePrice")

        sns.histplot(
            np.log1p(train_after_processing["SalePrice"]),
            kde=True,
            ax=axes[1],
            color="teal",
        )
        axes[1].set_title("log1p(SalePrice) After Transformation")
        axes[1].set_xlabel("log1p(SalePrice)")
        plt.tight_layout()
        plt.savefig(post_eda_dir / "saleprice_log_distribution.png", dpi=150)
        plt.close()

        before_price = train_before_outlier["SalePrice"].dropna()
        after_price = train_after_processing["SalePrice"].dropna()
        after_log = np.log1p(after_price)

        skew_rows = []
        for name, values in [
            ("before_saleprice", before_price),
            ("after_outlier_saleprice", after_price),
            ("after_log1p_saleprice", after_log),
        ]:
            jb_result = jarque_bera(values)
            skew_rows.append(
                {
                    "Version": name,
                    "Skewness": float(values.skew()),
                    "ExcessKurtosis": float(kurtosis(values, fisher=True, bias=False)),
                    "JarqueBeraStat": float(jb_result.statistic),
                    "JarqueBeraPValue": float(jb_result.pvalue),
                    "Count": int(values.shape[0]),
                }
            )
        pd.DataFrame(skew_rows).to_csv(post_eda_dir / "skewness_before_after.csv", index=False)

    if {"TotalSF", "SalePrice"}.issubset(train_after_processing.columns):
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            x=train_after_processing["TotalSF"],
            y=train_after_processing["SalePrice"],
            alpha=0.7,
        )
        sns.regplot(
            x=train_after_processing["TotalSF"],
            y=train_after_processing["SalePrice"],
            scatter=False,
            order=2,
            ci=None,
            line_kws={"color": "red", "lw": 1.5},
        )
        plt.title("TotalSF vs SalePrice")
        plt.tight_layout()
        plt.savefig(post_eda_dir / "totalsf_vs_saleprice.png", dpi=150)
        plt.close()

        corr_candidates = [c for c in ["1stFlrSF", "2ndFlrSF", "TotalBsmtSF", "GrLivArea", "TotalSF"] if c in train_after_processing.columns]
        corr_rows = []
        for col in corr_candidates:
            valid = train_after_processing[[col, "SalePrice"]].dropna()
            if len(valid) < 3:
                continue
            corr_val, _ = pearsonr(valid[col], valid["SalePrice"])
            corr_rows.append({"Feature": col, "PearsonWithSalePrice": float(corr_val)})
        if corr_rows:
            pd.DataFrame(corr_rows).sort_values(
                "PearsonWithSalePrice", ascending=False
            ).to_csv(post_eda_dir / "feature_engineering_correlation_gain.csv", index=False)

    if "Residual" in error_df.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(error_df["Residual"], kde=True, color="#2E8B57")
        plt.axvline(0, color="red")
        plt.title("Residual Distribution")
        plt.tight_layout()
        plt.savefig(post_eda_dir / "residual_distribution.png", dpi=150)
        plt.close()


def build_summary(
    output_path: Path,
    data_source: str,
    train_shape: Tuple[int, int],
    test_shape: Tuple[int, int],
    skew_info: Dict[str, float],
    outlier_count: int,
    used_outlier_removed: bool,
    sensitivity_df: pd.DataFrame,
    multicollinearity_df: pd.DataFrame,
    pearson_df: pd.DataFrame,
    spearman_df: pd.DataFrame,
    mi_df: pd.DataFrame,
    residual_metrics: Dict[str, float],
    error_df: pd.DataFrame,
    coef_df: pd.DataFrame,
) -> None:
    top_pearson = pearson_df[pearson_df["Feature"] != "SalePrice"].head(10)
    top_spearman = spearman_df.head(10)
    top_mi = mi_df.head(10)
    lines: List[str] = [
        "# Ames EDA Workflow Summary",
        "",
        f"- Data source: {data_source}",
        f"- Train shape: {train_shape[0]} x {train_shape[1]}",
        f"- Test shape: {test_shape[0]} x {test_shape[1]}",
        f"- Domain outliers found: {outlier_count}",
        f"- Downstream analysis uses outlier-removed train: {used_outlier_removed}",
        (
            f"- SalePrice skewness: {skew_info['saleprice_skew']:.4f} "
            f"(log1p: {skew_info['saleprice_log_skew']:.4f})"
        ),
        (
            f"- SalePrice excess kurtosis: {skew_info['saleprice_kurtosis']:.4f} "
            f"(log1p: {skew_info['saleprice_log_kurtosis']:.4f})"
        ),
        (
            f"- Jarque-Bera p-value: {skew_info['saleprice_jb_pvalue']:.6f} "
            f"(log1p: {skew_info['saleprice_log_jb_pvalue']:.6f})"
        ),
        "",
        "## Outlier Sensitivity (Linear Regression Baseline)",
        sensitivity_df.to_string(index=False) if not sensitivity_df.empty else "Not enough columns to run.",
        "",
        "## Top Pearson Correlation with SalePrice",
        top_pearson.to_string(index=False),
        "",
        "## Top Spearman Correlation with SalePrice",
        top_spearman.to_string(index=False),
        "",
        "## Top Mutual Information Features",
        top_mi.to_string(index=False),
        "",
        "## Residual Metrics",
        f"- MAE: {residual_metrics['MAE']:.2f}",
        f"- RMSE: {residual_metrics['RMSE']:.2f}",
        f"- R2: {residual_metrics['R2']:.4f}",
        (
            "- Heteroscedasticity proxy (corr(abs residual, predicted)) "
            f"Pearson={residual_metrics['AbsResidualPredictedPearson']:.4f} "
            f"(p={residual_metrics['AbsResidualPredictedPearsonPValue']:.6f}), "
            f"Spearman={residual_metrics['AbsResidualPredictedSpearman']:.4f} "
            f"(p={residual_metrics['AbsResidualPredictedSpearmanPValue']:.6f})"
        ),
    ]

    if not multicollinearity_df.empty:
        lines.extend(
            [
                "",
                "## Top Multicollinearity Pairs (|Pearson| >= 0.8)",
                multicollinearity_df.head(10).to_string(index=False),
            ]
        )

    if not coef_df.empty:
        lines.extend(
            [
                "",
                "## Linear Model Interpretability (Top |Coefficient|)",
                coef_df.head(10)[["Feature", "Coefficient", "AbsCoefficient"]].to_string(index=False),
            ]
        )

    if "Neighborhood" in error_df.columns:
        top_hotspots = (
            error_df.groupby("Neighborhood")["AbsResidual"].mean().sort_values(ascending=False).head(5)
        )
        lines.extend(
            [
                "",
                "## Residual Hotspots by Neighborhood",
                top_hotspots.to_string(),
            ]
        )

    if "OverallQual" in error_df.columns:
        top_quality = (
            error_df.groupby("OverallQual")["AbsResidual"].mean().sort_values(ascending=False).head(5)
        )
        lines.extend(
            [
                "",
                "## Residual Hotspots by OverallQual",
                top_quality.to_string(),
            ]
        )

    if "PriceSegment" in error_df.columns:
        segment_error = (
            error_df.groupby("PriceSegment", observed=False)["AbsResidual"]
            .mean()
            .sort_values(ascending=False)
        )
        lines.extend(
            [
                "",
                "## Residual Hotspots by Price Segment",
                segment_error.to_string(),
            ]
        )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    clean_output_dir(output_dir, keep_old_output=args.keep_old_output)
    figures_dir, tables_dir, interactive_dir, post_eda_dir = ensure_dirs(output_dir)

    train_raw, test_raw, source = load_train_test(args)
    if "SalePrice" not in train_raw.columns:
        raise ValueError("Train dataset must include 'SalePrice'.")

    train_clean, test_clean = apply_domain_missing_strategy(
        train_df=train_raw,
        test_df=test_raw,
        data_description_path=Path(args.data_description_path),
        figures_dir=figures_dir,
        tables_dir=tables_dir,
    )

    train_clean = add_engineered_features(train_clean)
    test_clean = add_engineered_features(test_clean)
    train_clean.to_csv(tables_dir / "train_after_domain_cleaning.csv", index=False)
    test_clean.to_csv(tables_dir / "test_after_domain_cleaning.csv", index=False)

    outlier_mask = detect_domain_outliers(
        train_clean,
        grliv_threshold=args.outlier_grliv_threshold,
        saleprice_threshold=args.outlier_saleprice_threshold,
    )
    outlier_candidates = train_clean.loc[outlier_mask].copy()
    outlier_candidates.to_csv(tables_dir / "domain_outlier_candidates.csv", index=False)
    save_outlier_scatter(
        train_clean,
        outlier_mask,
        figures_dir,
        title_suffix=(
            f" (Rule: GrLivArea > {args.outlier_grliv_threshold:.0f} & "
            f"SalePrice < {args.outlier_saleprice_threshold:.0f})"
        ),
    )

    sensitivity_df = run_outlier_sensitivity_analysis(
        train_df=train_clean,
        outlier_mask=outlier_mask,
        tables_dir=tables_dir,
        figures_dir=figures_dir,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    analysis_train = train_clean.loc[~outlier_mask].copy() if args.drop_domain_outliers else train_clean.copy()

    skew_info = save_saleprice_distributions(analysis_train, figures_dir, tables_dir)
    save_scatter_plots(analysis_train, figures_dir, tables_dir)
    pearson_df, corr_matrix = compute_pearson(analysis_train, figures_dir, tables_dir)
    multicollinearity_df = compute_multicollinearity_pairs(corr_matrix, tables_dir, threshold=0.8)
    spearman_df = compute_spearman(analysis_train, tables_dir)
    mi_df = compute_mutual_info(analysis_train, tables_dir)
    save_segment_plots(analysis_train, figures_dir, tables_dir)
    residual_metrics, error_df, coef_df = residual_analysis(
        analysis_train,
        figures_dir=figures_dir,
        tables_dir=tables_dir,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    save_post_preprocessing_evidence(
        train_before_outlier=train_clean,
        train_after_processing=analysis_train,
        outlier_mask=outlier_mask.loc[train_clean.index],
        error_df=error_df,
        post_eda_dir=post_eda_dir,
    )

    if not args.no_interactive:
        save_interactive_plots(
            df=analysis_train,
            corr_matrix=corr_matrix,
            error_df=error_df,
            outlier_mask=outlier_mask.loc[analysis_train.index],
            interactive_dir=interactive_dir,
        )

    summary_path = output_dir / "eda_summary.md"
    build_summary(
        output_path=summary_path,
        data_source=source,
        train_shape=train_clean.shape,
        test_shape=test_clean.shape,
        skew_info=skew_info,
        outlier_count=int(outlier_mask.sum()),
        used_outlier_removed=bool(args.drop_domain_outliers),
        sensitivity_df=sensitivity_df,
        multicollinearity_df=multicollinearity_df,
        pearson_df=pearson_df,
        spearman_df=spearman_df,
        mi_df=mi_df,
        residual_metrics=residual_metrics,
        error_df=error_df,
        coef_df=coef_df,
    )

    print("[DONE] EDA workflow completed.")
    print(f"[DONE] Summary: {summary_path}")
    print(f"[DONE] Figures: {figures_dir}")
    print(f"[DONE] Tables: {tables_dir}")
    print(f"[DONE] Post-preprocessing EDA: {post_eda_dir}")
    if not args.no_interactive:
        print(f"[DONE] Interactive HTML: {interactive_dir}")


if __name__ == "__main__":
    main()
