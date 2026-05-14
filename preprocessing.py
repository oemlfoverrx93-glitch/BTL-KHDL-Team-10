from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COL = "SalePrice"
ID_COL = "Id"
RANDOM_STATE = 42

BASE_DIR = Path(__file__).resolve().parent
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"

NONE_AS_CATEGORY_COLUMNS = [
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


def _resolve_input_path(file_name: str) -> Path:
    """Read from data/raw first, then fallback to project root for compatibility."""
    candidates = [
        DATA_RAW_DIR / file_name,
        BASE_DIR / file_name,
    ]
    for path in candidates:
        if path.exists():
            return path
    checked = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Cannot find {file_name}. Checked: {checked}")


def load_raw_data(
    train_file: str = "train.csv",
    test_file: str = "test.csv",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Load raw train/test and return test Ids for submission."""
    train_path = _resolve_input_path(train_file)
    test_path = _resolve_input_path(test_file)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    if ID_COL not in test_df.columns:
        raise KeyError(f"Column '{ID_COL}' is missing in test data.")

    return train_df, test_df, test_df[ID_COL].copy()


def apply_basic_cleaning(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply deterministic data cleaning using train statistics for both sets.
    This avoids test-driven imputation leakage.
    """
    train = train_df.copy()
    test = test_df.copy()

    for col in NONE_AS_CATEGORY_COLUMNS:
        if col in train.columns:
            train[col] = train[col].fillna("None")
        if col in test.columns:
            test[col] = test[col].fillna("None")

    numeric_cols = train.select_dtypes(include=["number"]).columns.tolist()
    for col in numeric_cols:
        if col == TARGET_COL:
            continue
        median_value = train[col].median()
        train[col] = train[col].fillna(median_value)
        if col in test.columns:
            test[col] = test[col].fillna(median_value)

    categorical_cols = train.select_dtypes(include=["object"]).columns.tolist()
    for col in categorical_cols:
        mode_series = train[col].mode(dropna=True)
        fill_value = mode_series.iloc[0] if not mode_series.empty else "None"
        train[col] = train[col].fillna(fill_value)
        if col in test.columns:
            test[col] = test[col].fillna(fill_value)

    return train, test


def remove_outliers(train_df: pd.DataFrame) -> pd.DataFrame:
    """Remove known outliers from training data only."""
    if {"GrLivArea", TARGET_COL}.issubset(train_df.columns):
        mask = ~(
            (train_df["GrLivArea"] > 4000)
            & (train_df[TARGET_COL] < 300000)
        )
        return train_df.loc[mask].copy()
    return train_df.copy()


def split_training_data(
    train_df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
):
    if TARGET_COL not in train_df.columns:
        raise KeyError(f"Column '{TARGET_COL}' is missing in train data.")

    X = train_df.drop(columns=[TARGET_COL])
    y = train_df[TARGET_COL]

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )


def _build_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", _build_one_hot_encoder()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )


def _to_dense(matrix):
    return matrix.toarray() if hasattr(matrix, "toarray") else matrix


def preprocess_data_for_reporting() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create processed CSVs for reporting only.
    Training should use raw data + Pipeline in building_models.py to avoid leakage.
    """
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    train_df, test_df, _ = load_raw_data()
    train_df, test_df = apply_basic_cleaning(train_df, test_df)
    train_df = remove_outliers(train_df)

    X = train_df.drop(columns=[TARGET_COL])
    y = train_df[TARGET_COL].reset_index(drop=True)

    preprocessor = build_preprocessor(X)
    X_processed = _to_dense(preprocessor.fit_transform(X))
    test_processed = _to_dense(preprocessor.transform(test_df))

    feature_names = preprocessor.get_feature_names_out()

    processed_train = pd.DataFrame(X_processed, columns=feature_names)
    processed_train[TARGET_COL] = y

    processed_test = pd.DataFrame(test_processed, columns=feature_names)

    processed_train.to_csv(
        DATA_PROCESSED_DIR / "processed_train.csv",
        index=False,
    )
    processed_test.to_csv(
        DATA_PROCESSED_DIR / "processed_test.csv",
        index=False,
    )

    print("Preprocessing completed.")
    print(f"Processed train shape: {processed_train.shape}")
    print(f"Processed test shape: {processed_test.shape}")
    print("Saved: data/processed/processed_train.csv and data/processed/processed_test.csv")

    return processed_train, processed_test


if __name__ == "__main__":
    preprocess_data_for_reporting()
