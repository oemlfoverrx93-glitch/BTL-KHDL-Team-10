from pathlib import Path

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

TRAIN_RAW = RAW_DIR / "train.csv"
TEST_RAW = RAW_DIR / "test.csv"
SAMPLE_SUB = RAW_DIR / "sample_submission.csv"
BEST_MODEL = MODELS_DIR / "best_model.pkl"
OUT_SUBMISSION = RESULTS_DIR / "submission.csv"

FEATURE_COLS = [
    "OverallQual",
    "GrLivArea",
    "GarageCars",
    "TotalBsmtSF",
    "FullBath",
    "YearBuilt",
    "LotArea",
    "BedroomAbvGr",
    "TotRmsAbvGrd",
    "GarageArea",
]


def main() -> None:
    for path in [TRAIN_RAW, TEST_RAW, SAMPLE_SUB, BEST_MODEL]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    train_df = pd.read_csv(TRAIN_RAW)
    test_df = pd.read_csv(TEST_RAW)
    submission_df = pd.read_csv(SAMPLE_SUB)

    X_train = train_df[FEATURE_COLS].copy()
    X_test = test_df[FEATURE_COLS].copy()

    # Fill missing values with train medians.
    medians = X_train.median(numeric_only=True)
    X_train = X_train.fillna(medians)
    X_test = X_test.fillna(medians)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = joblib.load(BEST_MODEL)
    preds = model.predict(X_test_scaled)

    submission_df["SalePrice"] = preds
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(OUT_SUBMISSION, index=False)

    print(f"Saved submission: {OUT_SUBMISSION}")


if __name__ == "__main__":
    main()

