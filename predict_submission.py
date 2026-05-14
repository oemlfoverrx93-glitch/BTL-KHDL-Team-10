from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from preprocessing import apply_basic_cleaning, load_raw_data

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"


def predict_submission() -> pd.DataFrame:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / "best_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Cannot find {model_path}. Run building_models.py first."
        )

    train_df, test_df, test_ids = load_raw_data()
    train_df, test_df = apply_basic_cleaning(train_df, test_df)

    model = joblib.load(model_path)
    predictions = model.predict(test_df)

    submission_df = pd.DataFrame(
        {
            "Id": test_ids,
            "SalePrice": predictions,
        }
    )

    submission_path = RESULTS_DIR / "submission.csv"
    submission_df.to_csv(submission_path, index=False)

    print("Saved submission: results/submission.csv")

    return submission_df


if __name__ == "__main__":
    predict_submission()
