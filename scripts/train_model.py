from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "exported" / "clean_house_data.csv"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

TARGET_COL = "SalePrice"
RANDOM_STATE = 42


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing input file: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    models = {
        "linearregression_model.pkl": LinearRegression(),
        "ridgeregression_model.pkl": Ridge(alpha=1.0),
        "randomforest_model.pkl": RandomForestRegressor(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "gradientboosting_model.pkl": GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05, random_state=RANDOM_STATE
        ),
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    compare_df = pd.DataFrame({"Actual_SalePrice": y_val.reset_index(drop=True)})
    rmse_rows = []
    best_model_name = None
    best_rmse = float("inf")

    for file_name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        rmse = mean_squared_error(y_val, preds) ** 0.5
        rmse_rows.append({"ModelFile": file_name, "RMSE": rmse})

        joblib.dump(model, MODELS_DIR / file_name)
        compare_df[file_name.replace("_model.pkl", "")] = preds

        if rmse < best_rmse:
            best_rmse = rmse
            best_model_name = file_name

    if best_model_name is None:
        raise RuntimeError("No model was trained.")

    best_model = joblib.load(MODELS_DIR / best_model_name)
    joblib.dump(best_model, MODELS_DIR / "best_model.pkl")

    compare_df.to_csv(RESULTS_DIR / "model_predictions_comparison.csv", index=False)
    pd.DataFrame(rmse_rows).sort_values("RMSE").to_csv(
        RESULTS_DIR / "model_training_rmse.csv", index=False
    )

    print(f"Training complete. Best model: {best_model_name} (RMSE={best_rmse:.4f})")


if __name__ == "__main__":
    main()

