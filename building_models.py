from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from preprocessing import (
    apply_basic_cleaning,
    build_preprocessor,
    load_raw_data,
    remove_outliers,
    split_training_data,
)

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"


def get_models() -> dict[str, object]:
    return {
        "Linear_Regression": LinearRegression(),
        "Decision_Tree": DecisionTreeRegressor(max_depth=8, random_state=42),
        "Random_Forest": RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            random_state=42,
        ),
        "Gradient_Boosting": GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
        ),
    }


def run_ml_models() -> pd.DataFrame:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    train_df, test_df, test_ids = load_raw_data()
    train_df, test_df = apply_basic_cleaning(train_df, test_df)
    train_df = remove_outliers(train_df)

    X_train, X_val, y_train, y_val = split_training_data(train_df)

    models = get_models()

    results = []
    comparison_results = pd.DataFrame({"Id": test_ids})

    best_model_name = None
    best_pipeline = None
    best_rmse = np.inf

    print("Training models...")

    for model_name, model in models.items():
        preprocessor = build_preprocessor(X_train)
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        pipeline.fit(X_train, y_train)

        train_predictions = pipeline.predict(X_train)
        val_predictions = pipeline.predict(X_val)
        test_predictions = pipeline.predict(test_df)

        mae = mean_absolute_error(y_val, val_predictions)
        mse = mean_squared_error(y_val, val_predictions)
        rmse = float(np.sqrt(mse))
        r2 = r2_score(y_val, val_predictions)
        train_r2 = r2_score(y_train, train_predictions)

        overfitting_status = (
            "Potential overfitting"
            if (train_r2 - r2) > 0.1
            else "Stable"
        )

        results.append(
            {
                "Model": model_name,
                "MAE": round(mae, 4),
                "MSE": round(mse, 4),
                "RMSE": round(rmse, 4),
                "R2_Score": round(r2, 4),
                "Train_R2": round(train_r2, 4),
                "Overfitting": overfitting_status,
            }
        )

        comparison_results[model_name] = test_predictions

        model_path = MODELS_DIR / f"{model_name}.pkl"
        joblib.dump(pipeline, model_path)

        print(f"{model_name:18} | RMSE: {rmse:10.4f} | R2: {r2:.4f}")

        if rmse < best_rmse:
            best_rmse = rmse
            best_model_name = model_name
            best_pipeline = pipeline

    if best_pipeline is None:
        raise RuntimeError("No model was trained. Check input data.")

    results_df = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)

    metrics_path = RESULTS_DIR / "model_evaluation_metrics.csv"
    results_df.to_csv(metrics_path, index=False)

    comparison_path = RESULTS_DIR / "model_predictions_comparison.csv"
    comparison_results.to_csv(comparison_path, index=False)

    best_model_path = MODELS_DIR / "best_model.pkl"
    joblib.dump(best_pipeline, best_model_path)

    print("\nTraining finished.")
    print(f"Best model: {best_model_name} (RMSE={best_rmse:.4f})")
    print("Saved metrics: results/model_evaluation_metrics.csv")
    print("Saved predictions: results/model_predictions_comparison.csv")
    print("Saved best model: models/best_model.pkl")

    return results_df


if __name__ == "__main__":
    run_ml_models()
