from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "exported" / "clean_house_data.csv"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

TARGET_COL = "SalePrice"
RANDOM_STATE = 42

MODEL_FILES = {
    "Linear Regression": "linearregression_model.pkl",
    "Ridge Regression": "ridgeregression_model.pkl",
    "Random Forest": "randomforest_model.pkl",
    "Gradient Boosting": "gradientboosting_model.pkl",
}


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing input file: {DATA_PATH}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    _, X_val, _, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    rows = []
    preds_by_model = {}

    for model_name, model_file in MODEL_FILES.items():
        full_path = MODELS_DIR / model_file
        if not full_path.exists():
            continue

        model = joblib.load(full_path)
        preds = model.predict(X_val)
        preds_by_model[model_name] = preds

        mae = mean_absolute_error(y_val, preds)
        mse = mean_squared_error(y_val, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_val, preds)

        rows.append(
            {
                "Model": model_name,
                "MAE": round(mae, 4),
                "MSE": round(mse, 4),
                "RMSE": round(rmse, 4),
                "R2_Score": round(r2, 4),
            }
        )

    if not rows:
        raise RuntimeError("No model file found for evaluation.")

    metrics_df = pd.DataFrame(rows).sort_values("RMSE")
    metrics_df.to_csv(RESULTS_DIR / "model_evaluation_metrics.csv", index=False)

    best_model_name = metrics_df.iloc[0]["Model"]
    best_preds = preds_by_model[best_model_name]

    eval_df = pd.DataFrame(
        {
            "actual": y_val.reset_index(drop=True),
            "predicted": best_preds,
        }
    )
    eval_df["residual"] = eval_df["actual"] - eval_df["predicted"]
    eval_df["abs_error"] = eval_df["residual"].abs()
    eval_df.to_csv(RESULTS_DIR / "error_analysis.csv", index=False)

    # Actual vs Predicted
    plt.figure(figsize=(8, 6))
    plt.scatter(eval_df["actual"], eval_df["predicted"], alpha=0.5)
    lims = [
        min(eval_df["actual"].min(), eval_df["predicted"].min()),
        max(eval_df["actual"].max(), eval_df["predicted"].max()),
    ]
    plt.plot(lims, lims, "r--")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"Actual vs Predicted ({best_model_name})")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "actual_vs_predicted.png", dpi=150)
    plt.close()

    # Residual plot
    plt.figure(figsize=(8, 6))
    plt.scatter(eval_df["predicted"], eval_df["residual"], alpha=0.5)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Predicted")
    plt.ylabel("Residual")
    plt.title(f"Residual Plot ({best_model_name})")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "residual_plot.png", dpi=150)
    plt.close()

    # Feature importance / coefficients
    best_file = MODELS_DIR / MODEL_FILES[best_model_name]
    best_model = joblib.load(best_file)
    feature_values = None
    if hasattr(best_model, "feature_importances_"):
        feature_values = best_model.feature_importances_
    elif hasattr(best_model, "coef_"):
        feature_values = np.abs(best_model.coef_)

    if feature_values is not None:
        importance_df = pd.DataFrame({"Feature": X.columns, "Importance": feature_values})
        importance_df = importance_df.sort_values("Importance", ascending=False).head(15)

        plt.figure(figsize=(10, 6))
        plt.barh(importance_df["Feature"], importance_df["Importance"])
        plt.gca().invert_yaxis()
        plt.title(f"Feature Importance ({best_model_name})")
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "feature_importance.png", dpi=150)
        plt.close()

    print(f"Evaluation complete. Best model: {best_model_name}")


if __name__ == "__main__":
    main()

