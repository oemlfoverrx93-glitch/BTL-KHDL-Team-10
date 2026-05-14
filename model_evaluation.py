from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from preprocessing import (
    apply_basic_cleaning,
    load_raw_data,
    remove_outliers,
    split_training_data,
)

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

MODEL_FILES = {
    "Linear_Regression": "Linear_Regression.pkl",
    "Decision_Tree": "Decision_Tree.pkl",
    "Random_Forest": "Random_Forest.pkl",
    "Gradient_Boosting": "Gradient_Boosting.pkl",
    "best_model": "best_model.pkl",
}


def _save_actual_vs_predicted_plot(
    y_true: pd.Series,
    y_pred: np.ndarray,
    output_path: Path,
) -> None:
    plt.figure(figsize=(10, 5))
    n_points = min(120, len(y_true))
    plt.plot(y_true.values[:n_points], label="Actual", linewidth=2)
    plt.plot(y_pred[:n_points], label="Predicted", linewidth=2)
    plt.title("Actual vs Predicted (Validation)")
    plt.xlabel("Sample Index")
    plt.ylabel("SalePrice")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def _save_residual_plot(
    y_true: pd.Series,
    y_pred: np.ndarray,
    output_path: Path,
) -> None:
    residuals = y_true.values - y_pred

    plt.figure(figsize=(8, 5))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(0, color="red", linestyle="--", linewidth=1)
    plt.title("Residual Plot (Validation)")
    plt.xlabel("Predicted SalePrice")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def _save_feature_importance_plot(pipeline, output_path: Path) -> bool:
    model = pipeline.named_steps.get("model")
    preprocessor = pipeline.named_steps.get("preprocessor")

    if model is None or preprocessor is None:
        return False

    if not hasattr(preprocessor, "get_feature_names_out"):
        return False

    feature_names = preprocessor.get_feature_names_out()

    if hasattr(model, "feature_importances_"):
        importance_values = np.asarray(model.feature_importances_)
    elif hasattr(model, "coef_"):
        importance_values = np.abs(np.asarray(model.coef_).ravel())
    else:
        return False

    if len(feature_names) != len(importance_values):
        return False

    importance_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": importance_values,
        }
    ).sort_values("Importance", ascending=False)

    top_features = importance_df.head(15)

    plt.figure(figsize=(10, 6))
    plt.barh(top_features["Feature"], top_features["Importance"])
    plt.gca().invert_yaxis()
    plt.title("Top Feature Importance (Best Model)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    return True


def _save_error_analysis(
    y_true: pd.Series,
    y_pred: np.ndarray,
    output_path: Path,
) -> None:
    error_df = pd.DataFrame(
        {
            "Actual": y_true.values,
            "Predicted": y_pred,
        }
    )
    error_df["Residual"] = error_df["Actual"] - error_df["Predicted"]
    error_df["AbsError"] = error_df["Residual"].abs()
    error_df = error_df.sort_values("AbsError", ascending=False).reset_index(drop=True)
    error_df.to_csv(output_path, index=False)


def evaluate_models() -> pd.DataFrame:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    train_df, test_df, _ = load_raw_data()
    train_df, test_df = apply_basic_cleaning(train_df, test_df)
    train_df = remove_outliers(train_df)

    X_train, X_val, y_train, y_val = split_training_data(train_df)

    rows = []
    val_predictions_by_model = {}
    loaded_pipelines = {}

    print("Evaluating saved models...")

    for model_name, model_file in MODEL_FILES.items():
        model_path = MODELS_DIR / model_file
        if not model_path.exists():
            continue

        pipeline = joblib.load(model_path)
        loaded_pipelines[model_name] = pipeline

        train_predictions = pipeline.predict(X_train)
        val_predictions = pipeline.predict(X_val)
        val_predictions_by_model[model_name] = val_predictions

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

        rows.append(
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

        print(f"{model_name:18} | RMSE: {rmse:10.4f} | R2: {r2:.4f}")

    if not rows:
        raise FileNotFoundError(
            f"No model files found in {MODELS_DIR}. Run building_models.py first."
        )

    results_df = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    metrics_path = RESULTS_DIR / "model_evaluation_metrics.csv"
    results_df.to_csv(metrics_path, index=False)

    best_model_name = results_df.iloc[0]["Model"]
    best_predictions = val_predictions_by_model[best_model_name]
    best_pipeline = loaded_pipelines[best_model_name]

    actual_vs_predicted_path = RESULTS_DIR / "actual_vs_predicted.png"
    residual_plot_path = RESULTS_DIR / "residual_plot.png"
    feature_importance_path = RESULTS_DIR / "feature_importance.png"
    error_analysis_path = RESULTS_DIR / "error_analysis.csv"

    _save_actual_vs_predicted_plot(y_val, best_predictions, actual_vs_predicted_path)
    _save_residual_plot(y_val, best_predictions, residual_plot_path)
    has_feature_plot = _save_feature_importance_plot(best_pipeline, feature_importance_path)
    _save_error_analysis(y_val, best_predictions, error_analysis_path)

    print("\nEvaluation finished.")
    print("Saved metrics: results/model_evaluation_metrics.csv")
    print("Saved plot: results/actual_vs_predicted.png")
    print("Saved plot: results/residual_plot.png")
    print("Saved error analysis: results/error_analysis.csv")
    if has_feature_plot:
        print("Saved plot: results/feature_importance.png")
    else:
        print("Skipped feature importance plot for best model.")

    print(f"Best model by RMSE: {best_model_name}")

    return results_df


if __name__ == "__main__":
    evaluate_models()
