# House Price Prediction (SQL-first Pipeline)

## 1. Project Overview
This project predicts house prices with a **SQL-first workflow**:
- SQL handles data import, cleaning, feature engineering, and model-ready view creation.
- Python handles model training, evaluation, and Kaggle-style submission generation.

## 2. Dataset
Source files are in `data/raw/`:
- `train.csv`
- `test.csv`
- `sample_submission.csv`
- `data_description.txt`

SQL-exported modeling data is in:
- `data/exported/clean_house_data.csv`

## 3. SQL Pipeline
SQL scripts are in `sql/` and run in this order:
1. `01_import_data.sql`
2. `02_data_cleaning.sql`
3. `03_feature_engineering.sql`
4. `04_analysis_queries.sql`
5. `05_create_model_table.sql`

## 4. ML Models Used
- Linear Regression
- Ridge Regression
- Random Forest Regressor
- Gradient Boosting Regressor

## 5. Evaluation Metrics
- MAE
- MSE
- RMSE
- R2 Score

Metrics output:
- `results/model_evaluation_metrics.csv`

## 6. Best Model
Best model is selected by lowest RMSE and saved as:
- `models/best_model.pkl`

## 7. How To Run
From project root:

1. Run SQL import script.
2. Run SQL cleaning script.
3. Run SQL feature engineering script.
4. Export clean data to `data/exported/clean_house_data.csv`.
5. Run training:
   - `python scripts/train_model.py`
6. Run evaluation:
   - `python scripts/model_evaluation.py`
7. Run submission prediction:
   - `python scripts/predict_submission.py`
