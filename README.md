# House Prices Prediction Pipeline

This project is organized as an end-to-end Data Science workflow:

Raw Data -> Preprocessing -> Train/Validation Split -> Multi-model Training -> Evaluation -> Best Model -> Submission

## Folder Structure

- `data/raw/`: `train.csv`, `test.csv`, `sample_submission.csv`, `data_description.txt`
- `data/processed/`: generated processed files for reporting
- `models/`: trained model artifacts (`*.pkl`) and `best_model.pkl`
- `results/`: metrics, prediction comparison, plots, error analysis, and submission output
- `sql/`: SQL scripts for table creation and analysis
- `notebooks/`: optional EDA and SQL notebooks

## Scripts

- `preprocessing.py`
  - Loads raw data
  - Applies basic cleaning rules
  - Exports `data/processed/processed_train.csv` and `data/processed/processed_test.csv` for reporting

- `building_models.py`
  - Trains multiple models with `sklearn.Pipeline(preprocessor + model)`
  - Evaluates on validation split
  - Saves all models to `models/`
  - Saves best model as `models/best_model.pkl`
  - Writes metrics and test prediction comparison to `results/`

- `model_evaluation.py`
  - Reloads saved models
  - Re-evaluates with the same split strategy
  - Generates plots in `results/`
  - Exports `results/error_analysis.csv` for residual/error inspection

- `predict_submission.py`
  - Loads `models/best_model.pkl`
  - Predicts on raw `test.csv`
  - Exports `results/submission.csv`

## Run Order

1. Put `train.csv` and `test.csv` into `data/raw/`.
2. Run preprocessing for reporting artifacts:

```bash
python preprocessing.py
```

3. Train and select best model:

```bash
python building_models.py
```

4. Evaluate and export charts:

```bash
python model_evaluation.py
```

5. Create final submission:

```bash
python predict_submission.py
```

## Leakage Control

- The training workflow uses raw data and fits preprocessing inside each model pipeline on training folds.
- `processed_train.csv` is generated for reporting convenience only, not as the source for model training.
