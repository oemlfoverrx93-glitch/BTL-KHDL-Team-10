# SQL Pipeline Guide

Run scripts in this exact order:

1. `01_import_data.sql`
2. `02_data_cleaning.sql`
3. `03_feature_engineering.sql`
4. `04_analysis_queries.sql`
5. `05_create_model_table.sql`

## Output for Python
After step 5, export `vw_model_data` (or `clean_houses_train`) to:

- `data/exported/clean_house_data.csv`

This file is used by:
- `scripts/train_model.py`
- `scripts/model_evaluation.py`

