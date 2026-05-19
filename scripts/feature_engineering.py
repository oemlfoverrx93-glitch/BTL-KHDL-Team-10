import os
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine
from sqlalchemy.types import Float

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "data" / "exported" / "clean_house_data.csv"

SQL_SERVER = os.getenv("SQL_SERVER", r"localhost\SQLEXPRESS")
SQL_DATABASE = os.getenv("SQL_DATABASE", "HousePrices")

CONNECTION_STRING = (
    f"mssql+pyodbc://@{SQL_SERVER}/{SQL_DATABASE}"
    "?driver=ODBC+Driver+17+for+SQL+Server"
    "&trusted_connection=yes"
)

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

QUERY = """
SELECT
    OverallQual,
    GrLivArea,
    GarageCars,
    TotalBsmtSF,
    FullBath,
    YearBuilt,
    LotArea,
    BedroomAbvGr,
    TotRmsAbvGrd,
    GarageArea,
    SalePrice
FROM clean_houses_train
WHERE SalePrice > 0
"""


def main() -> None:
    engine = create_engine(CONNECTION_STRING)
    df = pd.read_sql(QUERY, engine)
    df = df.dropna().copy()

    scaler = StandardScaler()
    df[FEATURE_COLS] = scaler.fit_transform(df[FEATURE_COLS])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    # Optional SQL output for rechecking with SQL queries.
    df.to_sql(
        "clean_houses_train_scaled",
        engine,
        if_exists="replace",
        index=False,
        dtype={col: Float() for col in FEATURE_COLS},
    )

    print(f"Saved: {OUTPUT_PATH}")
    print("Updated SQL table: clean_houses_train_scaled")


if __name__ == "__main__":
    main()

