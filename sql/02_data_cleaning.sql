/*
02_data_cleaning.sql
Goal: Build cleaned training table for downstream feature engineering.
*/

IF OBJECT_ID('dbo.clean_houses_train', 'U') IS NOT NULL
    DROP TABLE dbo.clean_houses_train;

SELECT
    Id,
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
INTO dbo.clean_houses_train
FROM dbo.house_prices_staging
WHERE SalePrice IS NOT NULL
  AND SalePrice > 0
  AND GrLivArea IS NOT NULL
  AND GrLivArea > 0
  AND OverallQual IS NOT NULL
  AND YearBuilt IS NOT NULL;

-- Optional simple outlier filter (adjust thresholds if needed)
DELETE FROM dbo.clean_houses_train
WHERE SalePrice > 1000000
   OR GrLivArea > 5000;

