/*
03_feature_engineering.sql
Goal: Add SQL-engineered features.
*/

IF COL_LENGTH('dbo.clean_houses_train', 'HouseAge') IS NULL
BEGIN
    ALTER TABLE dbo.clean_houses_train ADD HouseAge INT;
END;

IF COL_LENGTH('dbo.clean_houses_train', 'TotalSF') IS NULL
BEGIN
    ALTER TABLE dbo.clean_houses_train ADD TotalSF INT;
END;

UPDATE dbo.clean_houses_train
SET
    HouseAge = YEAR(GETDATE()) - YearBuilt,
    TotalSF = ISNULL(GrLivArea, 0) + ISNULL(TotalBsmtSF, 0);

