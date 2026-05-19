/*
05_create_model_table.sql
Goal: Create final SQL object for modeling/export.
*/

IF OBJECT_ID('dbo.vw_model_data', 'V') IS NOT NULL
    DROP VIEW dbo.vw_model_data;
GO

CREATE VIEW dbo.vw_model_data AS
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
    HouseAge,
    TotalSF,
    SalePrice
FROM dbo.clean_houses_train;
GO

