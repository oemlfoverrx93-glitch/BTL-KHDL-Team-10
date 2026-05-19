/*
01_import_data.sql
Goal: Import or restore source data into SQL Server.
Adjust paths/server names to your machine.
*/

-- Option A: Restore database from .bak
-- USE master;
-- GO
-- RESTORE DATABASE HousePrices
-- FROM DISK = 'D:\PTIT\Nhap mon KHDL\BTL\DDGN BTL\database\Houseprices.bak'
-- WITH REPLACE;
-- GO

-- Option B: Import CSV to staging table
IF OBJECT_ID('dbo.house_prices_staging', 'U') IS NOT NULL
    DROP TABLE dbo.house_prices_staging;

CREATE TABLE dbo.house_prices_staging (
    Id INT,
    MSSubClass INT,
    MSZoning NVARCHAR(50),
    LotFrontage FLOAT,
    LotArea INT,
    Street NVARCHAR(20),
    Neighborhood NVARCHAR(100),
    OverallQual INT,
    OverallCond INT,
    YearBuilt INT,
    GrLivArea INT,
    GarageCars INT,
    GarageArea INT,
    TotalBsmtSF INT,
    FullBath INT,
    BedroomAbvGr INT,
    TotRmsAbvGrd INT,
    SalePrice FLOAT
);

/*
Example BULK INSERT (uncomment and edit path):
BULK INSERT dbo.house_prices_staging
FROM 'D:\PTIT\Nhap mon KHDL\BTL\DDGN BTL\data\raw\train.csv'
WITH (
    FIRSTROW = 2,
    FIELDTERMINATOR = ',',
    ROWTERMINATOR = '0x0a',
    TABLOCK
);
*/

