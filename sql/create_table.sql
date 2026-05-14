-- SQL Server / Generic SQL template

CREATE TABLE house_prices (
    Id INT PRIMARY KEY,
    MSSubClass INT,
    MSZoning VARCHAR(50),
    LotFrontage FLOAT,
    LotArea INT,
    Street VARCHAR(20),
    Neighborhood VARCHAR(100),
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
