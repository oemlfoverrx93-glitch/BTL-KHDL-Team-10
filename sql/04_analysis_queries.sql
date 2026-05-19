/*
04_analysis_queries.sql
Goal: Exploratory SQL queries for report/EDA.
*/

-- 1) Price distribution by neighborhood
SELECT
    Neighborhood,
    COUNT(*) AS total_houses,
    AVG(SalePrice) AS avg_price,
    MIN(SalePrice) AS min_price,
    MAX(SalePrice) AS max_price
FROM dbo.house_prices_staging
GROUP BY Neighborhood
ORDER BY avg_price DESC;

-- 2) Average price by quality
SELECT
    OverallQual,
    COUNT(*) AS total_houses,
    AVG(SalePrice) AS avg_price
FROM dbo.clean_houses_train
GROUP BY OverallQual
ORDER BY OverallQual DESC;

-- 3) Top houses by price per square foot
SELECT TOP 20
    Id,
    GrLivArea,
    SalePrice,
    (SalePrice * 1.0 / NULLIF(GrLivArea, 0)) AS price_per_sqft
FROM dbo.clean_houses_train
ORDER BY price_per_sqft DESC;

-- 4) Build year trend
SELECT
    YearBuilt,
    COUNT(*) AS total_houses,
    AVG(SalePrice) AS avg_price
FROM dbo.clean_houses_train
GROUP BY YearBuilt
ORDER BY YearBuilt;

