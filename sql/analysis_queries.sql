-- 1) Basic distribution by neighborhood
SELECT
    Neighborhood,
    COUNT(*) AS total_houses,
    AVG(SalePrice) AS avg_price,
    MIN(SalePrice) AS min_price,
    MAX(SalePrice) AS max_price
FROM house_prices
GROUP BY Neighborhood
ORDER BY avg_price DESC;

-- 2) Price by overall quality
SELECT
    OverallQual,
    COUNT(*) AS total_houses,
    AVG(SalePrice) AS avg_price
FROM house_prices
GROUP BY OverallQual
ORDER BY OverallQual DESC;

-- 3) Living area impact
SELECT
    TOP 20
    Id,
    GrLivArea,
    SalePrice,
    (SalePrice * 1.0 / NULLIF(GrLivArea, 0)) AS price_per_sqft
FROM house_prices
ORDER BY price_per_sqft DESC;

-- 4) Build year trend
SELECT
    YearBuilt,
    COUNT(*) AS total_houses,
    AVG(SalePrice) AS avg_price
FROM house_prices
GROUP BY YearBuilt
ORDER BY YearBuilt;
