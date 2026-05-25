# Ames EDA Workflow Summary

- Data source: CSV:train (data\raw\train.csv); CSV:test (data\raw\test.csv)
- Train shape: 1460 x 85
- Test shape: 1459 x 84
- Domain outliers found: 2
- Downstream analysis uses outlier-removed train: False
- SalePrice skewness: 1.8829 (log1p: 0.1213)
- SalePrice excess kurtosis: 6.5363 (log1p: 0.8095)
- Jarque-Bera p-value: 0.000000 (log1p: 0.000000)

## Outlier Sensitivity (Linear Regression Baseline)
        Scenario  Rows          MAE         RMSE       R2  OutlierCount
   with_outliers  1460 28121.524152 43032.150480 0.758581             2
without_outliers  1458 26877.315105 35091.580124 0.777067             0

## Top Pearson Correlation with SalePrice
     Feature  PearsonCorr
 OverallQual     0.790982
     TotalSF     0.782260
   GrLivArea     0.708624
  GarageCars     0.640409
   TotalBath     0.631731
  GarageArea     0.623431
 TotalBsmtSF     0.613581
    1stFlrSF     0.605852
    FullBath     0.560664
TotRmsAbvGrd     0.533723

## Top Spearman Correlation with SalePrice
    Feature  SpearmanCorr
    TotalSF      0.819679
OverallQual      0.809829
  GrLivArea      0.731310
  TotalBath      0.703731
 GarageCars      0.690711
  YearBuilt      0.652682
 GarageArea      0.649379
   FullBath      0.635957
TotalBsmtSF      0.602725
   1stFlrSF      0.575408

## Top Mutual Information Features
     Feature  MI_Score
     TotalSF  0.679139
 OverallQual  0.564680
Neighborhood  0.511182
   GrLivArea  0.482876
  GarageCars  0.367782
 TotalBsmtSF  0.367548
   YearBuilt  0.363437
  GarageArea  0.360749
    HouseAge  0.339541
   TotalBath  0.336898

## Residual Metrics
- MAE: 21747.28
- RMSE: 34894.95
- R2: 0.8413
- Heteroscedasticity proxy (corr(abs residual, predicted)) Pearson=0.4347 (p=0.000000), Spearman=0.1810 (p=0.001897)

## Top Multicollinearity Pairs (|Pearson| >= 0.8)
    FeatureA     FeatureB  AbsPearsonCorr
   YearBuilt     HouseAge        0.999036
YearRemodAdd     RemodAge        0.997930
  GarageCars   GarageArea        0.882475
   GrLivArea      TotalSF        0.874373
 TotalBsmtSF      TotalSF        0.826742
   GrLivArea TotRmsAbvGrd        0.825489
 TotalBsmtSF     1stFlrSF        0.819530
    1stFlrSF      TotalSF        0.800350

## Linear Model Interpretability (Top |Coefficient|)
     Feature    Coefficient  AbsCoefficient
      PoolQC -107732.125521   107732.125521
   Utilities  -54242.011862    54242.011862
      Street   21594.239449    21594.239449
KitchenAbvGr  -12404.221182    12404.221182
   LandSlope   11366.899438    11366.899438
  GarageCars   11170.136353    11170.136353
  Condition2  -10784.067421    10784.067421
 OverallQual   10640.733022    10640.733022
 KitchenQual   -9428.942963     9428.942963
   ExterQual   -9086.770272     9086.770272

## Residual Hotspots by Neighborhood
Neighborhood
NoRidge    66533.070562
StoneBr    56724.135315
NridgHt    46984.441145
Crawfor    30060.581756
ClearCr    29710.957008

## Residual Hotspots by OverallQual
OverallQual
10    94407.121698
9     64061.553763
8     29090.888600
3     22823.171965
7     20934.834354

## Residual Hotspots by Price Segment
PriceSegment
Luxury_10pct        55377.621865
UpperMid_40pct      19144.786559
Mainstream_50pct    16904.672804