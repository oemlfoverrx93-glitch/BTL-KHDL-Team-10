import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from catboost import CatBoostRegressor, Pool

# =========================
# 1. PATH CONFIGURATION
# =========================
BASE_DIR = Path(__file__).parent.parent
REPORTS_DIR = BASE_DIR / "reports" / "tables"
OUTPUT_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = REPORTS_DIR / "train_after_domain_cleaning.csv"
TEST_PATH = REPORTS_DIR / "test_after_domain_cleaning.csv"

# =========================
# 2. LOAD CLEAN DATA
# =========================
print("Loading data...")
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

# =========================
# 3. FEATURE ENGINEERING
# =========================
print("Feature engineering...")
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Tổng diện tích sử dụng
    if {"TotalBsmtSF", "1stFlrSF", "2ndFlrSF"}.issubset(out.columns):
        out["TotalSF"] = out["TotalBsmtSF"] + out["1stFlrSF"] + out["2ndFlrSF"]

    # Tuổi nhà
    if {"YrSold", "YearBuilt"}.issubset(out.columns):
        out["HouseAge"] = out["YrSold"] - out["YearBuilt"]

    # Tuổi sau cải tạo
    if {"YrSold", "YearRemodAdd"}.issubset(out.columns):
        out["RemodAge"] = out["YrSold"] - out["YearRemodAdd"]

    # Tổng số phòng tắm quy đổi
    if {"FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath"}.issubset(out.columns):
        out["TotalBath"] = (
            out["FullBath"]
            + 0.5 * out["HalfBath"]
            + out["BsmtFullBath"]
            + 0.5 * out["BsmtHalfBath"]
        )

    # Tổng diện tích porch
    porch_cols = [c for c in ["OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch"] if c in out.columns]
    if porch_cols:
        out["TotalPorchSF"] = out[porch_cols].sum(axis=1)

    # Binary engineered features
    if "GarageArea" in out.columns:
        out["HasGarage"] = (out["GarageArea"] > 0).astype(int)
    if "TotalBsmtSF" in out.columns:
        out["HasBasement"] = (out["TotalBsmtSF"] > 0).astype(int)
    if "Fireplaces" in out.columns:
        out["HasFireplace"] = (out["Fireplaces"] > 0).astype(int)
    if "PoolArea" in out.columns:
        out["HasPool"] = (out["PoolArea"] > 0).astype(int)

    # Chỉ số chất lượng tổng hợp
    quality_cols = [c for c in ["OverallQual", "OverallCond"] if c in out.columns]
    if quality_cols:
        out["QualityIndex"] = out[quality_cols].mean(axis=1)

    return out

train = add_features(train)
test = add_features(test)

# =========================
# 4. TARGET TRANSFORM
# =========================
y = np.log1p(train["SalePrice"])

# =========================
# 5. DROP ID + TARGET
# =========================
X = train.drop(columns=["SalePrice", "Id"], errors="ignore").copy()
X_test = test.drop(columns=["Id"], errors="ignore").copy()

# =========================
# 6. FEATURE GROUPS
# =========================
# (Sử dụng danh sách chính xác từ Ames)
ordinal_cols = [
    "ExterQual", "ExterCond", "KitchenQual", "HeatingQC",
    "BsmtQual", "BsmtCond", "BsmtExposure",
    "BsmtFinType1", "BsmtFinType2",
    "GarageQual", "GarageCond",
    "FireplaceQu", "PoolQC", "GarageFinish",
    "LotShape", "LandSlope"
]

nominal_cols = [
    "MSZoning", "Neighborhood", "Street", "LandContour",
    "LotConfig", "Utilities", "LotShape", "LandSlope",
    "Condition1", "Condition2", "BldgType", "HouseStyle",
    "RoofStyle", "RoofMatl", "Exterior1st", "Exterior2nd",
    "MasVnrType", "Foundation", "Heating", "Electrical",
    "GarageType", "SaleType", "SaleCondition", "PavedDrive"
]

special_coded_cols = [
    "MSSubClass", "MoSold"
]

binary_cols = [
    "CentralAir"
]

# =========================
# 7. ORDINAL MAPPING
# =========================
print("Applying mappings...")
ordinal_maps = {
    "ExterQual":     {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "ExterCond":     {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "KitchenQual":   {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "HeatingQC":     {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "BsmtQual":      {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "BsmtCond":      {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "GarageQual":    {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "GarageCond":    {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "FireplaceQu":   {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "PoolQC":        {"None": 0, "Fa": 1, "TA": 2, "Gd": 3, "Ex": 4},
    "GarageFinish":  {"None": 0, "Unf": 1, "RFn": 2, "Fin": 3},
    "BsmtExposure":  {"None": 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4},
    "BsmtFinType1":  {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
    "BsmtFinType2":  {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
    "LotShape":      {"IR3": 1, "IR2": 2, "IR1": 3, "Reg": 4},
    "LandSlope":     {"Sev": 1, "Mod": 2, "Gtl": 3},
}

def apply_ordinal_mapping(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col, mp in ordinal_maps.items():
        if col in out.columns:
            out[col] = out[col].fillna("None").map(mp)
    return out

X = apply_ordinal_mapping(X)
X_test = apply_ordinal_mapping(X_test)

# =========================
# 8. BINARY FEATURES
# =========================
binary_maps = {
    "CentralAir": {"N": 0, "Y": 1}
}

for col, mp in binary_maps.items():
    if col in X.columns:
        X[col] = X[col].map(mp)
    if col in X_test.columns:
        X_test[col] = X_test[col].map(mp)

# =========================
# 9. CATEGORICAL FEATURES FOR CATBOOST
# =========================
cat_features = [c for c in (nominal_cols + special_coded_cols) if c in X.columns]
remaining_object_cols = X.select_dtypes(include=["object"]).columns.tolist()

for c in remaining_object_cols:
    if c not in cat_features and c not in ordinal_maps and c not in binary_maps:
        cat_features.append(c)

for col in cat_features:
    X[col] = X[col].astype(str).fillna("None")
    X_test[col] = X_test[col].astype(str).fillna("None")
for col in cat_features:
        X[col] = X[col].astype(str).fillna("None")
        X_test[col] = X_test[col].astype(str).fillna("None")

    # --- CHÈN VÀO ĐÂY ---
print(f"\n--- THỐNG KÊ SỐ CHIỀU (CATBOOST PIPELINE) ---")
print(f"Số cột gốc (Raw Features): {train.drop(columns=['SalePrice', 'Id'], errors='ignore').shape[1]}")
print(f"Số cột sau khi xử lý (X): {X.shape[1]}")
print("------------------------------------------\n")
# =========================
# 10. SAVE PROCESSED ML-READY DATA
# =========================
print("Saving processed files...")
train_out = X.copy()
train_out["SalePrice"] = train["SalePrice"]
train_out.to_csv(OUTPUT_DIR / "processed_train_catboost.csv", index=False)
X_test.to_csv(OUTPUT_DIR / "processed_test_catboost.csv", index=False)

# =========================
# 11. SPLIT TRAIN / VALID
# =========================
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 12. BUILD POOLS
# =========================
train_pool = Pool(data=X_train, label=y_train, cat_features=cat_features)
valid_pool = Pool(data=X_valid, label=y_valid, cat_features=cat_features)
test_pool = Pool(data=X_test, cat_features=cat_features)

# =========================
# 13. TRAIN CATBOOST ON VALIDATION SET
# =========================
print("Training CatBoost on Validation set...")
model = CatBoostRegressor(
    loss_function="RMSE",
    eval_metric="RMSE",
    iterations=5000,
    learning_rate=0.03,
    depth=8,
    l2_leaf_reg=3,
    random_seed=42,
    od_type="Iter",
    od_wait=200,
    verbose=500
)

model.fit(train_pool, eval_set=valid_pool, use_best_model=True)

# =========================
# 14. VALIDATION METRICS
# =========================
valid_pred_log = model.predict(valid_pool)
valid_pred = np.expm1(valid_pred_log)
y_valid_raw = np.expm1(y_valid)

mae = mean_absolute_error(y_valid_raw, valid_pred)
rmse = np.sqrt(mean_squared_error(y_valid_raw, valid_pred))
r2 = r2_score(y_valid_raw, valid_pred)

print(f"MAE : {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2  : {r2:.4f}")

pd.DataFrame([{"MAE": mae, "RMSE": rmse, "R2": r2}]).to_csv(
    OUTPUT_DIR / "catboost_validation_metrics.csv", index=False
)

# =========================
# 15. FULL TRAIN & PREDICT
# =========================
print("Training final model on full data...")
full_pool = Pool(data=X, label=y, cat_features=cat_features)

final_model = CatBoostRegressor(
    loss_function="RMSE",
    eval_metric="RMSE",
    iterations=model.get_best_iteration() if model.get_best_iteration() else 5000,
    learning_rate=0.03,
    depth=8,
    l2_leaf_reg=3,
    random_seed=42,
    verbose=0
)

final_model.fit(full_pool)
test_pred_log = final_model.predict(test_pool)
test_pred = np.expm1(test_pred_log)

# =========================
# 16. SAVE OUTPUTS
# =========================
print("Saving artifacts...")
# 16.1. Submission
submission = pd.DataFrame({"Id": test["Id"], "SalePrice": test_pred})
submission.to_csv(OUTPUT_DIR / "submission_catboost.csv", index=False)

# 16.2. Model
final_model.save_model(str(OUTPUT_DIR / "catboost_ames.cbm"))

# 16.3. Feature Importance
feature_importance = final_model.get_feature_importance(full_pool)
fi_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": feature_importance
}).sort_values("Importance", ascending=False)
fi_df.to_csv(OUTPUT_DIR / "catboost_feature_importance.csv", index=False)

print("SUCCESS: CatBoost Pipeline completed!")
