import os
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge

# --- CONFIGURATION ---
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
REPORTS_DIR = BASE_DIR / "reports" / "tables"
OUTPUT_DIR_DATA = BASE_DIR / "data" / "processed"
OUTPUT_DIR_DATA.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = REPORTS_DIR / "train_after_domain_cleaning.csv"
TEST_PATH = REPORTS_DIR / "test_after_domain_cleaning.csv"
FEATURE_TYPES_PATH = REPORTS_DIR / "feature_type_classification.csv"

# --- CUSTOM TRANSFORMERS ---
class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Creates new features from existing columns.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_new = X.copy()
        
        # Numeric Features
        if all(c in X.columns for c in ["TotalBsmtSF", "1stFlrSF", "2ndFlrSF"]):
            X_new["TotalSF"] = X_new["TotalBsmtSF"] + X_new["1stFlrSF"] + X_new["2ndFlrSF"]
            
        if all(c in X.columns for c in ["YrSold", "YearBuilt"]):
            X_new["HouseAge"] = X_new["YrSold"] - X_new["YearBuilt"]
            
        if all(c in X.columns for c in ["YrSold", "YearRemodAdd"]):
            X_new["RemodAge"] = X_new["YrSold"] - X_new["YearRemodAdd"]
            
        if all(c in X.columns for c in ["FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath"]):
            X_new["TotalBath"] = X_new["FullBath"] + 0.5 * X_new["HalfBath"] + X_new["BsmtFullBath"] + 0.5 * X_new["BsmtHalfBath"]
            
        # Binary Features
        if "GarageArea" in X.columns:
            X_new["HasGarage"] = (X_new["GarageArea"] > 0).astype(int)
        if "TotalBsmtSF" in X.columns:
            X_new["HasBasement"] = (X_new["TotalBsmtSF"] > 0).astype(int)
        if "Fireplaces" in X.columns:
            X_new["HasFireplace"] = (X_new["Fireplaces"] > 0).astype(int)
            
        return X_new

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            raise ValueError("input_features must be provided")
        new_feats = ["TotalSF", "HouseAge", "RemodAge", "TotalBath", "HasGarage", "HasBasement", "HasFireplace"]
        return np.array(list(input_features) + new_feats)


class SpecialTypeConverter(BaseEstimator, TransformerMixin):
    """
    Converts special coded variables to string so they can be treated as Nominal Categorical.
    """
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_new = X.copy()
        for col in self.columns:
            if col in X_new.columns:
                X_new[col] = X_new[col].astype(str)
        return X_new


class BinaryMapper(BaseEstimator, TransformerMixin):
    """
    Maps specific binary string values to 0/1.
    """
    def __init__(self, mapping_dict):
        self.mapping_dict = mapping_dict

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_new = X.copy()
        for col, mapping in self.mapping_dict.items():
            if col in X_new.columns:
                X_new[col] = X_new[col].map(mapping).fillna(0).astype(int)
        return X_new

    def get_feature_names_out(self, input_features=None):
        return np.array(list(self.mapping_dict.keys()))


# --- MAIN PIPELINE SCRIPT ---
def main():
    print("1. Loading data...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    
    # Read mappings if needed
    ft_map = pd.read_csv(FEATURE_TYPES_PATH)
    
    # Target and ID
    target = "SalePrice"
    
    y_train = np.log1p(train_df[target]) if target in train_df.columns else None
    
    # Drop target and Id from features
    X_train_raw = train_df.drop(columns=[target, "Id"], errors="ignore")
    X_test_raw = test_df.drop(columns=["Id"], errors="ignore")
    
    # Define Column groupings
    ordinal_cols = ft_map[ft_map['Type'] == 'Ordinal Categorical']['Feature'].tolist()
    ordinal_cols = [c for c in ordinal_cols if c in X_train_raw.columns]
    
    nominal_cols = ft_map[ft_map['Type'] == 'Nominal Categorical']['Feature'].tolist()
    special_cols = ft_map[ft_map['Type'] == 'Special Coded Category']['Feature'].tolist()
    all_nominal_cols = [c for c in nominal_cols + special_cols if c in X_train_raw.columns]
    
    binary_cols = ft_map[ft_map['Type'] == 'Binary']['Feature'].tolist()
    binary_cols = [c for c in binary_cols if c in X_train_raw.columns]
    
    numeric_cont = ft_map[ft_map['Type'] == 'Numerical Continuous']['Feature'].tolist()
    numeric_count = ft_map[ft_map['Type'] == 'Numerical Count']['Feature'].tolist()
    # Thêm các biến tự tạo (Engineered) vào danh sách numeric
    engineered_numeric = ["TotalSF", "HouseAge", "RemodAge", "TotalBath"]
    engineered_binary = ["HasGarage", "HasBasement", "HasFireplace"]
    all_numeric_cols = [c for c in numeric_cont + numeric_count if c in X_train_raw.columns] + engineered_numeric
    
    # --- BUILD PIPELINE ---
    print("2. Building Pipeline...")
    
    # Pre-processing binary mappings
    binary_mapping = {
        "CentralAir": {"Y": 1, "N": 0},
        "Street": {"Pave": 1, "Grvl": 0}
    }
    
    # Ordinal Mapping
    ordinal_cats = ["None", "Po", "Fa", "TA", "Gd", "Ex"]
    ordinal_categories = [ordinal_cats for _ in ordinal_cols]
    
    # Column Transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), all_numeric_cols),
            ("ord", OrdinalEncoder(categories=ordinal_categories, handle_unknown='use_encoded_value', unknown_value=-1), ordinal_cols),
            ("nom", OneHotEncoder(handle_unknown='ignore', sparse_output=False), all_nominal_cols),
            ("bin", BinaryMapper(binary_mapping), binary_cols),
            ("eng_bin", "passthrough", engineered_binary) # Engineered binary pass through
        ],
        remainder="drop" # Bỏ qua các cột không xác định
    )

    # Full Pipeline
    pipeline = Pipeline([
        ('feat_eng', FeatureEngineer()),
        ('special_conv', SpecialTypeConverter(columns=special_cols)),
        ('preprocessor', preprocessor)
    ])
    
    # --- FIT AND TRANSFORM ---
    print("3. Executing Pipeline (Fit/Transform)...")
    X_train_trans = pipeline.fit_transform(X_train_raw)
    X_test_trans = pipeline.transform(X_test_raw)
    
    # --- CHÈN LỆNH KIỂM TRA TẠI ĐÂY ---
    print(f"\n--- THỐNG KÊ SỐ CHIỀU (DIMENSIONALITY) ---")
    print(f"Số cột gốc (Raw Features): {X_train_raw.shape[1]}")
    print(f"Số cột sau khi Encode: {X_train_trans.shape[1]}")
    print(f"Số lượng cột tăng thêm: {X_train_trans.shape[1] - X_train_raw.shape[1]}")
    print("------------------------------------------\n")
    # -----------------------------------
    # Get feature names
    print("4. Extracting feature names...")
    # Get names from preprocessor
    num_names = all_numeric_cols
    ord_names = ordinal_cols
    nom_names = pipeline.named_steps['preprocessor'].named_transformers_['nom'].get_feature_names_out(all_nominal_cols)
    bin_names = binary_cols
    eng_bin_names = engineered_binary
    
    all_feature_names = num_names + ord_names + list(nom_names) + bin_names + eng_bin_names
    
    # Convert to DataFrames
    train_after_df = pd.DataFrame(X_train_trans, columns=all_feature_names, index=X_train_raw.index)
    test_after_df = pd.DataFrame(X_test_trans, columns=all_feature_names, index=X_test_raw.index)
    
    # Add target back for ML Ready file
    if target in train_df.columns:
        train_after_df[target] = train_df[target]
    
    # Output 1, 2, 3: ML READY
    print("5. Generating ML Ready Outputs...")
    train_after_df.to_csv(OUTPUT_DIR_DATA / "train_after_feature_engineering.csv", index=False)
    test_after_df.to_csv(OUTPUT_DIR_DATA / "test_after_feature_engineering.csv", index=False)
    joblib.dump(pipeline, OUTPUT_DIR_DATA / "preprocessing_pipeline.pkl")
    
    # Output 4: engineered_features_reference.csv
    print("6. Generating Documentation Outputs...")
    engineered_features = pd.DataFrame([
        {"Feature": "TotalSF", "Formula": "TotalBsmtSF + 1stFlrSF + 2ndFlrSF", "Meaning": "Tổng diện tích sàn và hầm"},
        {"Feature": "HouseAge", "Formula": "YrSold - YearBuilt", "Meaning": "Tuổi thọ ngôi nhà khi bán"},
        {"Feature": "RemodAge", "Formula": "YrSold - YearRemodAdd", "Meaning": "Số năm từ lần cải tạo cuối"},
        {"Feature": "TotalBath", "Formula": "FullBath + 0.5*HalfBath + BsmtFullBath + 0.5*BsmtHalfBath", "Meaning": "Tổng số phòng tắm quy đổi"},
        {"Feature": "HasGarage", "Formula": "GarageArea > 0", "Meaning": "Nhà có garage (1/0)"},
        {"Feature": "HasBasement", "Formula": "TotalBsmtSF > 0", "Meaning": "Nhà có tầng hầm (1/0)"},
        {"Feature": "HasFireplace", "Formula": "Fireplaces > 0", "Meaning": "Nhà có lò sưởi (1/0)"}
    ])
    engineered_features.to_csv(REPORTS_DIR / "engineered_features_reference.csv", index=False)
    
    # Output 5: ordinal_mapping_reference.csv
    ordinal_map = pd.DataFrame([
        {"Original": "Ex", "Encoded": 5},
        {"Original": "Gd", "Encoded": 4},
        {"Original": "TA", "Encoded": 3},
        {"Original": "Fa", "Encoded": 2},
        {"Original": "Po", "Encoded": 1},
        {"Original": "None", "Encoded": 0}
    ])
    ordinal_map.to_csv(REPORTS_DIR / "ordinal_mapping_reference.csv", index=False)
    
    # Output 6: encoding_strategy_comparison.md
    with open(REPORTS_DIR / "encoding_strategy_comparison.md", "w", encoding="utf-8") as f:
        f.write("# So sánh các chiến lược Encoding\n\n")
        f.write("| Method | Ưu điểm | Nhược điểm | Dùng khi nào |\n")
        f.write("|---|---|---|---|\n")
        f.write("| One-Hot Encoding | Tuyệt đối an toàn, không giả định thứ tự, được ML Models ưa chuộng | Gây ra lời nguyền đa chiều (Dimensional Explosion) | Nominal Categorical, Low Cardinality |\n")
        f.write("| Target Encoding | Giữ nguyên số chiều, correlation cao với target | Dễ bị Data Leakage và Overfitting nếu không kfold | High Cardinality Nominal Categories |\n")
        f.write("| Ordinal Encoding | Giữ được thông tin phân cấp (rank), nhỏ gọn | Phải tự gán map logic cẩn thận, dễ sai sót | Ordinal Categorical (Qual, Cond, etc.) |\n")
        f.write("| CatBoost/WOE | Rất mạnh, chống overfit tốt hơn Target Encoding | Triển khai phức tạp, chạy chậm | Data cạnh tranh Kaggle, Boosting models |\n")
        
    # Output 7: encoded_feature_dimension.csv
    dim_data = pd.DataFrame([
        {"Stage": "Raw Data", "Num Features": X_train_raw.shape[1]},
        {"Stage": "After Feature Engineering", "Num Features": X_train_raw.shape[1] + len(engineered_numeric) + len(engineered_binary)},
        {"Stage": "After Encoding & Pipeline", "Num Features": len(all_feature_names)}
    ])
    dim_data.to_csv(REPORTS_DIR / "encoded_feature_dimension.csv", index=False)
    
    # Output 8: preprocessing_structure.txt
    with open(REPORTS_DIR / "preprocessing_structure.txt", "w", encoding="utf-8") as f:
        f.write("--- BLUEPRINT PREPROCESSING ---\n\n")
        f.write("1. Numeric Features (Continuous & Count):\n")
        f.write("   - StandardScaler\n")
        f.write(f"   - Cols: {len(all_numeric_cols)}\n\n")
        f.write("2. Ordinal Features:\n")
        f.write("   - OrdinalEncoder (Ex=5, Gd=4...)\n")
        f.write(f"   - Cols: {len(ordinal_cols)}\n\n")
        f.write("3. Nominal & Special Coded Features:\n")
        f.write("   - Convert to String -> OneHotEncoder(handle_unknown='ignore')\n")
        f.write(f"   - Cols: {len(all_nominal_cols)}\n\n")
        f.write("4. Binary Features:\n")
        f.write("   - Manual Map (Y=1, N=0, Pave=1, Grvl=0)\n")
        f.write(f"   - Cols: {len(binary_cols)}\n")
        
    # Output 9: transformed_feature_names.csv
    pd.DataFrame({"Transformed_Feature": all_feature_names}).to_csv(REPORTS_DIR / "transformed_feature_names.csv", index=False)
    
    # Output 10: train_test_feature_alignment_check.csv
    train_cols = set(train_after_df.drop(columns=[target], errors='ignore').columns)
    test_cols = set(test_after_df.columns)
    missing_in_test = train_cols - test_cols
    missing_in_train = test_cols - train_cols
    
    alignment_data = pd.DataFrame([
        {"Check": "Same columns count", "Result": "PASS" if len(train_cols) == len(test_cols) else "FAIL"},
        {"Check": "Exactly matching columns", "Result": "PASS" if train_cols == test_cols else "FAIL"},
        {"Check": "Missing cols in test", "Result": len(missing_in_test)},
        {"Check": "Missing cols in train", "Result": len(missing_in_train)}
    ])
    alignment_data.to_csv(REPORTS_DIR / "train_test_feature_alignment_check.csv", index=False)
    
    # Output 11: feature_correlation_after_encoding.csv
    print("7. Calculating correlations...")
    # Calculate correlation matrix on transformed training data
    corr_matrix = train_after_df.corr(numeric_only=True)
    corr_matrix.to_csv(REPORTS_DIR / "feature_correlation_after_encoding.csv")
    
    # Output 12: feature_importance_ready_list.csv
    feature_list = []
    for f in all_feature_names:
        if f in all_numeric_cols:
            type_feat = "Engineered Numeric" if f in engineered_numeric else "Numeric"
        elif f in ordinal_cols:
            type_feat = "Ordinal Encoded"
        elif f in bin_names or f in eng_bin_names:
            type_feat = "Binary"
        else:
            type_feat = "OneHot Encoded"
            
        feature_list.append({"Feature": f, "Type": type_feat})
        
    pd.DataFrame(feature_list).to_csv(REPORTS_DIR / "feature_importance_ready_list.csv", index=False)
    
    print("8. Baseline Model Cross-Validation...")
    model = Ridge(alpha=10.0)
    scores = cross_val_score(model, X_train_trans, y_train, cv=5, scoring='neg_root_mean_squared_error')
    rmse_scores = -scores
    print(f"-> Ridge Baseline CV RMSE (Log SalePrice): {rmse_scores.mean():.4f} +/- {rmse_scores.std():.4f}")
    
    print("\n[SUCCESS] Pipeline execution finished! All 12 professional outputs generated.")
 
if __name__ == "__main__":
    main()
