# So sánh các chiến lược Encoding

| Method | Ưu điểm | Nhược điểm | Dùng khi nào |
|---|---|---|---|
| One-Hot Encoding | Tuyệt đối an toàn, không giả định thứ tự, được ML Models ưa chuộng | Gây ra lời nguyền đa chiều (Dimensional Explosion) | Nominal Categorical, Low Cardinality |
| Target Encoding | Giữ nguyên số chiều, correlation cao với target | Dễ bị Data Leakage và Overfitting nếu không kfold | High Cardinality Nominal Categories |
| Ordinal Encoding | Giữ được thông tin phân cấp (rank), nhỏ gọn | Phải tự gán map logic cẩn thận, dễ sai sót | Ordinal Categorical (Qual, Cond, etc.) |
| CatBoost/WOE | Rất mạnh, chống overfit tốt hơn Target Encoding | Triển khai phức tạp, chạy chậm | Data cạnh tranh Kaggle, Boosting models |
