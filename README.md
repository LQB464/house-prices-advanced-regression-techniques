# 📝 **README – Hướng dẫn chạy dự án House Price Prediction**

## 📌 **Giới thiệu**

Dự án này triển khai một pipeline khoa học dữ liệu hoàn chỉnh theo yêu cầu đồ án môn *Python cho Khoa học Dữ liệu – K23*, bao gồm:

* **Phần 1:** Tiền xử lý dữ liệu  
* **Phần 2:** Xây dựng và tối ưu mô hình học máy  
* **Phần 3:** Trực quan hóa & phân tích đặc trưng  

Dữ liệu sử dụng: **Ames Housing Dataset** (tập train.csv và test.csv trong thư mục Data).

---

# 📁 **Cấu trúc thư mục**

```

house-prices-advanced-regression-techniques/
│
├── Code/
│   ├── data_preprocessing.py       # Phần 1 – Tiền xử lý (class DataPreprocessor)
│   ├── eda_utils.py                # Phần 3 – EDA & phân tích đặc trưng
│   ├── model_trainer.py            # Phần 2 – Huấn luyện & tối ưu mô hình
│   └── main.py                     # Script chạy chính
│
├── Data/
│   ├── train.csv                   # Dữ liệu huấn luyện
│   ├── test.csv                    # Dữ liệu test của Kaggle (optional)
│   ├── sample_submission.csv       # File nộp Kaggle (optional)
│   └── data_description.txt        # Mô tả biến
│
├── PythonProject_requirement.pdf   # File yêu cầu đồ án
├── README.md                       # Hướng dẫn chạy
└── requirements.txt                # Thư viện cần cài

````

---

# ⚙️ **1. Cài đặt môi trường**

Yêu cầu Python ≥ 3.8.

Chạy:

```bash
pip install -r requirements.txt
````

Danh sách tối thiểu trong `requirements.txt`:

```
numpy
pandas
scikit-learn
optuna
joblib
matplotlib
seaborn
shap
lightgbm
xgboost
catboost
```

---

# 🚀 **2. Chạy toàn bộ pipeline**

Chạy file:

```bash
python main.py
```

Script tự động thực hiện:

1. Load dữ liệu từ thư mục **Data/**
2. Chia train/test
3. Xây dựng pipeline tiền xử lý
4. Huấn luyện các mô hình: Ridge, Lasso, ElasticNet, RandomForest, SVR
5. Tối ưu siêu tham số bằng Optuna
6. Đánh giá mô hình
7. Xuất kết quả + biểu đồ
8. Lưu mô hình tốt nhất dạng `.joblib`

---

# 🧩 **3. Chạy từng phần (nếu cần)**

---

## 🔹 **3.1 Phần 1 – Tiền xử lý dữ liệu**

```python
from data_preprocessing import DataPreprocessor

dp = DataPreprocessor(target_col="SalePrice")
df = dp.load_data("../Data/train.csv")

X, y = dp.split_features_target(df)

dp.build_feature_pipeline(X, X)
X_processed = dp.fit_transform_train(X, y)
```

---

## 🔹 **3.2 Phần 2 – Huấn luyện mô hình**

```python
from model_trainer import ModelTrainer

trainer = ModelTrainer(
    target_col="SalePrice",
    test_size=0.2,
    random_state=42,
    output_dir="model_outputs"
)

trainer.run(
    csv_path="../Data/train.csv",
    tune_optuna=True     # chuyển False nếu không muốn chạy tuning
)
```

Sau khi chạy, thư mục `model_outputs/` sẽ chứa:

* `model_results.csv` – Bảng so sánh RMSE / R2
* `rmse_comparison.png` – Biểu đồ RMSE
* `training.log` – Nhật ký huấn luyện
* `*.joblib` – Mô hình đã huấn luyện (ví dụ: `random_forest_tuned.joblib`)

---

## 🔹 **3.3 Phần 3 – Trực quan hóa & phân tích mô hình**

```python
from eda_utils import EDAVisualizer
import pandas as pd

df = pd.read_csv("../Data/train.csv")

eda = EDAVisualizer(df, target_col="SalePrice", output_dir="eda_plots")

eda.plot_target_distribution()
eda.plot_missing_values()
eda.plot_numeric_histograms()
eda.plot_correlation_heatmap()
eda.plot_boxplots_for_top_categories("Neighborhood")
```

### 🔸 Feature importance / SHAP / PDP

```python
from eda_utils import (
    plot_feature_importance_from_model,
    plot_permutation_importance,
    plot_shap_summary,
    plot_partial_dependence_for_features
)

model = trainer.models_["random_forest"]  # ví dụ

plot_feature_importance_from_model(
    model,
    feature_names=[f"f{i}" for i in range(200)],
    output_path="eda_plots/importance.png"
)

plot_shap_summary(
    model,
    trainer.X_train_,
    output_dir="eda_plots"
)
```

---

# 📊 **4. Các file output quan trọng**

### 📂 `model_outputs/`

* `model_results.csv`
* `rmse_comparison.png`
* `training.log`
* `*.joblib`

### 📂 `eda_plots/`

* `target_distribution.png`
* `missing_values_fraction.png`
* `numeric_histograms.png`
* `correlation_heatmap_subset.png`
* `boxplot_SalePrice_by_Neighborhood.png`
* `importance.png`
* `shap_summary.png`

---

# 💡 **5. Tùy chỉnh khi chạy**

Có thể chỉnh trong `main.py`:

* `test_size`
* `random_state`
* danh sách mô hình cần train
* bật / tắt Optuna
* thêm GridSearchCV
* thêm model mới như LightGBM, CatBoost, XGBoost

---

# 🎓 **6. Phục vụ báo cáo**

Bạn có thể sử dụng các kết quả sau:

* Bảng số liệu: `model_results.csv`
* Biểu đồ RMSE: `rmse_comparison.png`
* Biểu đồ EDA: `eda_plots/*`
* Nhật ký huấn luyện: `training.log`
* Sơ đồ pipeline mô tả DataPreprocessor & ModelTrainer

---

# ✔️ **7. Kết luận**

Project hoàn chỉnh theo đúng yêu cầu đồ án:

* **Phần 1:** Tiền xử lý dữ liệu với `DataPreprocessor`
* **Phần 2:** Huấn luyện & tối ưu mô hình với `ModelTrainer`
* **Phần 3:** Trực quan hóa & giải thích mô hình với `EDAVisualizer`

Cấu trúc rõ ràng, dễ mở rộng và dễ tái sử dụng cho các bài toán dự đoán tương tự.
