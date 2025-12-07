"""
main.py

Script chạy chính của dự án:
- Phần 1: Tiền xử lý dữ liệu (DataPreprocessor)
- Phần 2: Huấn luyện & tối ưu mô hình học máy (ModelTrainer)
- Phần 3: Trực quan hóa & phân tích dữ liệu (EDAVisualizer)
"""

from pathlib import Path
import pandas as pd

from modeling import ModelTrainer
from eda_utils import EDAVisualizer 


def main():
    # Đường dẫn tuyệt đối của file main.py
    this_file = Path(__file__).resolve()

    # Thư mục Code/
    code_dir = this_file.parent

    # Thư mục gốc project: house-prices-advanced-regression-techniques/
    project_root = code_dir.parent

    # File CSV trong thư mục Data/
    csv_path = project_root / "Data" / "train.csv"

    # Output sẽ được ghi vào thư mục gốc
    output_dir = project_root / "model_outputs"
    eda_dir = project_root / "eda_plots"

    print("=== STEP 1: MODEL TRAINING PIPELINE ===")

    trainer = ModelTrainer(
        target_col="SalePrice",
        test_size=0.2,
        random_state=42,
        output_dir=output_dir,   # <----- quan trọng
    )

    results = trainer.run(
        csv_path=str(csv_path),
        tune_optuna=True
    )

    print("Training completed.")
    print(results)

    print("\n=== STEP 2: RUN EDA ===")

    df = pd.read_csv(csv_path)
    eda = EDAVisualizer(df, target_col="SalePrice", output_dir=eda_dir)

    eda.plot_target_distribution()
    eda.plot_missing_values()
    eda.plot_numeric_histograms()
    eda.plot_correlation_heatmap()
    eda.plot_boxplots_for_top_categories("Neighborhood")

    print(f"EDA completed. Charts saved to {eda_dir}")

    print("\nDONE!")


if __name__ == "__main__":
    main()
