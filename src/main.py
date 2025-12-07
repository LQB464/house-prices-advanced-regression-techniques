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
from .preprocessing import Preprocessor, build_feature_pipeline
import argparse

def parser_args():
    parser = argparse.ArgumentParser(description="House Prices Advanced Regression Techniques")
    parser.add_argument("--target-col", type=str, default="SalePrice", help="Name of the target column in the dataset")
    parser.add_argument("--test-size", type=float, default=0.2, help="Proportion of the dataset to include in the test split")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--data-path", type=Path, default=Path("./data/house_prices.csv"), help="Path to the dataset CSV file")
    parser.add_argument("--output-dir", type=Path, default=Path("./output/"), help="Directory to save output results")
    return parser.parse_args()

def main():
    
    args = parser_args()
    trainer = ModelTrainer(
        target_col=args.target_col,
        test_size=args.test_size,
        random_state=args.random_state,
        data_path=args.data_path,
    )

    dataset = ModelTrainer.load_data(args.data_path)
    X_train, X_test, y_train, y_test = trainer.split_data(dataset)
    
    preprocessor = Preprocessor()
    

if __name__ == "__main__":
    main()
