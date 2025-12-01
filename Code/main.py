"""
main.py

Script chay chinh cua du an:
- Phan 1: Tien xu ly du lieu (DataPreprocessor)
- Phan 2: Mo hinh hoc may (ModelTrainer)
- Phan 3: Truc quan hoa (EDAVisualizer) - optional

Chay: python main.py
"""

from model_trainer import ModelTrainer
from eda_utils import EDAVisualizer
import pandas as pd


def main():
    csv_path = "train-house-prices-advanced-regression-techniques.csv"

    print("=== STEP 1: MODEL TRAINING PIPELINE ===")
    trainer = ModelTrainer(
        target_col="SalePrice",
        test_size=0.2,
        random_state=42,
        output_dir="model_outputs"
    )

    # Run toàn bộ training + tuning
    results = trainer.run(csv_path=csv_path, tune_optuna=True)
    print("Training completed.")
    print(results)

    print("\n=== STEP 2: RUN EDA (Optional) ===")
    df = pd.read_csv(csv_path)
    eda = EDAVisualizer(df, target_col="SalePrice", output_dir="eda_plots")

    eda.plot_target_distribution()
    eda.plot_missing_values()
    eda.plot_numeric_histograms()
    eda.plot_correlation_heatmap()
    eda.plot_boxplots_for_top_categories("Neighborhood")

    print("EDA completed. Charts saved to eda_plots/")

    print("\nDone!")


if __name__ == "__main__":
    main()