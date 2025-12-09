# src/modeling/persistence_mixin.py

from typing import Optional

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class PersistenceMixin:
    """
    Save, load model và kết quả.

    Yêu cầu:
        self.models_, self.results_, self.output_dir, self.logger
    """

    def save_model(self, name: str) -> None:
        if name not in self.models_:
            raise ValueError(f"Model '{name}' not found in self.models_.")
        path = Path(self.output_dir) / f"{name}.joblib"
        joblib.dump(self.models_[name], path)
        self.logger.info(f"Saved model '{name}' to {path}")

    def load_model(self, path: str, name: Optional[str] = None) -> None:
        """
        Load model từ file joblib và thêm vào self.models_ với key name.
        """
        model = joblib.load(path)
        key = name if name is not None else Path(path).stem
        self.models_[key] = model
        self.logger.info(f"Loaded model from {path} as '{key}'")

    def save_results(self) -> None:
        """
        Lưu bảng kết quả models và vẽ biểu đồ RMSE.
        """
        if not self.results_:
            self.logger.warning("No results to save.")
            return

        df = pd.DataFrame(self.results_).T
        csv_path = Path(self.output_dir) / "model_results.csv"
        df.to_csv(csv_path, index=True)
        self.logger.info(f"Saved model results to {csv_path}")

        # Vẽ bar chart theo test_rmse nếu có
        if "test_rmse" in df.columns:
            df_plot = df.sort_values("test_rmse")
            plt.figure(figsize=(10, 5))
            plt.bar(df_plot.index.astype(str), df_plot["test_rmse"])
            plt.xticks(rotation=45, ha="right")
            plt.ylabel("Test RMSE")
            plt.title("Model comparison by Test RMSE")
            plt.tight_layout()
            fig_path = Path(self.output_dir) / "rmse_comparison.png"
            plt.savefig(fig_path)
            plt.close()
            self.logger.info(f"Saved RMSE comparison plot to {fig_path}")
