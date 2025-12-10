"""
modeling/persistence_mixin.py

Utilities for saving and loading trained models, exporting evaluation results,
and generating comparison plots for model performance.

Extended Description
--------------------
This module defines the `PersistenceMixin`, which standardizes how models and
evaluation artifacts are persisted during the training workflow. It supports:
- saving fitted models to disk via joblib
- loading serialized models back into the trainer
- exporting evaluation metrics as CSV files
- generating bar charts comparing model RMSE scores

The mixin expects to be combined with TrainerConfig, which provides:
`self.models_`, `self.results_`, `self.output_dir`, and a shared logger.

Main Components
---------------
- save_model: serialize and store fitted model
- load_model: load a saved model from disk
- save_results: export metrics and produce RMSE visualizations

Usage Example
-------------
>>> class Trainer(TrainerConfig, PersistenceMixin):
...     pass
>>> trainer = Trainer()
>>> trainer.save_model("ridge")
>>> trainer.load_model("model_outputs/ridge.joblib")
>>> trainer.save_results()

Notes
-----
Plots and CSV files are stored inside the configured output directory.
The mixin gracefully handles missing results via warnings.
"""

from typing import Optional

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class PersistenceMixin:
    """
    Model and result persistence utilities for the training workflow.

    Extended Description
    --------------------
    The `PersistenceMixin` provides standardized persistence operations:
    - saving fitted models via joblib
    - loading serialized models with optional renaming
    - exporting evaluation metrics as a CSV file
    - generating RMSE bar charts when available

    It relies on attributes supplied by TrainerConfig:
    `self.models_`, `self.results_`, `self.output_dir`, and `self.logger`.

    Parameters
    ----------
    None
        This mixin introduces no constructor; TrainerConfig provides state.

    Attributes
    ----------
    models_ : dict
        Registry of trained model pipelines.
    results_ : dict
        Evaluation metrics keyed by model name.
    output_dir : str or Path
        Directory where persistence outputs are stored.

    Examples
    --------
    >>> trainer.save_model("xgb")
    >>> trainer.load_model("model_outputs/xgb.joblib")
    >>> trainer.save_results()
    """

    def save_model(self, name: str) -> None:
        if name not in self.models_:
            raise ValueError(f"Model '{name}' not found in self.models_.")
        path = Path(self.output_dir) / f"{name}.joblib"
        joblib.dump(self.models_[name], path)
        self.logger.info(f"Saved model '{name}' to {path}")

    def load_model(self, path: str, name: Optional[str] = None) -> None:
        model = joblib.load(path)
        key = name if name is not None else Path(path).stem
        self.models_[key] = model
        self.logger.info(f"Loaded model from {path} as '{key}'")

    def save_results(self) -> None:
        if not self.results_:
            self.logger.warning("No results to save.")
            return

        df = pd.DataFrame(self.results_).T
        csv_path = Path(self.output_dir) / "model_results.csv"
        df.to_csv(csv_path, index=True)
        self.logger.info(f"Saved model results to {csv_path}")
        
        # Plot RMSE comparison if available
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
