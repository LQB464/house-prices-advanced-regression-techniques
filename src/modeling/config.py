"""
modeling/config.py

Central configuration manager coordinating experiment metadata, directory
management, logging, dataset placeholders, preprocessing references, and
runtime state for model training workflows.

Extended Description
--------------------
This module defines a `TrainerConfig` class that acts as the shared state holder
for the entire training pipeline. It supports:
- storing core experiment parameters (target column, test size, random seed)
- managing output directories for artifacts and logs
- initializing a unified logger shared across all trainer components
- holding dataset splits (X_train, X_test, y_train, y_test)
- storing the preprocessing pipeline used to transform features
- maintaining a registry of trained models and evaluation results
- integrating a `Preprocessor` helper for feature engineering and pipeline construction

Trainer components rely on this module to maintain consistent state and
reproducible experiment structure.

Main Components
---------------
- TrainerConfig: end-to-end configuration container with:
  - logger construction with file + console handlers
  - storage for dataset splits
  - storage for preprocessing pipeline
  - storage for fitted models and evaluation metrics
  - shared Preprocessor instance for building feature pipelines

Usage Example
-------------
>>> from modeling.config import TrainerConfig
>>> cfg = TrainerConfig(target_col="SalePrice", test_size=0.2)
>>> cfg.logger.info("Trainer configuration initialized.")
>>> # Later in the pipeline:
>>> # cfg.df_ = pd.read_csv("train.csv")
>>> # cfg.feature_pipe_ = cfg.dp.build_feature_pipeline(df_train=cfg.df_)
>>> # cfg.models_["xgboost"] = trained_model

Notes
-----
The TrainerConfig is intended as the central state carrier for the training
pipeline. It does not load data or train models by itself; instead, it provides
the structure and storage that other trainer mixins build upon.
"""

from pathlib import Path
from typing import Optional
import logging

import pandas as pd

from preprocessing import Preprocessor


class TrainerConfig:
    """
    Central configuration manager for the model training workflow.

    Extended Description
    --------------------
    The `TrainerConfig` class acts as the shared runtime state for the entire
    training pipeline. It stores experiment settings, manages output
    directories, initializes logging, and provides placeholders for dataset
    splits, preprocessing pipelines, trained models, and evaluation results.

    Trainer modules and mixins rely on this object to access:
    - target column metadata
    - train test split settings
    - random seed configuration
    - preprocessing pipeline reference
    - output directory for artifacts
    - unified logger
    - runtime storage for fitted models and their metrics

    Parameters
    ----------
    target_col : str, optional
        Name of the target column to predict.
    test_size : float, default 0.2
        Fraction of the dataset allocated to the test split.
    random_state : int, default 42
        Random seed used across the training pipeline.
    output_dir : str, default "model_outputs"
        Root directory where logs, artifacts, and files are generated.
    log_level : int, default logging.INFO
        Logging verbosity level.

    Attributes
    ----------
    output_dir : pathlib.Path
        Directory containing all training outputs.
    logger : logging.Logger
        Unified logger writing both to console and to `training.log`.
    df_ : DataFrame or None
        Raw dataset loaded by the trainer.
    X_train_ : DataFrame or None
        Training feature matrix.
    X_test_ : DataFrame or None
        Test feature matrix.
    y_train_ : Series or None
        Target vector for training.
    y_test_ : Series or None
        Target vector for testing.
    feature_pipe_ : sklearn.Pipeline or None
        Preprocessing pipeline for transforming feature matrices.
    models_ : dict
        Dictionary mapping model names to fitted model instances.
    results_ : dict
        Evaluation metrics keyed by model name.
    dp : Preprocessor
        Preprocessor helper bound to the configured target column.

    Examples
    --------
    >>> cfg = TrainerConfig(target_col="SalePrice", test_size=0.2)
    >>> cfg.logger.info("Configuration initialized.")
    >>> # Later in a training pipeline:
    >>> # cfg.df_ = pd.read_csv("train.csv")
    >>> # cfg.feature_pipe_ = cfg.dp.build_feature_pipeline(df_train=cfg.df_)
    >>> # cfg.X_train_, cfg.X_test_, cfg.y_train_, cfg.y_test_ = ...
    """

    def __init__(
        self,
        target_col: str = "SalePrice",
        test_size: float = 0.2,
        random_state: int = 42,
        output_dir: str = "model_outputs",
        log_level: int = logging.INFO,
    ) -> None:
        self.target_col = target_col
        self.test_size = float(test_size)
        self.random_state = int(random_state)

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.logger = self._build_logger(log_level=log_level)

        # Data containers
        self.df_: Optional[pd.DataFrame] = None
        self.X_train_: Optional[pd.DataFrame] = None
        self.X_test_: Optional[pd.DataFrame] = None
        self.y_train_: Optional[pd.Series] = None
        self.y_test_: Optional[pd.Series] = None

        # Preprocessing pipeline
        self.feature_pipe_ = None

        # Model registry và kết quả
        self.models_ = {}
        self.results_ = {}

        # Data preprocessor helper
        self.dp = Preprocessor(target_col=self.target_col)

    def _build_logger(self, log_level: int = logging.INFO) -> logging.Logger:
        """
        Tạo logger ghi ra file training.log và in ra console.
        """
        logger = logging.getLogger("modeling.ModelTrainer")
        logger.setLevel(log_level)

        if logger.handlers:
            # Đã được tạo trước đó
            return logger

        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # File handler
        log_path = self.output_dir / "training.log"
        file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Console handler
        console = logging.StreamHandler()
        console.setLevel(log_level)
        console.setFormatter(formatter)
        logger.addHandler(console)

        return logger
