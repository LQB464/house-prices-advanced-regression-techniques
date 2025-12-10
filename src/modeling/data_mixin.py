"""
modeling/data_mixin.py

Lightweight data-loading and splitting utilities used by the training pipeline,
providing consistent integration with the Preprocessor and TrainerConfig.

Extended Description
--------------------
This module defines the `DataMixin`, which encapsulates data ingestion and
train test splitting logic. It supports:
- loading datasets through the Preprocessor component
- separating features and target via Preprocessor utilities
- performing stratified train test split for regression tasks using target
  quantile binning
- updating shared TrainerConfig state with the resulting dataset partitions

The mixin assumes it is combined with a TrainerConfig instance that already
defines:
- self.dp (Preprocessor helper)
- self.logger (unified logger)
- self.df_, self.X_train_, self.X_test_, self.y_train_, self.y_test_
- self.test_size, self.random_state

Main Components
---------------
- DataMixin:
  - load_data: load raw dataset through Preprocessor
  - split_data: perform stratified or fallback random splitting

Usage Example
-------------
>>> from modeling.data_mixin import DataMixin
>>> class Trainer(TrainerConfig, DataMixin):
...     pass
>>> trainer = Trainer()
>>> trainer.load_data("train.csv")
>>> X_train, X_test, y_train, y_test = trainer.split_data()

Notes
-----
This module does not train models. It only manages data ingestion and
partitioning. Stratified regression splitting is implemented via quantile
binning, but gracefully falls back to random splitting when binning fails.
"""

from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


class DataMixin:
    """
    Data ingestion and splitting helper mixed into the training pipeline.

    Extended Description
    --------------------
    The `DataMixin` provides two core responsibilities:
    - loading raw datasets through the Preprocessor (`self.dp`)
    - splitting features and targets into train and test partitions using
      stratified quantile binning when possible

    It relies on attributes defined in TrainerConfig, including:
    `self.dp`, `self.logger`, `self.df_`, train test containers,
    and split hyperparameters (`self.test_size`, `self.random_state`).

    Parameters
    ----------
    None
        This mixin expects configuration attributes from TrainerConfig and
        does not define its own constructor.

    Attributes
    ----------
    df_ : DataFrame or None
        Raw dataset loaded from disk.
    X_train_, X_test_ : DataFrame or None
        Train and test feature matrices.
    y_train_, y_test_ : Series or None
        Train and test target vectors.

    Examples
    --------
    >>> class Trainer(TrainerConfig, DataMixin):
    ...     pass
    >>> trainer = Trainer()
    >>> trainer.load_data("train.csv")
    >>> trainer.split_data()
    """

    def load_data(self, csv_path: str) -> None:
        self.logger.info(f"Loading data from {csv_path}")
        self.df_ = self.dp.load_data(csv_path)
        self.logger.info(f"Data loaded with shape {self.df_.shape}")

    def split_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        if self.df_ is None:
            raise RuntimeError("No data loaded. Call load_data first.")

        # Split features and target
        X, y = self.dp.split_features_target(self.df_)

        # Stratified binning
        n_bins = 15
        try:
            y_binned = pd.qcut(y, q=n_bins, duplicates="drop", labels=False)
            stratify_label = y_binned
            self.logger.info(
                f"Using stratified split with {n_bins} quantile bins for target."
            )
        except Exception as e:
            self.logger.warning(
                "Could not build quantile bins for stratified split. "
                f"Falling back to random split. Error: {e}"
            )
            stratify_label = None

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify_label,
        )

        self.X_train_, self.X_test_ = X_train, X_test
        self.y_train_, self.y_test_ = y_train, y_test

        self.logger.info(
            f"Train shape: {X_train.shape}, Test shape: {X_test.shape}"
        )

        return X_train, X_test, y_train, y_test
