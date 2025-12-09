# src/modeling/data_mixin.py

from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


class DataMixin:
    """
    Chịu trách nhiệm về:
        - load_data
        - split_data

    Yêu cầu TrainerConfig đã định nghĩa:
        self.dp, self.logger
        self.df_, self.X_train_, self.X_test_, self.y_train_, self.y_test_
        self.test_size, self.random_state
    """

    def load_data(self, csv_path: str) -> None:
        """
        Load dữ liệu từ CSV và lưu vào self.df_ thông qua Preprocessor.
        """
        self.logger.info(f"Loading data from {csv_path}")
        self.df_ = self.dp.load_data(csv_path)
        self.logger.info(f"Data loaded with shape {self.df_.shape}")

    def split_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split dataframe thành train và test.

        Với regression: thực hiện stratified split bằng cách binning y
        thành quantile bins để giữ phân phối mục tiêu giữa train và test.
        """
        if self.df_ is None:
            raise RuntimeError("No data loaded. Call load_data first.")

        # Tách features và target
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
