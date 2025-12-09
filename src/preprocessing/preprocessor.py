"""
preprocessing/preprocessor.py

High-level data preprocessing manager wrapping loading, splitting, pipeline construction,
feature engineering, and sklearn transformations.

Extended Description
--------------------
This module defines a `Preprocessor` dataclass that centralizes all preprocessing
operations for structured tabular ML tasks. It supports:
- reading datasets (CSV / Excel / JSON)
- splitting features and target
- building complex sklearn Pipelines with ordinal mapping, domain features,
  target encoding, variance filters, and mutual information selectors
- producing processed feature matrices for train/validation/test
- logging preprocessing steps for debugging and reproducibility.

Main Components
---------------
- Preprocessor: end-to-end preprocessing controller with:
  - load_data: load raw dataset
  - split_features_target: separate X/y
  - build_feature_pipeline: assemble the sklearn pipeline
  - fit_transform: fit + transform training data
  - transform_new: transform unseen data
  - save_log, get_log_df: access and persist logs
  - get_feature_names_out: retrieve final feature names

Usage Example
-------------
>>> from preprocessing.preprocessor import Preprocessor
>>> pp = Preprocessor(target_col="SalePrice")
>>> df = pp.load_data("train.csv")
>>> X, y = pp.split_features_target(df)
>>> X_train = pp.fit_transform(X, y)
>>> X_test = pp.transform_new(X_test_raw)

Notes
-----
The Preprocessor is intended as the main user-facing interface. Internally,
it relies on build_feature_pipeline() from pipeline.py, which integrates all
transformers and selection steps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline

from .config import ORDINAL_MAP_CANONICAL
from .pipeline import build_feature_pipeline

@dataclass
class Preprocessor:
    """
    High-level preprocessing manager for structured ML datasets.

    Extended Description
    --------------------
    The `Preprocessor` class unifies multiple steps of the preprocessing workflow:
    - File loading (CSV, Excel, JSON)
    - Column dropping (ID fields)
    - Feature/target splitting
    - Construction of a full sklearn Pipeline
    - Fitting/transforming training data
    - Transforming new/unseen data
    - Collecting logs for inspection or debugging
    - Retrieving names of generated features

    It offers extensive configurability for:
    - ordinal mappings
    - domain feature engineering
    - target encoding
    - variance-based feature selection
    - mutual-information-based K-best feature selection

    Parameters
    ----------
    target_col : str, optional
        Name of the target column to predict.
    ordinal_mapping : dict, optional
        Mapping for ordinal feature conversion (canonical or numeric dict).
    id_cols : list of str, optional
        Identifier columns to drop before modeling.
    use_domain_features : bool, default True
        Whether to generate domain-engineered features.
    use_target_encoding : bool, default False
        Whether to apply target encoding.
    target_enc_cols : list of str, optional
        Columns on which to apply target encoding.
    enable_variance_selector : bool, default True
        Whether to enable variance thresholding.
    variance_threshold : float
        Minimum variance required to retain a feature.
    enable_kbest_mi : bool, default False
        Whether to apply MI-based K-best selection.
    k_best_features : int
        Number of top MI-ranked features to keep.
    mi_random_state : int
        Random seed for MI scoring.

    Attributes
    ----------
    df_raw_ : DataFrame or None
        Raw dataframe loaded from disk.
    feature_pipe_ : sklearn.Pipeline or None
        Constructed preprocessing pipeline.
    logs : list of str
        Ordered human-readable log entries.

    Examples
    --------
    >>> pp = Preprocessor(target_col="SalePrice", use_domain_features=True)
    >>> df = pp.load_data("train.csv")
    >>> X, y = pp.split_features_target(df)
    >>> X_train = pp.fit_transform(X, y)
    """

    target_col: Optional[str] = None
    ordinal_mapping: Optional[Mapping[str, Any]] = None
    id_cols: Optional[List[str]] = None
    use_domain_features: bool = True
    use_target_encoding: bool = False
    target_enc_cols: Optional[List[str]] = None

    enable_variance_selector: bool = True
    variance_threshold: float = 0.0
    enable_kbest_mi: bool = False
    k_best_features: int = 150
    mi_random_state: int = 0

    df_raw_: Optional[pd.DataFrame] = None
    feature_pipe_: Optional[Pipeline] = None
    logs: List[str] = field(default_factory=list)


    @staticmethod
    def _detect_file_type(path: str) -> str:
        path = path.lower()
        if path.endswith(".csv"):
            return "csv"
        if path.endswith(".xlsx") or path.endswith(".xls"):
            return "excel"
        if path.endswith(".json"):
            return "json"
        raise ValueError("Unable to detect file type. Supported: .csv, .xlsx/.xls, .json")

    def __repr__(self) -> str:
        ord_cols = (
            list(self.ordinal_mapping.keys()) if self.ordinal_mapping else []
        )
        return (
            f"Preprocessor(target_col={self.target_col!r}, "
            f"id_cols={self.id_cols}, "
            f"ordinal_cols={ord_cols}, "
            f"use_domain_features={self.use_domain_features}, "
            f"use_target_encoding={self.use_target_encoding})"
        )

    def _log(self, msg: str) -> None:
        self.logs.append(msg)


    def load_data(self, path: str, **read_kwargs) -> pd.DataFrame:
        ftype = self._detect_file_type(path)
        try:
            if ftype == "csv":
                df = pd.read_csv(path, **read_kwargs)
            elif ftype == "excel":
                df = pd.read_excel(path, **read_kwargs)
            else:  # json
                df = pd.read_json(path, **read_kwargs)
        except Exception as e:
            err_msg = f"[load_data] Error reading file {path}: {e}"
            self._log(err_msg)
            raise IOError(f"Cannot read file {path}: {e}") from e

        self.df_raw_ = df
        self._log(f"[load_data] Loaded {path} with shape={df.shape}")
        return df


    def _drop_id_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.id_cols:
            return df
        cols_to_drop = [c for c in self.id_cols if c in df.columns]
        return df.drop(columns=cols_to_drop, errors="ignore")

    def split_features_target(
        self, df: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        if df is None:
            if self.df_raw_ is None:
                raise RuntimeError("load_data has not been called and no df was provided.")
            df = self.df_raw_

        df = df.copy()
        df = self._drop_id_columns(df)

        if self.target_col is None:
            return df, None

        if self.target_col not in df.columns:
            raise ValueError(f"Target column '{self.target_col}' not found in DataFrame.")

        X = df.drop(columns=[self.target_col])
        y = df[self.target_col]
        
        if self.target_col is None:
            self._log(f"[split_features_target] No target_col provided; returning X only with shape={df.shape}")
            return df, None
        
        self._log(
            f"[split_features_target] X shape={X.shape}, y len={len(y)}, target={self.target_col}"
        )
        return X, y



    def build_feature_pipeline(self, X_train: pd.DataFrame) -> Pipeline:
        self.feature_pipe_ = build_feature_pipeline(
            df_train=X_train,
            ordinal_mapping=self.ordinal_mapping or ORDINAL_MAP_CANONICAL,
            use_domain_features=self.use_domain_features,
            use_target_encoding=self.use_target_encoding,
            target_enc_cols=self.target_enc_cols,
            enable_variance_selector=self.enable_variance_selector,
            variance_threshold=self.variance_threshold,
            enable_kbest_mi=self.enable_kbest_mi,
            k_best_features=self.k_best_features,
            mi_random_state=self.mi_random_state,
        )

        self._log(
            "[build_feature_pipeline] Pipeline created with "
            f"use_domain_features={self.use_domain_features}, "
            f"use_target_encoding={self.use_target_encoding}, "
            f"enable_variance_selector={self.enable_variance_selector}, "
            f"variance_threshold={self.variance_threshold}, "
            f"enable_kbest_mi={self.enable_kbest_mi}, "
            f"k_best_features={self.k_best_features}, "
            f"mi_random_state={self.mi_random_state}"
        )
        return self.feature_pipe_

    def fit_transform(
        self, X_train: pd.DataFrame, y_train: Optional[pd.Series] = None
    ) -> np.ndarray:
        if self.feature_pipe_ is None:
            self.build_feature_pipeline(X_train)
        X_proc = self.feature_pipe_.fit_transform(X_train, y_train)
        self._log(
            f"[fit_transform] Fitted pipeline on X_train shape={X_train.shape}, "
            f"y_train length={len(y_train) if y_train is not None else 'None'}, "
            f"output shape={X_proc.shape}"
        )
        return X_proc

    def transform_new(self, X_new: pd.DataFrame) -> np.ndarray:
        if self.feature_pipe_ is None:
            msg = "Pipeline has not been fitted. Call fit_transform before transform_new."
            self._log(f"[transform_new] ERROR: {msg}")
            raise RuntimeError(msg)
        X_proc = self.feature_pipe_.transform(X_new)
        self._log(f"[transform_new] Transformed X_new shape={X_new.shape} -> shape={X_proc.shape}")
        return X_proc
    
    def save_log(self, path: str = "preprocessing.log") -> None:
        with open(path, "w", encoding="utf-8") as f:
            for line in self.logs:
                f.write(line + "\n")

    def get_log_df(self) -> pd.DataFrame:
        return pd.DataFrame({"step": range(len(self.logs)), "message": self.logs})
    
    def get_feature_names_out(self) -> np.ndarray:
        if self.feature_pipe_ is None:
            raise RuntimeError(
                "feature_pipe_ has not been created. Call build_feature_pipeline or fit_transform first."
            )
        
        try:
            return self.feature_pipe_.get_feature_names_out()
        except AttributeError as e:
            raise NotFittedError(
                "Pipeline does not fully support get_feature_names_out "
                "(some steps may not implement it)."
            ) from e