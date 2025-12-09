"""
preprocessing/transformers.py

Collection of custom sklearn-compatible preprocessing transformers.

Extended Description
--------------------
This module implements reusable transformer classes for preparing structured/tabular
data. These transformers extend sklearn’s BaseEstimator and TransformerMixin, covering:
- ordinal mapping
- missingness indicators
- rare-category grouping
- outlier clipping via IQR
- finite-value cleaning
- dropping fully-NaN columns
- target encoding
- variance-based feature selection
- mutual information–based K-best selection

All transformers implement get_feature_names_out where applicable so they integrate
properly with ColumnTransformer and advanced pipelines.

Main Components
---------------
- OrdinalMapper: Map ordered categories to numeric values.
- MissingnessIndicator: Add `<col>_was_missing` flags for numeric columns.
- RareCategoryGrouper: Merge low-frequency categories into "Other".
- OutlierClipper: Apply IQR clipping to numeric features.
- FiniteCleaner: Replace inf/-inf with NaN.
- DropAllNaNColumns: Remove columns that are fully NaN.
- TargetEncoderTransformer: Simple target encoding for selected categorical cols.
- VarianceFeatureSelector: Keep columns with variance above a threshold.
- KBestMutualInfoSelector: Select top-k features based on mutual information.

Usage Example
-------------
>>> from preprocessing.transformers import OrdinalMapper, OutlierClipper
>>> mapper = OrdinalMapper(mapping={"Qual": ["Poor", "Fair", "Good"]})
>>> df2 = mapper.fit_transform(df)
>>> clipper = OutlierClipper(factor=1.5)
>>> df3 = clipper.fit_transform(df2)

Notes
-----
All transformers are designed to be pipeline-safe and able to handle
unexpected column types gracefully. Some transformers require y during fit
(e.g., TargetEncoderTransformer, KBestMutualInfoSelector).
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from functools import partial

class OrdinalMapper(BaseEstimator, TransformerMixin):
    """
    Transformer that converts ordinal categorical columns to numeric values.

    Extended Description
    --------------------
    This transformer maps user-specified ordinal levels to numeric scores.
    It accepts either:
    - a dict of ordered lists (canonical form), or
    - a dict mapping category → numeric value.

    Any categories not present in the mapping become NaN, allowing them to be
    handled later by imputation.

    Parameters
    ----------
    mapping : dict, optional
        Mapping specification for each ordinal column.

    Attributes
    ----------
    mapping_ : dict
        Processed numeric mapping for each column.
    cols_ : list of str
        Columns actually present in the input during fit.
    feature_names_in_ : ndarray
        Original input column names.

    Examples
    --------
    >>> mp = {"Qual": ["Poor", "Fair", "Good", "Excellent"]}
    >>> mapper = OrdinalMapper(mp)
    >>> df2 = mapper.fit_transform(df)
    """

    def __init__(self, mapping: Optional[Mapping[str, Any]] = None):
        self.mapping = mapping

        self.mapping_raw = mapping or {}

        self.mapping_: Dict[str, Dict[str, float]] = {}
        self.cols_: List[str] = []

    @staticmethod
    def _canon_to_numeric_map(mapping: Mapping[str, Any]) -> Dict[str, Dict[str, float]]:
        final: Dict[str, Dict[str, float]] = {}
        for col, spec in mapping.items():
            if isinstance(spec, dict):
                final[col] = {k: float(v) for k, v in spec.items()}
            else:
                levels: Iterable[Any] = spec
                final[col] = {v: float(i) for i, v in enumerate(levels, start=1)}
        return final

    def fit(self, X: pd.DataFrame, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.feature_names_in_ = np.asarray(X.columns, dtype=object)

        numeric_map = self._canon_to_numeric_map(self.mapping_raw)
        self.mapping_ = {
            col: mp for col, mp in numeric_map.items() if col in X.columns
        }
        self.cols_ = list(self.mapping_.keys())
        return self

    def transform(self, X: pd.DataFrame):
        if not hasattr(self, "mapping_"):
            raise NotFittedError("OrdinalMapper is not fitted yet.")
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        X = X.copy()
        for col in self.cols_:
            mp = self.mapping_[col]
            # Map to float, unknown values become NaN
            X[col] = X[col].map(mp).astype(float)
        return X

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = getattr(self, "feature_names_in_", None)
        if input_features is None:
            raise NotFittedError("OrdinalMapper has not been fitted yet.")
        return np.asarray(input_features, dtype=object)


class MissingnessIndicator(BaseEstimator, TransformerMixin):
    """
    Add binary '<col>_was_missing' flags for numeric columns with missing values.

    Extended Description
    --------------------
    This transformer inspects numeric columns during fit() and identifies which
    columns contain NaN values. During transform(), it appends a binary indicator
    column for each such feature, marking where missing values occurred.

    Attributes
    ----------
    num_cols_with_nan_ : list of str
        Numeric columns containing missing values.
    """

    def __init__(self):
        self.num_cols_with_nan_: List[str] = []

    def fit(self, X: pd.DataFrame, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        num_cols = X.select_dtypes(include=[np.number]).columns
        self.num_cols_with_nan_ = [c for c in num_cols if X[c].isna().any()]
        return self

    def transform(self, X: pd.DataFrame):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        X = X.copy()
        for c in self.num_cols_with_nan_:
            X[f"{c}_was_missing"] = X[c].isna().astype(int)
        return X

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = getattr(self, "feature_names_in_", None)
        if input_features is None:
            raise NotFittedError("MissingnessIndicator has not been fitted.")

        base = list(input_features)
        extra = [f"{c}_was_missing" for c in self.num_cols_with_nan_]
        return np.asarray(base + extra, dtype=object)


class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    """
    Group rare categories in categorical columns under the label 'Other'.

    Parameters
    ----------
    min_freq : int, default=20
        Minimum frequency required to retain a category.

    Attributes
    ----------
    category_maps_ : dict
        Mapping of column → kept categories.
    """

    def __init__(self, min_freq: int = 20):
        self.min_freq = int(min_freq)
        self.category_maps_: Dict[str, List[str]] = {}

    def fit(self, X: pd.DataFrame, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        cat_cols = X.select_dtypes(include=["object", "category"]).columns
        for c in cat_cols:
            vc = X[c].value_counts(dropna=False)
            keep = vc[vc >= self.min_freq].index.astype(str).tolist()
            self.category_maps_[c] = keep
        return self

    def transform(self, X: pd.DataFrame):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        X = X.copy()
        for c, keep in self.category_maps_.items():
            if c not in X.columns:
                continue
            X[c] = X[c].astype(str)
            X[c] = np.where(X[c].isin(keep), X[c], "Other")
        return X

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = getattr(self, "feature_names_in_", None)
        if input_features is None:
            raise NotFittedError("RareCategoryGrouper has not been fitted.")
        return np.asarray(input_features, dtype=object)


class OutlierClipper(BaseEstimator, TransformerMixin):
    """
    Clip outliers in numeric columns using the IQR rule.

    Extended Description
    --------------------
    For each numeric column:
    - Compute Q1, Q3, and IQR.
    - Clip values outside [Q1 - factor * IQR, Q3 + factor * IQR].

    Parameters
    ----------
    factor : float, default=1.5
        Scaling factor for IQR clipping.

    Attributes
    ----------
    bounds_ : dict
        Per-column clipping bounds.
    """

    def __init__(self, factor: float = 1.5):
        self.factor = float(factor)
        self.bounds_: Dict[str, Tuple[float, float]] = {}

    def fit(self, X: pd.DataFrame, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        num_cols = X.select_dtypes(include=[np.number]).columns
        for c in num_cols:
            q1 = X[c].quantile(0.25)
            q3 = X[c].quantile(0.75)
            iqr = q3 - q1
            if pd.isna(iqr) or iqr == 0:
                continue
            lower = q1 - self.factor * iqr
            upper = q3 + self.factor * iqr
            self.bounds_[c] = (lower, upper)
        return self

    def transform(self, X: pd.DataFrame):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        X = X.copy()
        for c, (lower, upper) in self.bounds_.items():
            if c not in X.columns:
                continue
            X[c] = X[c].clip(lower=lower, upper=upper)
        return X

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = getattr(self, "feature_names_in_", None)
        if input_features is None:
            raise NotFittedError("OutlierClipper has not been fitted.")
        return np.asarray(input_features, dtype=object)


class FiniteCleaner(BaseEstimator, TransformerMixin):
    """
    Replace inf and -inf values with NaN to prepare for imputation.

    Notes
    -----
    This transformer has no learned parameters.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        return X.replace([np.inf, -np.inf], np.nan)
    
    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = getattr(self, "feature_names_in_", None)
        if input_features is None:
            raise NotFittedError("FiniteCleaner has not been fitted.")
        return np.asarray(input_features, dtype=object)


class DropAllNaNColumns(BaseEstimator, TransformerMixin):
    """
    Drop columns containing only NaN values.

    Attributes
    ----------
    keep_cols_ : list of int
        Column indices retained after fit().
    """

    def __init__(self):
        self.keep_cols_: Optional[List[int]] = None

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        self.keep_cols_ = [
            i for i, c in enumerate(X_df.columns) if not X_df[c].isna().all()
        ]
        return self

    def transform(self, X):
        if self.keep_cols_ is None:
            raise NotFittedError("DropAllNaNColumns has not been fitted.") 
        X_df = pd.DataFrame(X)
        return X_df.iloc[:, self.keep_cols_].values
    
    def get_feature_names_out(self, input_features=None):
        if self.keep_cols_ is None:
            raise NotFittedError("DropAllNaNColumns has not been fitted.") 

        if input_features is None:
            input_features = self.feature_names_in_
        if input_features is None:
            raise NotFittedError("No input_features information available in DropAllNaNColumns.")

        input_features = np.asarray(input_features, dtype=object)
        return input_features[self.keep_cols_]


class TargetEncoderTransformer(BaseEstimator, TransformerMixin):
    """
    Simple target encoding for selected categorical columns.

    Extended Description
    --------------------
    Computes mean(target) for each category and replaces categories with their
    encoded numeric mean. Keeps the original column and adds a new 'TE_<col>' column.

    Parameters
    ----------
    cols : list of str
        Columns to apply target encoding.

    Attributes
    ----------
    mapping_ : dict
        Per-column category → mean(target) mappings.
    global_mean_ : float
        Used when unseen categories arise during transform.
    """

    def __init__(self, cols: Optional[List[str]] = None):
        self.cols = cols or []
        self.global_mean_: float = 0.0
        self.mapping_: Dict[str, Dict[Any, float]] = {}

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        if y is None:
            raise ValueError("TargetEncoderTransformer requires y to fit.")
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        y = pd.Series(y)
        self.global_mean_ = float(y.mean())

        for col in self.cols:
            if col not in X.columns:
                continue
            df = pd.DataFrame({"col": X[col], "y": y})
            mapping = df.groupby("col")["y"].mean().to_dict()
            self.mapping_[col] = mapping

        return self

    def transform(self, X: pd.DataFrame):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        X = X.copy()
        for col, mapping in self.mapping_.items():
            if col not in X.columns:
                continue
            te_col = X[col].map(mapping)
            te_col = te_col.fillna(self.global_mean_)
            X[f"TE_{col}"] = te_col.astype(float)

        return X
    
    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = getattr(self, "feature_names_in_", None)
        if input_features is None:
            raise NotFittedError("TargetEncoderTransformer has not been fitted.")

        base = list(input_features)
        extra = [f"TE_{col}" for col in self.mapping_.keys()]
        return np.asarray(base + extra, dtype=object)


class VarianceFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Select features whose variance exceeds a given threshold.

    Parameters
    ----------
    threshold : float
        Minimum variance required to retain a feature.

    Attributes
    ----------
    keep_indices_ : ndarray
        Indices of selected features.
    """

    def __init__(self, threshold: float = 0.0):
        self.threshold = float(threshold)
        self.keep_indices_: Optional[np.ndarray] = None

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        var = X_df.var(axis=0)
        self.keep_indices_ = np.where(var > self.threshold)[0]
        return self

    def transform(self, X):
        if self.keep_indices_ is None:
            raise NotFittedError("VarianceFeatureSelector has not been fitted yet.")
        X_df = pd.DataFrame(X)
        return X_df.iloc[:, self.keep_indices_].values

    def get_feature_names_out(self, input_features=None):
        if self.keep_indices_ is None:
            raise NotFittedError("VarianceFeatureSelector has not been fitted yet.")
        if input_features is None:
            input_features = getattr(self, "feature_names_in_", None)
        if input_features is None:
            raise NotFittedError(
                "VarianceFeatureSelector does not have input feature information."
            )
        input_features = np.asarray(input_features, dtype=object)
        return input_features[self.keep_indices_]


class KBestMutualInfoSelector(BaseEstimator, TransformerMixin):
    """
    Select top-k features based on mutual information with the target.

    Parameters
    ----------
    k : int, default=100
        Number of features to keep.
    random_state : int, default=0
        Random state for MI computation.

    Attributes
    ----------
    selector_ : SelectKBest
        Wrapped sklearn selector.
    feature_names_in_ : array-like
        Original feature names.
    """

    def __init__(self, k: int = 100, random_state: int = 0):
        self.k = int(k)
        self.random_state = int(random_state)
        self.selector_: Optional[SelectKBest] = None
        self.feature_names_in_ = None

    def fit(self, X, y):
        if y is None:
            raise ValueError("KBestMutualInfoSelector requires y to fit.")

        X_arr = np.asarray(X)
        y_arr = np.asarray(y)

        n_features = X_arr.shape[1]
        k = min(self.k, n_features)

        score_func = partial(
            mutual_info_regression,
            random_state=self.random_state,
        )

        self.selector_ = SelectKBest(score_func=score_func, k=k)

        self.feature_names_in_ = getattr(X, "columns", None)

        self.selector_.fit(X_arr, y_arr)
        return self


    def transform(self, X):
        if self.selector_ is None:
            raise NotFittedError("KBestMutualInfoSelector has not been fitted.")
        return self.selector_.transform(np.asarray(X))

    def get_feature_names_out(self, input_features=None):
        if self.selector_ is None:
            raise NotFittedError("KBestMutualInfoSelector has not been fitted.")

        if input_features is None:
            input_features = self.feature_names_in_

        if input_features is None:
            raise NotFittedError("input_features not found.")

        input_features = np.asarray(input_features, dtype=object)
        mask = self.selector_.get_support()
        return input_features[mask]


__all__ = [
    "OrdinalMapper",
    "MissingnessIndicator",
    "RareCategoryGrouper",
    "OutlierClipper",
    "FiniteCleaner",
    "DropAllNaNColumns",
    "TargetEncoderTransformer",
    "VarianceFeatureSelector",
    "KBestMutualInfoSelector",
]