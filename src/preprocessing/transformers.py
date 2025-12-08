# preprocessing/transformers.py
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
    Map ordinal columns to numeric values using a mapping dict.

    mapping can be:
        - Dict[str, List[str]]  list of ordered levels
        - Dict[str, Dict[str, int or float]]
    """

    def __init__(self, mapping: Optional[Mapping[str, Any]] = None):
        # Hyperparameter that sklearn expects in get_params / set_params
        self.mapping = mapping

        # Internal raw mapping storage
        self.mapping_raw = mapping or {}

        # Fitted attributes
        self.mapping_: Dict[str, Dict[str, float]] = {}
        self.cols_: List[str] = []

    @staticmethod
    def _canon_to_numeric_map(mapping: Mapping[str, Any]) -> Dict[str, Dict[str, float]]:
        """
        Convert:
            - Dict[str, List[str]]  canonical order
        or
            - Dict[str, Dict[str, int]]
        into:
            Dict[str, Dict[str, float]] used by OrdinalMapper.
        """
        final: Dict[str, Dict[str, float]] = {}
        for col, spec in mapping.items():
            if isinstance(spec, dict):
                final[col] = {k: float(v) for k, v in spec.items()}
            else:
                # Assume iterable of ordered levels
                levels: Iterable[Any] = spec
                final[col] = {v: float(i) for i, v in enumerate(levels, start=1)}
        return final

    def fit(self, X: pd.DataFrame, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # Save feature names for get_feature_names_out
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)

        # Normalize mapping into numeric dict of dict
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
    Tạo các cột `<col>_was_missing` = 1 nếu giá trị ban đầu NaN.
    Chỉ áp dụng cho các cột numeric.
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
            raise NotFittedError("MissingnessIndicator chưa được fit.")

        base = list(input_features)
        extra = [f"{c}_was_missing" for c in self.num_cols_with_nan_]
        return np.asarray(base + extra, dtype=object)


class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    """
    Gộp các nhóm hiếm trong biến phân loại thành nhãn 'Other'.

    Parameters
    ----------
    min_freq : int, default=20
        Số lượng xuất hiện tối thiểu để một category được giữ riêng.
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
            raise NotFittedError("RareCategoryGrouper chưa được fit.")
        return np.asarray(input_features, dtype=object)


class OutlierClipper(BaseEstimator, TransformerMixin):
    """
    Cắt (clip) outlier của các cột numeric bằng quy tắc IQR.

    Với mỗi cột:
        Q1, Q3 -> IQR = Q3 - Q1
        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR
        Giá trị < lower -> lower; > upper -> upper
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
            raise NotFittedError("OutlierClipper chưa được fit.")
        return np.asarray(input_features, dtype=object)


class FiniteCleaner(BaseEstimator, TransformerMixin):
    """
    Biến đổi inf, -inf thành NaN (để Imputer xử lý tiếp).
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
            raise NotFittedError("FiniteCleaner chưa được fit.")
        return np.asarray(input_features, dtype=object)


class DropAllNaNColumns(BaseEstimator, TransformerMixin):
    """
    Bỏ các cột toàn NaN sau khi tiền xử lý (phòng trường hợp
    sau OneHot hoặc các bước khác sinh ra cột rỗng).
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
            raise NotFittedError("DropAllNaNColumns chưa được fit.")
        X_df = pd.DataFrame(X)
        return X_df.iloc[:, self.keep_cols_].values
    
    def get_feature_names_out(self, input_features=None):
        if self.keep_cols_ is None:
            raise NotFittedError("DropAllNaNColumns chưa được fit.")

        if input_features is None:
            input_features = self.feature_names_in_
        if input_features is None:
            raise NotFittedError(
                "Không có thông tin input_features trong DropAllNaNColumns."
            )

        input_features = np.asarray(input_features, dtype=object)
        return input_features[self.keep_cols_]


class TargetEncoderTransformer(BaseEstimator, TransformerMixin):
    """
    Target Encoding đơn giản cho một số cột categorical.

    - Với mỗi cột trong `cols`, tính mean(target) theo từng category trên tập train.
    - Tạo cột mới `TE_<col>` là giá trị encoded.
    - Giữ lại cột gốc để vẫn có thể one-hot nếu muốn.
    """

    def __init__(self, cols: Optional[List[str]] = None):
        self.cols = cols or []
        self.global_mean_: float = 0.0
        self.mapping_: Dict[str, Dict[Any, float]] = {}

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        if y is None:
            raise ValueError("TargetEncoderTransformer cần y để fit.")
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
            raise NotFittedError("TargetEncoderTransformer chưa được fit.")

        base = list(input_features)
        extra = [f"TE_{col}" for col in self.mapping_.keys()]
        return np.asarray(base + extra, dtype=object)


class VarianceFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Loại bỏ các cột có phương sai quá nhỏ (gần như hằng số).

    Parameters
    ----------
    threshold : float, default=0.0
        Giữ lại các cột có var > threshold.
    """

    def __init__(self, threshold: float = 0.0):
        self.threshold = float(threshold)
        self.keep_indices_: Optional[np.ndarray] = None

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        # var theo cột
        var = X_df.var(axis=0)
        self.keep_indices_ = np.where(var > self.threshold)[0]
        return self

    def transform(self, X):
        if self.keep_indices_ is None:
            raise NotFittedError("VarianceFeatureSelector chưa được fit.")
        X_df = pd.DataFrame(X)
        return X_df.iloc[:, self.keep_indices_].values

    def get_feature_names_out(self, input_features=None):
        if self.keep_indices_ is None:
            raise NotFittedError("VarianceFeatureSelector chưa được fit.")
        if input_features is None:
            input_features = getattr(self, "feature_names_in_", None)
        if input_features is None:
            raise NotFittedError(
                "VarianceFeatureSelector không có thông tin input_features."
            )
        input_features = np.asarray(input_features, dtype=object)
        return input_features[self.keep_indices_]


class KBestMutualInfoSelector(BaseEstimator, TransformerMixin):
    """
    Chọn top k feature theo Mutual Information với target.
    """

    def __init__(self, k: int = 100, random_state: int = 0):
        self.k = int(k)
        self.random_state = int(random_state)
        self.selector_: Optional[SelectKBest] = None
        self.feature_names_in_ = None

    def fit(self, X, y):
        if y is None:
            raise ValueError("KBestMutualInfoSelector cần y để fit.")

        X_arr = np.asarray(X)
        y_arr = np.asarray(y)

        # chỉnh đúng k
        n_features = X_arr.shape[1]
        k = min(self.k, n_features)

        # dùng partial của mutual_info_regression thay vì local function
        score_func = partial(
            mutual_info_regression,
            random_state=self.random_state,
        )

        self.selector_ = SelectKBest(score_func=score_func, k=k)

        # lưu tên cột nếu có
        self.feature_names_in_ = getattr(X, "columns", None)

        self.selector_.fit(X_arr, y_arr)
        return self

    def transform(self, X):
        if self.selector_ is None:
            raise NotFittedError("KBestMutualInfoSelector chưa được fit.")
        return self.selector_.transform(np.asarray(X))

    def get_feature_names_out(self, input_features=None):
        if self.selector_ is None:
            raise NotFittedError("KBestMutualInfoSelector chưa được fit.")

        if input_features is None:
            input_features = self.feature_names_in_

        if input_features is None:
            raise NotFittedError("Không tìm thấy input_features!")

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