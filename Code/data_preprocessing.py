"""
data_preprocessing.py

Phan 1: Tien xu ly du lieu cho du an House Prices (Ames).
Chua cac class va ham chuyen ve tien xu ly, co the tai su dung cho cac tap du lieu
co cau truc tuong tu.

Su dung:
    from data_preprocessing import DataPreprocessor, build_feature_pipeline
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer
from sklearn.preprocessing import FunctionTransformer


# ============================================================
# 1. Mapping cho cac bien ordinal trong Ames
# ============================================================

ORDINAL_MAP_CANONICAL: Dict[str, Dict[str, int]] = {
    # chat luong chung
    "ExterQual": {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "ExterCond": {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "BsmtQual": {"NA": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "BsmtCond": {"NA": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "HeatingQC": {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "KitchenQual": {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "FireplaceQu": {"NA": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "GarageQual": {"NA": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "GarageCond": {"NA": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "PoolQC": {"NA": 0, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},

    # Fence, Functional, CentralAir, PavedDrive
    "Fence": {"NA": 0, "MnWw": 1, "GdWo": 2, "MnPrv": 3, "GdPrv": 4},
    "Functional": {
        "Sal": 1,
        "Sev": 2,
        "Maj2": 3,
        "Maj1": 4,
        "Mod": 5,
        "Min2": 6,
        "Min1": 7,
        "Typ": 8,
    },
    "CentralAir": {"N": 0, "Y": 1},
    "PavedDrive": {"N": 0, "P": 1, "Y": 2},
}


# ============================================================
# 2. Cac transformer co ban
# ============================================================

class OrdinalMapper(BaseEstimator, TransformerMixin):
    """
    Map cac cot ordinal theo ORDINAL_MAP_CANONICAL sang so.
    Neu gap gia tri khong nam trong mapping thi gan NaN.
    """

    def __init__(self, mapping: Dict[str, Dict[str, int]] = ORDINAL_MAP_CANONICAL):
        self.mapping = mapping
        self.cols_: List[str] = []

    def fit(self, X: pd.DataFrame, y=None):
        self.cols_ = [c for c in self.mapping.keys() if c in X.columns]
        return self

    def transform(self, X: pd.DataFrame):
        X = X.copy()
        for col in self.cols_:
            mp = self.mapping[col]
            X[col] = X[col].map(mp).astype(float)
        return X


class MissingnessIndicator(BaseEstimator, TransformerMixin):
    """
    Tao cac cot _was_missing = 1 neu gia tri ban dau NaN.
    Chi ap dung cho cac cot numeric.
    """

    def __init__(self):
        self.num_cols_with_nan_: List[str] = []

    def fit(self, X: pd.DataFrame, y=None):
        num_cols = X.select_dtypes(include=[np.number]).columns
        self.num_cols_with_nan_ = [
            c for c in num_cols if X[c].isna().any()
        ]
        return self

    def transform(self, X: pd.DataFrame):
        X = X.copy()
        for c in self.num_cols_with_nan_:
            X[f"{c}_was_missing"] = X[c].isna().astype(int)
        return X


class RarePooler(BaseEstimator, TransformerMixin):
    """
    Gop cac nhom hiem trong bien phan loai thanh nhan 'Other'.
    """

    def __init__(self, min_count: int = 20):
        self.min_count = min_count
        self.category_maps_: Dict[str, List[str]] = {}

    def fit(self, X: pd.DataFrame, y=None):
        cat_cols = X.select_dtypes(include=["object", "category"]).columns
        for c in cat_cols:
            vc = X[c].value_counts()
            keep = vc[vc >= self.min_count].index.tolist()
            self.category_maps_[c] = keep
        return self

    def transform(self, X: pd.DataFrame):
        X = X.copy()
        for c, keep in self.category_maps_.items():
            if c not in X.columns:
                continue
            X[c] = np.where(X[c].isin(keep), X[c], "Other")
        return X


class TargetEncoderTransformer(BaseEstimator, TransformerMixin):
    """
    Target encoding don gian cho mot so cot categorical.
    """

    def __init__(self, cols: Optional[List[str]] = None, alpha: float = 5.0):
        self.cols = cols
        self.alpha = alpha
        self.global_mean_: float = 0.0
        self.encodings_: Dict[str, Dict[str, float]] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series):
        if self.cols is None:
            self.cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

        self.global_mean_ = float(y.mean())
        for c in self.cols:
            if c not in X.columns:
                continue
            df = pd.DataFrame({"col": X[c], "y": y})
            stats = df.groupby("col")["y"].agg(["mean", "count"])
            smooth = (stats["count"] * stats["mean"] + self.alpha * self.global_mean_) / (
                stats["count"] + self.alpha
            )
            self.encodings_[c] = smooth.to_dict()
        return self

    def transform(self, X: pd.DataFrame):
        X = X.copy()
        for c, mp in self.encodings_.items():
            if c not in X.columns:
                continue
            X[c + "_te"] = X[c].map(mp).fillna(self.global_mean_)
        return X


class FiniteCleaner(BaseEstimator, TransformerMixin):
    """
    Bien doi inf, -inf thanh NaN.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        X = X.replace([np.inf, -np.inf], np.nan)
        return X


class DropAllNaNColumns(BaseEstimator, TransformerMixin):
    """
    Bo cac cot toan NaN sau khi tien xu ly.
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
        X_df = pd.DataFrame(X)
        return X_df.iloc[:, self.keep_cols_].values


# ============================================================
# 3. Ham tao domain features cho Ames
# ============================================================

def add_domain_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Them cac feature tu domain cho Ames.
    Ham nay chi lam viec o level DataFrame thoi.
    """
    X = X.copy()

    # tong dien tich su dung
    X["TotalSF"] = X.get("TotalBsmtSF", 0) + X.get("1stFlrSF", 0) + X.get("2ndFlrSF", 0)

    # so phong tam
    X["TotalBath"] = (
        X.get("FullBath", 0)
        + 0.5 * X.get("HalfBath", 0)
        + X.get("BsmtFullBath", 0)
        + 0.5 * X.get("BsmtHalfBath", 0)
    )

    # tuoi nha
    if "YearBuilt" in X.columns:
        X["HouseAge"] = X["YrSold"] - X["YearBuilt"]

    if "YearRemodAdd" in X.columns:
        X["RemodAge"] = X["YrSold"] - X["YearRemodAdd"]
        X["IsRemodeled"] = (X["YearRemodAdd"] != X["YearBuilt"]).astype(int)

    # porch
    X["TotalPorchSF"] = (
        X.get("OpenPorchSF", 0)
        + X.get("EnclosedPorch", 0)
        + X.get("3SsnPorch", 0)
        + X.get("ScreenPorch", 0)
    )

    # ti le dien tich lot
    if "LotArea" in X.columns and "GrLivArea" in X.columns:
        X["LotAreaRatio"] = X["GrLivArea"] / X["LotArea"].replace(0, np.nan)

    # ma hoa thang ban
    if "MoSold" in X.columns:
        X["MoSold_sin"] = np.sin(2 * np.pi * X["MoSold"] / 12.0)
        X["MoSold_cos"] = np.cos(2 * np.pi * X["MoSold"] / 12.0)

    # log TotalSF
    if "TotalSF" in X.columns:
        X["Ln_TotalSF"] = np.log1p(X["TotalSF"])

    return X


# ============================================================
# 4. Xay dung ColumnTransformer va Pipeline tien xu ly
# ============================================================

def build_feature_lists(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Tach danh sach cot theo loai:
    ord_cols, cat_cols, num_cont, num_absence
    """
    df = df.copy()

    # MSSubClass de dang object
    if "MSSubClass" in df.columns:
        df["MSSubClass"] = df["MSSubClass"].astype(str)

    ord_cols = [c for c in ORDINAL_MAP_CANONICAL.keys() if c in df.columns]
    cat_cols = [
        c
        for c in df.select_dtypes(include=["object", "category"]).columns
        if c not in ord_cols
    ]
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    num_absence = [c for c in num_cols if df[c].isna().any()]
    num_cont = [c for c in num_cols if c not in num_absence]

    return ord_cols, cat_cols, num_cont, num_absence


def make_preprocessor(df_train: pd.DataFrame, df_test: Optional[pd.DataFrame] = None) -> ColumnTransformer:
    """
    Tao ColumnTransformer cho du lieu da duoc add_domain_features.
    """
    if df_test is None:
        df_test = df_train

    df_all = pd.concat([df_train, df_test], axis=0)
    ord_cols, cat_cols, num_cont, num_absence = build_feature_lists(df_all)

    # categorical branch
    cat_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False)),
        ]
    )

    # ordinal branch
    ord_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
        ]
    )

    # numeric continuous branch
    num_cont_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("qt", QuantileTransformer(output_distribution="normal")),
        ]
    )

    # numeric with absence indicator, chi impute median
    num_abs_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
        ]
    )

    preproc = ColumnTransformer(
        transformers=[
            ("cats", cat_pipeline, cat_cols),
            ("ords", ord_pipeline, ord_cols),
            ("num_cont", num_cont_pipeline, num_cont),
            ("num_abs", num_abs_pipeline, num_absence),
        ],
        remainder="drop",
    )

    return preproc


def build_feature_pipeline(df_train: pd.DataFrame, df_test: Optional[pd.DataFrame] = None) -> Pipeline:
    """
    Tao full Pipeline:
    - them domain features
    - map ordinal
    - tao missingness indicator
    - pool rare category
    - target encode
    - ColumnTransformer
    - clean inf va drop cot toan NaN
    """
    if df_test is None:
        df_test = df_train

    pipe = Pipeline(
        steps=[
            ("domain", FunctionTransformer(add_domain_features, validate=False)),
            ("ordinal_map", OrdinalMapper()),
            ("missing_indicator", MissingnessIndicator()),
            ("rare_pool", RarePooler(min_count=20)),
            ("target_encode", TargetEncoderTransformer(
                cols=[
                    "Neighborhood",
                    "MSZoning",
                    "Exterior1st",
                    "Exterior2nd",
                    "SaleCondition",
                ],
                alpha=5.0,
            )),
            ("col_transform", make_preprocessor(df_train, df_test)),
            ("finite_clean", FiniteCleaner()),
            ("drop_all_nan", DropAllNaNColumns()),
        ]
    )

    return pipe


# ============================================================
# 5. Lop DataPreprocessor cap cao (dung trong phan Model)
# ============================================================

class DataPreprocessor:
    """
    Lop tien xu ly du lieu cap cao.

    Chuc nang:
    - load_data: doc csv
    - split_features_target: tach X, y
    - build_feature_pipeline: tao sklearn Pipeline
    - fit_transform_train: fit pipeline va tra ve X_train_processed
    - transform_new: bien doi tap du lieu moi
    """

    def __init__(self, target_col: str = "SalePrice"):
        self.target_col = target_col
        self.df_raw_: Optional[pd.DataFrame] = None
        self.feature_pipe_: Optional[Pipeline] = None

    def load_data(self, csv_path: str) -> pd.DataFrame:
        df = pd.read_csv(csv_path)
        if self.target_col not in df.columns:
            raise ValueError(f"Khong tim thay cot target {self.target_col} trong file.")
        self.df_raw_ = df
        return df

    def split_features_target(self, df: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, pd.Series]:
        if df is None:
            if self.df_raw_ is None:
                raise RuntimeError("Chua goi load_data truoc.")
            df = self.df_raw_
        X = df.drop(columns=[self.target_col])
        y = df[self.target_col].astype(float)
        return X, y

    def build_feature_pipeline(self, X_train: pd.DataFrame, X_test: Optional[pd.DataFrame] = None) -> Pipeline:
        self.feature_pipe_ = build_feature_pipeline(X_train, X_test)
        return self.feature_pipe_

    def fit_transform_train(self, X_train: pd.DataFrame, y_train: pd.Series) -> np.ndarray:
        if self.feature_pipe_ is None:
            self.feature_pipe_ = build_feature_pipeline(X_train, X_train)
        X_proc = self.feature_pipe_.fit_transform(X_train, y_train)
        return X_proc

    def transform_new(self, X_new: pd.DataFrame) -> np.ndarray:
        if self.feature_pipe_ is None:
            raise RuntimeError("Pipeline chua duoc fit. Hay goi fit_transform_train truoc.")
        return self.feature_pipe_.transform(X_new)
