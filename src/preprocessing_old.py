"""
preprocessing.py

Tiền xử lý dữ liệu cho đồ án Ames House Prices (và các bài tabular tương tự).

- Hỗ trợ:
  + Ordinal mapping theo ORDINAL_MAP_CANONICAL (dict[str, list[str]] hoặc dict[str, dict[str,int]])
  + Thêm domain features (add_domain_features)
  + Xử lý missing, rare category, outlier, scale, one-hot
  + Target Encoding (tùy chọn)
  + Gói trong Preprocessor để dùng tiện trong phần model.

Sử dụng mẫu:

    from preprocessing import Preprocessor, ORDINAL_MAP_CANONICAL

    # 1. Khởi tạo
    prep = Preprocessor(
        target_col="SalePrice",
        ordinal_mapping=ORDINAL_MAP_CANONICAL,
        id_cols=["Id"],
        use_domain_features=True,
        use_target_encoding=False,
    )

    # 2. Train
    df_train = prep.load_data("../Data/train.csv")
    X_train, y_train = prep.split_features_target(df_train)
    prep.build_feature_pipeline(X_train)
    X_train_proc = prep.fit_transform(X_train, y_train)

    # 3. Save log
    prep.save_log("preprocessing.log")

"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Mapping, Iterable, Any

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.exceptions import NotFittedError
from sklearn.feature_selection import SelectKBest, mutual_info_regression


ORDINAL_MAP_CANONICAL: Dict[str, List[str]] = {
    "ExterQual": ["Po", "Fa", "TA", "Gd", "Ex"],
    "ExterCond": ["Po", "Fa", "TA", "Gd", "Ex"],
    "BsmtQual": ["NA", "Po", "Fa", "TA", "Gd", "Ex"],
    "BsmtCond": ["NA", "Po", "Fa", "TA", "Gd", "Ex"],
    "BsmtExposure": ["NA", "No", "Mn", "Av", "Gd"],
    "BsmtFinType1": ["NA", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    "BsmtFinType2": ["NA", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    "HeatingQC": ["Po", "Fa", "TA", "Gd", "Ex"],
    "KitchenQual": ["Po", "Fa", "TA", "Gd", "Ex"],
    "FireplaceQu": ["NA", "Po", "Fa", "TA", "Gd", "Ex"],
    "GarageFinish": ["NA", "Unf", "RFn", "Fin"],
    "GarageQual": ["NA", "Po", "Fa", "TA", "Gd", "Ex"],
    "GarageCond": ["NA", "Po", "Fa", "TA", "Gd", "Ex"],
    "PoolQC": ["NA", "Fa", "TA", "Gd", "Ex"],
    "Fence": ["NA", "MnWw", "GdWo", "MnPrv", "GdPrv"],
    "Functional": ["Sal", "Sev", "Maj2", "Maj1", "Mod", "Min2", "Min1", "Typ"],
    "PavedDrive": ["N", "P", "Y"],
    "Street": ["Grvl", "Pave"],
    "Alley": ["NA", "Grvl", "Pave"],
    "CentralAir": ["N", "Y"],
}


# ============================================================
# 1. Các transformer cơ bản
# ============================================================

class DomainFeatureAdder(BaseEstimator, TransformerMixin):
    """
    Thêm domain features cho Ames (bọc hàm add_domain_features)
    nhưng có hỗ trợ get_feature_names_out.
    """

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        # Lưu tên cột input
        self.feature_names_in_ = np.asarray(X_df.columns, dtype=object)
        # Chạy thử 1 lần để lấy tên cột output
        X_out = add_domain_features(X_df)
        self.feature_names_out_ = np.asarray(X_out.columns, dtype=object)
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X, columns=self.feature_names_in_)
        X_out = add_domain_features(X_df)
        return X_out

    def get_feature_names_out(self, input_features=None):
        if not hasattr(self, "feature_names_out_"):
            raise NotFittedError("DomainFeatureAdder chưa được fit.")
        return self.feature_names_out_


class OrdinalMapper(BaseEstimator, TransformerMixin):
    """
    Ánh xạ các cột ordinal theo dict mapping -> số.

    mapping có thể là:
        - Dict[str, List[str]]  (danh sách value theo thứ tự)
        - Dict[str, Dict[str, int/float]]
    """

    def __init__(self, mapping: Optional[Mapping[str, Any]] = None):
        self.mapping_raw = mapping or {}
        self.mapping_: Dict[str, Dict[str, float]] = {}
        self.cols_: List[str] = []

    @staticmethod
    def _canon_to_numeric_map(mapping: Mapping[str, Any]) -> Dict[str, Dict[str, float]]:
        """
        Chuyển:
            - Dict[str, List[str]]  (canonical order)
        hoặc
            - Dict[str, Dict[str, int]]
        thành:
            Dict[str, Dict[str, float]] dùng cho OrdinalMapper.
        """
        final: Dict[str, Dict[str, float]] = {}
        for col, spec in mapping.items():
            if isinstance(spec, dict):
                final[col] = {k: float(v) for k, v in spec.items()}
            else:
                # giả định là iterable các level theo thứ tự
                levels: Iterable[Any] = spec
                final[col] = {v: float(i) for i, v in enumerate(levels, start=1)}
        return final

    def fit(self, X: pd.DataFrame, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        # chuẩn hoá mapping sang dict-of-dict numeric
        numeric_map = self._canon_to_numeric_map(self.mapping_raw)
        self.mapping_ = {
            col: mp for col, mp in numeric_map.items() if col in X.columns
        }
        self.cols_ = list(self.mapping_.keys())
        return self

    def transform(self, X: pd.DataFrame):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        X = X.copy()
        for col in self.cols_:
            mp = self.mapping_[col]
            # map -> float, giá trị lạ sẽ thành NaN
            X[col] = X[col].map(mp).astype(float)
        return X

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = getattr(self, "feature_names_in_", None)
        if input_features is None:
            raise NotFittedError("OrdinalMapper chưa được fit.")
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

    - Áp dụng trên X sau khi đã qua toàn bộ preprocessing.
    - Phù hợp khi số feature lớn, cần giảm nhiễu.

    Parameters
    ----------
    k : int
        Số lượng feature giữ lại.
    random_state : int, default=0
        Seed cho mutual_info_regression.
    """

    def __init__(self, k: int = 100, random_state: int = 0):
        self.k = int(k)
        self.random_state = int(random_state)
        self.selector_: Optional[SelectKBest] = None

    def fit(self, X, y):
        if y is None:
            raise ValueError("KBestMutualInfoSelector cần y để fit.")
        X_arr = np.asarray(X)
        y_arr = np.asarray(y)

        # bọc mutual_info_regression để truyền random_state
        def _mi(X_, y_):
            return mutual_info_regression(
                X_, y_, random_state=self.random_state
            )

        self.selector_ = SelectKBest(score_func=_mi, k=self.k)
        self.selector_.fit(X_arr, y_arr)
        return self

    def transform(self, X):
        if self.selector_ is None:
            raise NotFittedError("KBestMutualInfoSelector chưa được fit.")
        X_arr = np.asarray(X)
        return self.selector_.transform(X_arr)

    def get_feature_names_out(self, input_features=None):
        if self.selector_ is None:
            raise NotFittedError("KBestMutualInfoSelector chưa được fit.")
        if input_features is None:
            input_features = getattr(self, "feature_names_in_", None)
        if input_features is None:
            raise NotFittedError(
                "KBestMutualInfoSelector không có thông tin input_features."
            )

        input_features = np.asarray(input_features, dtype=object)
        mask = self.selector_.get_support()
        return input_features[mask]



# ============================================================
# 2. Hàm tiện ích: tách list cột và tạo ColumnTransformer
# ============================================================

def build_feature_lists(
    df: pd.DataFrame,
    ordinal_cols: Optional[List[str]] = None,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Tách danh sách cột theo loại:
        - ord_cols: các cột ordinal (đã biết trước)
        - cat_cols: còn lại kiểu object/category
        - num_cols: các cột numeric
    """
    df = df.copy()
    ordinal_cols = ordinal_cols or []

    ord_cols = [c for c in ordinal_cols if c in df.columns]
    cat_cols = [
        c
        for c in df.select_dtypes(include=["object", "category"]).columns
        if c not in ord_cols
    ]
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return ord_cols, cat_cols, num_cols


def add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Thêm một số domain feature cho bài Ames:
        - TotalSF, TotalBath, HouseAge, RemodAge, GarageAge, ...
    Có thể gọi qua FunctionTransformer trong pipeline.
    """
    df = df.copy()

    # Diện tích tổng
    for c in ["TotalBsmtSF", "1stFlrSF", "2ndFlrSF"]:
        if c not in df.columns:
            df[c] = 0
    df["TotalSF"] = (
        df["TotalBsmtSF"].fillna(0)
        + df["1stFlrSF"].fillna(0)
        + df["2ndFlrSF"].fillna(0)
    )

    # Tổng số phòng tắm (full + half)
    for c in ["FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath"]:
        if c not in df.columns:
            df[c] = 0
    df["TotalBath"] = (
        df["FullBath"].fillna(0)
        + 0.5 * df["HalfBath"].fillna(0)
        + df["BsmtFullBath"].fillna(0)
        + 0.5 * df["BsmtHalfBath"].fillna(0)
    )

    # Tuổi nhà, tuổi sửa, tuổi garage
    for c in ["YrSold", "YearBuilt", "YearRemodAdd", "GarageYrBlt"]:
        if c not in df.columns:
            df[c] = np.nan
    df["HouseAge"] = (df["YrSold"] - df["YearBuilt"]).astype(float)
    df["RemodAge"] = (df["YrSold"] - df["YearRemodAdd"]).astype(float)
    df["GarageAge"] = (df["YrSold"] - df["GarageYrBlt"]).astype(float)

    # Cờ remodeled, có tầng 2 không
    df["IsRemodeled"] = (
        df.get("YearRemodAdd", df["YearBuilt"]) != df["YearBuilt"]
    ).astype(int)
    df["Has2ndFlr"] = (df["2ndFlrSF"] > 0).astype(int)

    # Tổng diện tích porch/deck
    for c in [
        "OpenPorchSF",
        "EnclosedPorch",
        "3SsnPorch",
        "ScreenPorch",
        "WoodDeckSF",
    ]:
        if c not in df.columns:
            df[c] = 0
    df["TotalPorchSF"] = (
        df["OpenPorchSF"]
        + df["EnclosedPorch"]
        + df["3SsnPorch"]
        + df["ScreenPorch"]
        + df["WoodDeckSF"]
    )

    # Một vài tỷ lệ
    df["BathPerBedroom"] = df.get("TotalBath", 0) / np.maximum(
        df.get("BedroomAbvGr", 1), 1
    )
    df["RoomsPerArea"] = df.get("TotRmsAbvGrd", 0) / np.maximum(
        df.get("GrLivArea", 1), 1
    )
    df["LotAreaRatio"] = df.get("LotArea", 0) / np.maximum(
        df.get("GrLivArea", 1), 1
    )

    # Mã hoá chu kỳ cho tháng bán
    if "MoSold" in df.columns:
        df["MoSold_sin"] = np.sin(2 * np.pi * (df["MoSold"].astype(float) / 12.0))
        df["MoSold_cos"] = np.cos(2 * np.pi * (df["MoSold"].astype(float) / 12.0))

    # Feature kết hợp khu vực + loại nhà
    if ("Neighborhood" in df.columns) and ("BldgType" in df.columns):
        df["Neighborhood_BldgType"] = (
            df["Neighborhood"].astype(str) + "|" + df["BldgType"].astype(str)
        )

    # Log TotalSF
    df["Ln_TotalSF"] = np.log1p(df.get("TotalSF", 0).astype(float))

    # Tương tác OverallQual với diện tích
    if "OverallQual" in df.columns:
        df["IQ_OQ_GrLiv"] = df["OverallQual"].astype(float) * df.get(
            "GrLivArea", 0
        ).astype(float)
        df["IQ_OQ_TotalSF"] = df["OverallQual"].astype(float) * df.get(
            "TotalSF", 0
        ).astype(float)

    # Winsorize nhẹ LotArea
    if "LotArea" in df.columns:
        q_hi = df["LotArea"].quantile(0.99)
        df["LotArea_clip"] = np.minimum(df["LotArea"], q_hi)

    return df


def make_column_transformer(
    df: pd.DataFrame,
    ordinal_cols: Optional[List[str]] = None,
) -> ColumnTransformer:
    """
    Tạo ColumnTransformer cho dữ liệu tabular.

    - Numeric: impute median + StandardScaler
    - Categorical: impute most_frequent + OneHot
    - Ordinal: chỉ impute most_frequent (đã map sang số trước đó)
    """
    ord_cols, cat_cols, num_cols = build_feature_lists(df, ordinal_cols)

    # đề phòng trùng cột
    num_cols = [c for c in num_cols if c not in ord_cols]

    # pipeline cho categorical
    cat_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    # pipeline cho ordinal (giờ đã là numeric sau OrdinalMapper)
    ord_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
        ]
    )

    # pipeline cho numeric
    num_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )

    preproc = ColumnTransformer(
        transformers=[
            ("cats", cat_pipeline, cat_cols),
            ("ords", ord_pipeline, ord_cols),
            ("nums", num_pipeline, num_cols),
        ],
        remainder="drop",
    )
    return preproc


# ============================================================
# 3. Pipeline tổng cho features
# ============================================================

def build_feature_pipeline(
    df_train: pd.DataFrame,
    ordinal_mapping: Optional[Mapping[str, Any]] = None,
    use_domain_features: bool = True,
    use_target_encoding: bool = False,
    target_enc_cols: Optional[List[str]] = None,
    # feature selection
    enable_variance_selector: bool = True,
    variance_threshold: float = 0.0,
    enable_kbest_mi: bool = False,
    k_best_features: int = 150,
    mi_random_state: int = 0,
) -> Pipeline:
    """
    Tạo sklearn Pipeline tiền xử lý đầy đủ:

    0. (optional) FunctionTransformer(add_domain_features)
    1. (optional) OrdinalMapper nếu có ordinal_mapping
    2. MissingnessIndicator cho numeric
    3. RareCategoryGrouper cho categorical
    4. (optional) TargetEncoderTransformer
    5. OutlierClipper cho numeric
    6. FiniteCleaner
    7. ColumnTransformer: scale numeric, one hot, v.v.
    8. DropAllNaNColumns
    9. (optional) VarianceFeatureSelector
    10. (optional) KBestMutualInfoSelector

    df_train chỉ dùng để suy ra loại cột, boundary IQR, v.v.
    """

    # chuẩn hóa mapping và lấy danh sách cột ordinal
    ordinal_mapping = ordinal_mapping or {}
    ordinal_cols = list(ordinal_mapping.keys()) if ordinal_mapping else []

    # nếu dùng domain features thì build ColumnTransformer trên df_train đã add domain
    df_for_cols = add_domain_features(df_train) if use_domain_features else df_train
    preproc_cols = make_column_transformer(df_for_cols, ordinal_cols=ordinal_cols)

    steps: List[Tuple[str, Any]] = []

    # 0) thêm domain features
    if use_domain_features:
        steps.append(("domain", DomainFeatureAdder()))

    # 1) ordinal mapping
    if ordinal_mapping:
        steps.append(("ordinal_map", OrdinalMapper(mapping=ordinal_mapping)))

    # 2) indicator missing
    steps.append(("missing_indicator", MissingnessIndicator()))

    # 3) gộp rare category
    steps.append(("rare_grouper", RareCategoryGrouper(min_freq=20)))

    # 4) target encoding
    if use_target_encoding:
        # nếu không truyền, dùng một vài cột phổ biến ở Ames
        if target_enc_cols is None:
            target_enc_cols = [
                "Neighborhood",
                "MSZoning",
                "Exterior1st",
                "Exterior2nd",
                "SaleType",
                "SaleCondition",
            ]
        steps.append(("target_encoder", TargetEncoderTransformer(cols=target_enc_cols)))

    # 5, 6, 7, 8
    steps.extend(
        [
            ("outlier_clip", OutlierClipper(factor=1.5)),
            ("finite_clean", FiniteCleaner()),
            ("col_transform", preproc_cols),
            ("drop_all_nan", DropAllNaNColumns()),
        ]
    )

    # 9) selector theo variance
    if enable_variance_selector:
        steps.append(
            ("var_selector", VarianceFeatureSelector(threshold=variance_threshold))
        )

    # 10) selector theo mutual information
    if enable_kbest_mi and k_best_features is not None:
        steps.append(
            (
                "mi_selector",
                KBestMutualInfoSelector(
                    k=int(k_best_features),
                    random_state=int(mi_random_state),
                ),
            )
        )

    pipe = Pipeline(steps=steps)
    return pipe


# ============================================================
# 4. Lớp cấp cao dùng trong phần Model
# ============================================================

@dataclass
class Preprocessor:
    """
    Lớp tiền xử lý dữ liệu cấp cao.

    Chức năng chính:
    - load_data: đọc csv/xlsx/json bằng pandas
    - split_features_target: tách X, y
    - build_feature_pipeline: tạo sklearn Pipeline cho X
    - fit_transform: fit pipeline trên tập train và trả về X_processed
    - transform_new: biến đổi tập dữ liệu mới (val/test)

    Parameters
    ----------
    target_col : str
        Tên cột target (label).
    ordinal_mapping : dict, optional
        mapping cho các cột ordinal:
            - có thể là Dict[str, List[str]] (canonical)
            - hoặc Dict[str, Dict[str, int/float]]
    id_cols : list, optional
        Các cột id cần loại bỏ khỏi X.
    use_domain_features : bool, default True
        Có thêm domain features (TotalSF, TotalBath, ...) hay không.
    use_target_encoding : bool, default False
        Có dùng Target Encoding cho một số cột categorical hay không.
    target_enc_cols : list, optional
        Danh sách cột áp dụng Target Encoding (nếu None sẽ dùng default).
    """

    target_col: Optional[str] = None
    ordinal_mapping: Optional[Mapping[str, Any]] = None
    id_cols: Optional[List[str]] = None
    use_domain_features: bool = True
    use_target_encoding: bool = False
    target_enc_cols: Optional[List[str]] = None

    # >>> thêm các tham số điều khiển feature selection <<<
    enable_variance_selector: bool = True
    variance_threshold: float = 0.0
    enable_kbest_mi: bool = False
    k_best_features: int = 150
    mi_random_state: int = 0

    # các thuộc tính được set sau
    df_raw_: Optional[pd.DataFrame] = None
    feature_pipe_: Optional[Pipeline] = None
    logs: List[str] = field(default_factory=list)

    # ----------------- phương thức tiện ích -----------------

    @staticmethod
    def _detect_file_type(path: str) -> str:
        path = path.lower()
        if path.endswith(".csv"):
            return "csv"
        if path.endswith(".xlsx") or path.endswith(".xls"):
            return "excel"
        if path.endswith(".json"):
            return "json"
        raise ValueError(
            "Không nhận diện được loại file. Hỗ trợ: .csv, .xlsx/.xls, .json"
        )

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
        """Ghi 1 dòng log vào bộ nhớ."""
        self.logs.append(msg)

    # ----------------- I/O -----------------

    def load_data(self, path: str, **read_kwargs) -> pd.DataFrame:
        """
        Đọc dữ liệu từ csv/xlsx/json bằng pandas.
        """
        ftype = self._detect_file_type(path)
        try:
            if ftype == "csv":
                df = pd.read_csv(path, **read_kwargs)
            elif ftype == "excel":
                df = pd.read_excel(path, **read_kwargs)
            else:  # json
                df = pd.read_json(path, **read_kwargs)
        except Exception as e:
            err_msg = f"[load_data] Lỗi đọc file {path}: {e}"
            self._log(err_msg)
            raise IOError(f"Không thể đọc file {path}: {e}") from e

        self.df_raw_ = df
        self._log(f"[load_data] Đã đọc {path} với shape={df.shape}")
        return df

    # ----------------- xử lý X, y -----------------

    def _drop_id_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.id_cols:
            return df
        cols_to_drop = [c for c in self.id_cols if c in df.columns]
        return df.drop(columns=cols_to_drop, errors="ignore")

    def split_features_target(
        self, df: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Tách X, y từ DataFrame.

        Nếu target_col = None -> chỉ trả về X, y = None (dùng cho test set).
        """
        if df is None:
            if self.df_raw_ is None:
                raise RuntimeError("Chưa gọi load_data hoặc chưa truyền df.")
            df = self.df_raw_

        df = df.copy()
        df = self._drop_id_columns(df)

        if self.target_col is None:
            return df, None

        if self.target_col not in df.columns:
            raise ValueError(
                f"Không tìm thấy cột target '{self.target_col}' trong DataFrame."
            )

        X = df.drop(columns=[self.target_col])
        y = df[self.target_col]
        
        if self.target_col is None:
            self._log(f"[split_features_target] Không có target_col, chỉ trả về X với shape={df.shape}")
            return df, None
        
        self._log(
            f"[split_features_target] X shape={X.shape}, y len={len(y)}, target={self.target_col}"
        )
        return X, y

    # ----------------- pipeline -----------------

    def build_feature_pipeline(self, X_train: pd.DataFrame) -> Pipeline:
        """
        Xây dựng pipeline dựa trên X_train, dùng các tham số của chính Preprocessor.
        """
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
            "[build_feature_pipeline] Pipeline được tạo với "
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
        """
        Fit pipeline trên tập train và trả về X_train_processed.
        """
        if self.feature_pipe_ is None:
            self.build_feature_pipeline(X_train)
        X_proc = self.feature_pipe_.fit_transform(X_train, y_train)
        self._log(
            f"[fit_transform] Đã fit pipeline trên X_train shape={X_train.shape}, "
            f"y_train len={len(y_train) if y_train is not None else 'None'}, "
            f"kết quả shape={X_proc.shape}"
        )
        return X_proc

    def transform_new(self, X_new: pd.DataFrame) -> np.ndarray:
        """
        Biến đổi tập dữ liệu mới (val/test) bằng pipeline đã fit.
        """
        if self.feature_pipe_ is None:
            msg = "Pipeline chưa được fit. Hãy gọi fit_transform trước khi transform_new."
            self._log(f"[transform_new] ERROR: {msg}")
            raise RuntimeError(msg)
        X_proc = self.feature_pipe_.transform(X_new)
        self._log(
            f"[transform_new] Biến đổi X_new shape={X_new.shape} -> shape={X_proc.shape}"
        )
        return X_proc
    
    def save_log(self, path: str = "preprocessing.log") -> None:
        """Ghi toàn bộ log ra file text."""
        with open(path, "w", encoding="utf-8") as f:
            for line in self.logs:
                f.write(line + "\n")

    def get_log_df(self) -> pd.DataFrame:
        """Trả về log ở dạng DataFrame để dễ xem / lưu CSV."""
        return pd.DataFrame({"step": range(len(self.logs)), "message": self.logs})
    
    def get_feature_names_out(self) -> np.ndarray:
        """
        Trả về tên các feature sau toàn bộ pipeline.
        Yêu cầu self.feature_pipe_ đã được fit.
        """
        if self.feature_pipe_ is None:
            raise RuntimeError(
                "feature_pipe_ chưa được tạo. Hãy gọi build_feature_pipeline / fit_transform trước."
            )
        try:
            return self.feature_pipe_.get_feature_names_out()
        except AttributeError as e:
            raise NotFittedError(
                "Pipeline chưa hỗ trợ get_feature_names_out đầy đủ "
                "(có thể một số step chưa cài đặt)."
            ) from e
            
            
__all__ = [
    build_feature_pipeline,
    Preprocessor,
]