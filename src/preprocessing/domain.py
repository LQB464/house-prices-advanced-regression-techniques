# preprocessing/domain.py
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError


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



class DomainFeatureAdder(BaseEstimator, TransformerMixin):
    """
    Transformer bọc add_domain_features, hỗ trợ get_feature_names_out.
    """

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        self.feature_names_in_ = X_df.columns.to_numpy(dtype=object)
        X_out = add_domain_features(X_df)
        self.feature_names_out_ = X_out.columns.to_numpy(dtype=object)
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X, columns=self.feature_names_in_)
        X_out = add_domain_features(X_df)
        return X_out

    def get_feature_names_out(self, input_features=None):
        if not hasattr(self, "feature_names_out_"):
            raise NotFittedError("DomainFeatureAdder chưa được fit.")
        return self.feature_names_out_


__all__ = ["add_domain_features", "DomainFeatureAdder"]
