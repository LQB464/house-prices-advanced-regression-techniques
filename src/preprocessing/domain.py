"""
preprocessing/domain.py

Domain-specific feature engineering utilities for housing price datasets (Ames-like).

Extended Description
--------------------
This module implements handcrafted domain features commonly used in the Ames Housing
prediction problem, such as total square footage, total bathrooms, age-related features,
porch areas, bathroom ratios, area ratios, cyclical encodings, and interaction terms.
It also provides a sklearn-compatible transformer wrapper, allowing seamless integration
into sklearn Pipelines.

Main Components
---------------
- add_domain_features: Apply domain-specific feature engineering on a DataFrame.
- DomainFeatureAdder: sklearn Transformer wrapper around add_domain_features,
  supporting get_feature_names_out.

Usage Example
-------------
>>> from preprocessing.domain import add_domain_features, DomainFeatureAdder
>>> df2 = add_domain_features(df)
>>> pipe = Pipeline([("domain", DomainFeatureAdder())])
>>> df_new = pipe.fit_transform(df)

Notes
-----
The feature engineering logic is highly customized for Ames-style datasets.
Missing columns are safely handled by auto-filling default values.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError


def add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate domain-specific engineered features for housing price prediction.

    This function applies a variety of handcrafted transformations used in the
    Ames Housing competition, including:
    - total square footage aggregation
    - total bathrooms (with weighting for half baths)
    - age-based features (house age, remodeling age, garage age)
    - porch area sums
    - binary structural flags
    - functional ratios
    - cyclical encoding of sale month
    - composite categorical features
    - mild winsorization of LotArea
    - interaction terms involving OverallQual

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing original Ames Housing features.

    Returns
    -------
    pandas.DataFrame
        A copy of the dataframe with many additional engineered features.

    Examples
    --------
    >>> df2 = add_domain_features(df)
    >>> df2.columns
    ... # includes TotalSF, TotalBath, HouseAge, etc.
    """

    df = df.copy()

    for c in ["TotalBsmtSF", "1stFlrSF", "2ndFlrSF"]:
        if c not in df.columns:
            df[c] = 0
    df["TotalSF"] = (
        df["TotalBsmtSF"].fillna(0)
        + df["1stFlrSF"].fillna(0)
        + df["2ndFlrSF"].fillna(0)
    )

    for c in ["FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath"]:
        if c not in df.columns:
            df[c] = 0
    df["TotalBath"] = (
        df["FullBath"].fillna(0)
        + 0.5 * df["HalfBath"].fillna(0)
        + df["BsmtFullBath"].fillna(0)
        + 0.5 * df["BsmtHalfBath"].fillna(0)
    )

    for c in ["YrSold", "YearBuilt", "YearRemodAdd", "GarageYrBlt"]:
        if c not in df.columns:
            df[c] = np.nan
    df["HouseAge"] = (df["YrSold"] - df["YearBuilt"]).astype(float)
    df["RemodAge"] = (df["YrSold"] - df["YearRemodAdd"]).astype(float)
    df["GarageAge"] = (df["YrSold"] - df["GarageYrBlt"]).astype(float)

    df["IsRemodeled"] = (
        df.get("YearRemodAdd", df["YearBuilt"]) != df["YearBuilt"]
    ).astype(int)
    df["Has2ndFlr"] = (df["2ndFlrSF"] > 0).astype(int)

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

    df["BathPerBedroom"] = df.get("TotalBath", 0) / np.maximum(
        df.get("BedroomAbvGr", 1), 1
    )
    df["RoomsPerArea"] = df.get("TotRmsAbvGrd", 0) / np.maximum(
        df.get("GrLivArea", 1), 1
    )
    df["LotAreaRatio"] = df.get("LotArea", 0) / np.maximum(
        df.get("GrLivArea", 1), 1
    )

    if "MoSold" in df.columns:
        df["MoSold_sin"] = np.sin(2 * np.pi * (df["MoSold"].astype(float) / 12.0))
        df["MoSold_cos"] = np.cos(2 * np.pi * (df["MoSold"].astype(float) / 12.0))

    if ("Neighborhood" in df.columns) and ("BldgType" in df.columns):
        df["Neighborhood_BldgType"] = (
            df["Neighborhood"].astype(str) + "|" + df["BldgType"].astype(str)
        )

    df["Ln_TotalSF"] = np.log1p(df.get("TotalSF", 0).astype(float))

    if "OverallQual" in df.columns:
        df["IQ_OQ_GrLiv"] = df["OverallQual"].astype(float) * df.get(
            "GrLivArea", 0
        ).astype(float)
        df["IQ_OQ_TotalSF"] = df["OverallQual"].astype(float) * df.get(
            "TotalSF", 0
        ).astype(float)

    if "LotArea" in df.columns:
        q_hi = df["LotArea"].quantile(0.99)
        df["LotArea_clip"] = np.minimum(df["LotArea"], q_hi)

    return df



class DomainFeatureAdder(BaseEstimator, TransformerMixin):
    """
    sklearn-compatible transformer that applies add_domain_features.

    Extended Description
    --------------------
    This transformer wraps `add_domain_features` to enable its integration inside
    sklearn Pipelines. It preserves column names and exposes consistent
    get_feature_names_out behavior so that downstream components such as selectors
    or model explainability tools can reference the expanded feature set.

    Attributes
    ----------
    feature_names_in_ : ndarray of str
        Original column names passed to fit().
    feature_names_out_ : ndarray of str
        Column names after domain feature expansion.

    Examples
    --------
    >>> t = DomainFeatureAdder()
    >>> X2 = t.fit_transform(df)
    >>> t.get_feature_names_out()
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
            raise NotFittedError("DomainFeatureAdder has not been fitted.")
        return self.feature_names_out_


__all__ = ["add_domain_features", "DomainFeatureAdder"]
