"""
preprocessing/columns.py

Utilities for detecting column types and building sklearn ColumnTransformer pipelines.

Extended Description
--------------------
This module provides helper functions to automatically separate dataframe columns
into ordinal, categorical, and numerical groups. It also builds a standardized
ColumnTransformer that handles imputing, scaling, and one-hot encoding based on
column types. This helps ensure consistent preprocessing logic across pipelines
and reduces boilerplate code when working with tabular data.

Main Components
---------------
- build_feature_lists: Identify ordinal, categorical, and numeric columns.
- make_column_transformer: Build a ColumnTransformer with appropriate pipelines
  for numeric, categorical, and ordinal columns.

Usage Example
-------------
>>> from preprocessing.columns import build_feature_lists, make_column_transformer
>>> ord_cols, cat_cols, num_cols = build_feature_lists(df, ordinal_cols=["Qual"])
>>> ct = make_column_transformer(df, ordinal_cols=["Qual"])
>>> X_transformed = ct.fit_transform(df)

Notes
-----
The implementation uses sklearn Pipelines and transformers such as SimpleImputer,
StandardScaler, and OneHotEncoder. Column ordering after transformation follows
sklearn’s default behavior.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


def build_feature_lists(
    df: pd.DataFrame,
    ordinal_cols: Optional[List[str]] = None,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Split dataframe columns into ordinal, categorical, and numeric groups.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset from which the column types are inferred.
    ordinal_cols : list of str, optional
        List of user-defined ordinal column names. Only columns present in `df`
        will be kept.

    Returns
    -------
    tuple of (list, list, list)
        A tuple containing:
        - ord_cols : list of ordinal column names detected in `df`
        - cat_cols : list of non-ordinal categorical column names
        - num_cols : list of numeric column names

    Examples
    --------
    >>> ord_cols, cat_cols, num_cols = build_feature_lists(df, ["Qual", "Cond"])
    >>> ord_cols
    ['Qual', 'Cond']
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


def make_column_transformer(
    df: pd.DataFrame,
    ordinal_cols: Optional[List[str]] = None,
) -> ColumnTransformer:
    """
    Build an sklearn ColumnTransformer for numeric, categorical, and ordinal columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset used to infer column groupings.
    ordinal_cols : list of str, optional
        List of ordinal columns that will undergo only imputation.

    Returns
    -------
    sklearn.compose.ColumnTransformer
        A fully constructed ColumnTransformer containing:
        - categorical pipeline: imputation + one-hot encoding
        - ordinal pipeline: imputation only
        - numeric pipeline: imputation + scaling

    Notes
    -----
    - Unknown categories in categorical columns are ignored during transform.
    - Numeric columns are scaled using StandardScaler.
    - Ordinal columns are not encoded or scaled by default.

    Examples
    --------
    >>> ct = make_column_transformer(df, ["ExterQual", "KitchenQual"])
    >>> X_processed = ct.fit_transform(df)
    """
    
    ord_cols, cat_cols, num_cols = build_feature_lists(df, ordinal_cols)
    num_cols = [c for c in num_cols if c not in ord_cols]

    cat_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    ord_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
        ]
    )

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


__all__ = ["build_feature_lists", "make_column_transformer"]